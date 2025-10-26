using Pkg
Pkg.activate(".")
Pkg.add("Revise")

using Revise

Pkg.develop(path = "../")

#Pkg.add(["JLD2", "Flux", "CannotWaitForTheseOptimisers", "LearningSchedules", "DLProteinFormats"])
#Pkg.add(["CUDA", "cuDNN"])

#Pkg.develop(path="../../ForwardBackward.jl/")
#Pkg.develop(path="../../Flowfusion.jl/")
#Pkg.develop(path="../../ChainStorm.jl/")

using DLProteinFormats, Flux, CannotWaitForTheseOptimisers, LearningSchedules, JLD2, Dates, BatchedTransformations, ProteinChains
using BranchingFlows, Flowfusion, Distributions, ForwardBackward, RandomFeatureMaps, InvariantPointAttention, Onion, StatsBase, Random
using DLProteinFormats: load, PDBSimpleFlat, batch_flatrecs, sample_batched_inds, length2batch
using CUDA

device!(1) #Because we have set CUDA_VISIBLE_DEVICES = GPUnum
device = gpu

rundir = "runs/condchain_20X0_$(Date(now()))_$(rand(100000:999999))"
mkpath("$(rundir)/samples")

X0_mean_length = 20
deletion_pad = 1.3
per_chain_upper_X0_len = 1 + quantile(Poisson(X0_mean_length), 0.95)

function textlog(filepath::String, l; also_print = true)
    f = open(filepath,"a")
        write(f, join(string.(l),", "))
        write(f, "\n")
    close(f)
    if also_print
        println(join(string.(l),", "))
    end
end

oldAdaLN(dim,cond_dim) = AdaLN(Flux.LayerNorm(dim), Dense(cond_dim, dim), Dense(cond_dim, dim))

ipa(l, f, x, pf, c, m) = l(f, x, pair_feats = pf, cond = c, mask = m)
crossipa(l, f1, f2, x, pf, c, m) = l(f1, f2, x, pair_feats = pf, cond = c, mask = m)

struct BranchChainV1{L}
    layers::L
end
Flux.@layer BranchChainV1
function BranchChainV1(dim::Int = 384, depth::Int = 6, f_depth::Int = 6)
    layers = (;
        depth = depth,
        f_depth = f_depth,
        t_rff = RandomFourierFeatures(1 => dim, 1f0),
        cond_t_encoding = Dense(dim => dim, bias=false),
        AApre_t_encoding = Dense(dim => dim, bias=false),
        pair_rff = RandomFourierFeatures(2 => 64, 1f0),
        pair_project = Dense(64 => 32, bias=false),
        AAencoder = Dense(21 => dim, bias=false),
        selfcond_crossipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = oldAdaLN(dim, dim)) for _ in 1:depth],
        selfcond_selfipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = oldAdaLN(dim, dim)) for _ in 1:depth],
        ipa_blocks = [IPAblock(dim, IPA(IPA_settings(dim, c_z = 32)), ln1 = oldAdaLN(dim, dim), ln2 = oldAdaLN(dim, dim)) for _ in 1:depth],
        framemovers = [Framemover(dim) for _ in 1:f_depth],
        AAdecoder = Chain(StarGLU(dim, 3dim), Dense(dim => 21, bias=false)),
        #New layesr to tune-in
        indelpre_t_encoding = Dense(dim => 3dim),
        count_decoder = StarGLU(Dense(3dim => 2dim, bias=false), Dense(2dim => 1, bias=false), Dense(3dim => 2dim, bias=false), Flux.swish),
        del_decoder   = StarGLU(Dense(3dim => 2dim, bias=false), Dense(2dim => 1, bias=false), Dense(3dim => 2dim, bias=false), Flux.swish)
    )
    return BranchChainV1(layers)
end
function (fc::BranchChainV1)(t, Xt, chainids, resinds; sc_frames = nothing)
    l = fc.layers
    pmask = Flux.Zygote.@ignore self_att_padding_mask(Xt[1].lmask)
    pre_z = Flux.Zygote.@ignore l.pair_rff(pair_encode(resinds, chainids))
    pair_feats = l.pair_project(pre_z)
    t_rff = Flux.Zygote.@ignore l.t_rff(t)
    cond = reshape(l.cond_t_encoding(t_rff), :, 1, size(t,2))
    frames = Translation(tensor(Xt[1])) ∘ Rotation(tensor(Xt[2]))
    AA_one_hots = tensor(Xt[3])
    x = l.AAencoder(AA_one_hots .+ 0)
    x_a,x_b,x_c = nothing, nothing, nothing
    for i in 1:l.depth
        if sc_frames !== nothing
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_selfipa[i], sc_frames, sc_frames, x, pair_feats, cond, pmask)
            f1, f2 = mod(i, 2) == 0 ? (frames, sc_frames) : (sc_frames, frames)
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_crossipa[i], f1, f2, x, pair_feats, cond, pmask)
        end
        x = Flux.Zygote.checkpointed(ipa, l.ipa_blocks[i], frames, x, pair_feats, cond, pmask)
        if i > l.depth - l.f_depth
            frames = l.framemovers[i - l.depth + l.f_depth](frames, x, t = t)
        end
        if i==4 (x_a = x) end
        if i==5 (x_b = x) end
        if i==6 (x_c = x) end
    end
    aa_logits = l.AAdecoder(x .+ reshape(l.AApre_t_encoding(t_rff), :, 1, size(t,2)))
    #New:
    catted = vcat(x_a,x_b,x_c)
    indel_pre_t = reshape(l.indelpre_t_encoding(t_rff), :, 1, size(t,2))
    count_log = reshape(l.count_decoder(catted .+ indel_pre_t),  :, length(t))
    del_logits = reshape(l.del_decoder(catted .+ indel_pre_t),  :, length(t))
    return frames, aa_logits, count_log, del_logits
end

struct CondBranchChainV1{L}
    layers::L
end
Flux.@layer CondBranchChainV1
function CondBranchChainV1(dim::Int = 384, depth::Int = 6, f_depth::Int = 6)
    layers = (;
        depth = depth,
        f_depth = f_depth,
        mask_embedder = Embedding(2 => dim),
        break_embedder = Embedding(2 => dim),
        t_rff = RandomFourierFeatures(1 => dim, 1f0),
        cond_t_encoding = Dense(dim => dim, bias=false),
        AApre_t_encoding = Dense(dim => dim, bias=false),
        pair_rff = RandomFourierFeatures(2 => 64, 1f0),
        pair_project = Dense(64 => 32, bias=false),
        AA_embedder = Embedding(21 => dim),
        selfcond_crossipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = oldAdaLN(dim, dim)) for _ in 1:depth],
        selfcond_selfipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = oldAdaLN(dim, dim)) for _ in 1:depth],
        ipa_blocks = [IPAblock(dim, IPA(IPA_settings(dim, c_z = 32)), ln1 = oldAdaLN(dim, dim), ln2 = oldAdaLN(dim, dim)) for _ in 1:depth],
        framemovers = [Framemover(dim) for _ in 1:f_depth],
        AAdecoder = Chain(StarGLU(dim, 3dim), Dense(dim => 21, bias=false)),
        #New layesr to tune-in
        indelpre_t_encoding = Dense(dim => 3dim),
        count_decoder = StarGLU(Dense(3dim => 2dim, bias=false), Dense(2dim => 1, bias=false), Dense(3dim => 2dim, bias=false), Flux.swish),
        del_decoder   = StarGLU(Dense(3dim => 2dim, bias=false), Dense(2dim => 1, bias=false), Dense(3dim => 2dim, bias=false), Flux.swish)
    )
    return CondBranchChainV1(layers)
end
function (fc::CondBranchChainV1)(t, BSXt, chainids, resinds, breaks; sc_frames = nothing)
    l = fc.layers
    Xt = BSXt.state
    cmask = BSXt.flowmask
    pmask = Flux.Zygote.@ignore self_att_padding_mask(BSXt.padmask)
    pre_z = Flux.Zygote.@ignore l.pair_rff(pair_encode(resinds, chainids))
    pair_feats = l.pair_project(pre_z)
    t_rff = Flux.Zygote.@ignore l.t_rff(t)
    cond = reshape(l.cond_t_encoding(t_rff), :, 1, size(t,2))
    frames = Translation(tensor(Xt[1])) ∘ Rotation(tensor(Xt[2]))
    x = l.AA_embedder(tensor(Xt[3])) .+ l.mask_embedder(cmask .+ 1) .+ reshape(l.break_embedder(breaks .+ 1), :, 1, size(t,2))
    x_a,x_b,x_c = nothing, nothing, nothing
    for i in 1:l.depth
        if sc_frames !== nothing
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_selfipa[i], sc_frames, sc_frames, x, pair_feats, cond, pmask)
            f1, f2 = mod(i, 2) == 0 ? (frames, sc_frames) : (sc_frames, frames)
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_crossipa[i], f1, f2, x, pair_feats, cond, pmask)
        end
        x = Flux.Zygote.checkpointed(ipa, l.ipa_blocks[i], frames, x, pair_feats, cond, pmask)
        if i > l.depth - l.f_depth
            frames = l.framemovers[i - l.depth + l.f_depth](frames, x, t = 1 .- (1 .- t .* 0.95f0).*cmask)
        end
        if i==4 (x_a = x) end
        if i==5 (x_b = x) end
        if i==6 (x_c = x) end
    end
    aa_logits = l.AAdecoder(x .+ reshape(l.AApre_t_encoding(t_rff), :, 1, size(t,2)))
    catted = vcat(x_a,x_b,x_c)
    indel_pre_t = reshape(l.indelpre_t_encoding(t_rff), :, 1, size(t,2))
    count_log = reshape(l.count_decoder(catted .+ indel_pre_t),  :, length(t))
    del_logits = reshape(l.del_decoder(catted .+ indel_pre_t),  :, length(t))
    return frames, aa_logits, count_log, del_logits
end


P = CoalescentFlow(((OUBridgeExpVar(100f0, 150f0, 0.000000001f0, dec = -3f0), 
                     ManifoldProcess(OUBridgeExpVar(100f0, 150f0, 0.000000001f0, dec = -3f0)), 
                     DistNoisyInterpolatingDiscreteFlow(D1=Beta(3.0,1.5)))), 
                    Beta(1,2))

const rotM = Flowfusion.Rotations(3)

X0sampler(root) = (ContinuousState(randn(Float32, 3, 1, 1)), 
                    ManifoldState(rotM, reshape(Array{Float32}.(Flowfusion.rand(rotM, 1)), 1)),
                    (DiscreteState(21, [21]))
)

#=
function rand_mask(chainids)
    l = length(chainids)
    mask = falses(l)
    for _ in 1:(1+rand(Poisson(rand()*12)))
        dir = rand([-1,1])
        pos = rand(1:l)
        span = rand(Poisson(rand()*30))
        ordered = minmax(pos, pos + dir*span)
        mask[max(1,ordered[1]):min(l,ordered[2])] .= true
    end
    if !any(mask)
        mask[rand(1:l)] = true
    end
    return mask
end
=#

function group_mask(numgroups)
    num_to_keep = rand(1:max(1,numgroups-1))
    groups_to_keep = sample(1:numgroups, num_to_keep, replace = false)
    return groups_to_keep
end

#This version masks entire chains, but the mask is often shared when chains are similar lengths to prevent cheating.
function rand_mask(chainids)
    l = length(chainids)
    if rand() < 0.2
        return trues(l)
    end
    #60% of the draws remainder: The case where we mask chains together when they are similar lengths.
    if rand() < 0.75 
        chain_length_dict = countmap(chainids)
        chain_lengths = collect(values(chain_length_dict))
        perturbed_lengths = chain_lengths .* rand(Uniform(0.99, 1.01), length(chain_lengths))
        perm = sortperm(perturbed_lengths)
        sorted_lengths = perturbed_lengths[perm]
        sorted_chains = collect(keys(chain_length_dict))[perm]
        groups = UnitRange{Int64}[]
        base_ind = 1
        for i in 2:length(sorted_lengths)
            if sorted_lengths[i] - sorted_lengths[base_ind] > 0.05*sorted_lengths[base_ind]
                push!(groups, base_ind:i-1)
                base_ind = i
            end
        end
        push!(groups, base_ind:length(sorted_lengths))
        chain_groups = [sorted_chains[g] for g in groups] #chain_groups is a vector of vectors of chain ids
        chain_group_inds_to_keep = group_mask(length(chain_groups))
        mask = falses(l)
        for cgi in chain_group_inds_to_keep
            for gi in chain_groups[cgi]
                mask[chainids .== gi] .= true
            end
        end
        return mask
    end
    #20% of the remainder:
    unique_chains = unique(chainids)
    chains_to_keep = group_mask(length(unique_chains))
    mask = falses(l)
    for ci in chains_to_keep
        mask[chainids .== ci] .= true
    end
    return mask
end

function nobreaks(resinds, chainids, cmask)
    for i in 1:length(resinds)-1
        if (cmask[i] || cmask[i+1]) && (chainids[i] == chainids[i+1]) && (resinds[i] + 1 != resinds[i+1])
            return false
        end
    end
    return true
end

function compoundstate(rec) #<-Switching this to nothing so it errors if you don't set it in the call.
    L = length(rec.AAs)
    cmask = rand_mask(rec.chainids)
    breaks = nobreaks(rec.resinds, rec.chainids, cmask)
    X1locs = MaskedState(ContinuousState(rec.locs), cmask, cmask)
    X1rots = MaskedState(ManifoldState(rotM,eachslice(rec.rots, dims=3)), cmask, cmask)
    X1aas = MaskedState((DiscreteState(21, rec.AAs)), cmask, cmask)
    X1 = BranchingState((X1locs, X1rots, X1aas), rec.chainids, flowmask = cmask, branchmask = cmask) #<- .state, .groupings
    return X1, breaks
end

#To test.
function X1_modifier(X1)
    X1.state[3].S.state[X1.del] .= 21
    return X1
end

@eval DLProteinFormats begin
function sample_batched_inds(lens, clusters; l2b = length2batch(1000, 1.9))
    sampled_inds = filter(ind -> l2b(lens[ind]) > 0, one_ind_per_cluster(clusters))
    indices_lengths_jitter = [(ind, lens[ind], lens[ind] + 2randn()) for ind in sampled_inds]
    sort!(indices_lengths_jitter, by = x -> x[3])
    batch_inds = Vector{Int}[]
    current_batch = Int[]
    current_max_len = 0
    for (sampled_idx, original_len, _) in indices_lengths_jitter
        potential_max_len = max(current_max_len, original_len)
        if isempty(current_batch) || (length(current_batch) + 1 <= l2b(potential_max_len))
            push!(current_batch, sampled_idx)
            current_max_len = potential_max_len
        else
            push!(batch_inds, current_batch)
            current_batch = [sampled_idx]
            current_max_len = original_len
        end
    end
    if !isempty(current_batch)
        push!(batch_inds, current_batch)
    end
    return shuffle(batch_inds)
end
sample_batched_inds(flatrecs::MergedVector; l2b = length2batch(1000, 1.9)) = sample_batched_inds(flatrecs.len, flatrecs.cluster, l2b = l2b)
end

function training_prep(b)
    sampled = compoundstate.(dat[b])
    X1s = [s[1] for s in sampled]
    hasnobreaks = [s[2] for s in sampled]
    t = Uniform(0f0,1f0)
    bat = branching_bridge(P, X0sampler, X1s, t, 
                            coalescence_factor = 1.0, 
                            use_branching_time_prob = 0.5,
                            merger = BranchingFlows.canonical_anchor_merge,
                            #maxlen = 1.35*maximum(dat.len[b]) #Because this OOM'd
                            length_mins = Poisson(X0_mean_length),
                            deletion_pad = deletion_pad,
                            X1_modifier = X1_modifier,
                        )
    rotξ = Guide(bat.Xt.state[2], bat.X1anchor[2])
    resinds = similar(bat.Xt.groupings) .= 1:size(bat.Xt.groupings, 1)
    return (;t = bat.t, chainids = bat.Xt.groupings, resinds,
                    Xt = bat.Xt, hasnobreaks = hasnobreaks,
                    rotξ_target = rotξ, X1_locs_target = bat.X1anchor[1], X1aas_target = bat.X1anchor[3],
                    splits_target = bat.splits_target, del = bat.del)
end

function losses(P, X1hat, ts)
    hat_frames, hat_aas, hat_splits, hat_del = X1hat
    rotangent = Flowfusion.so3_tangent_coordinates_stack(values(linear(hat_frames)), tensor(ts.Xt.state[2]))
    hat_loc, hat_rot, hat_aas = (values(translation(hat_frames)), rotangent, hat_aas)
    l_loc = floss(P.P[1], hat_loc, ts.X1_locs_target,                scalefloss(P.P[1], ts.t, 1, 0.2f0)) * 20
    l_rot = floss(P.P[2], hat_rot, ts.rotξ_target,                   scalefloss(P.P[2], ts.t, 1, 0.2f0)) * 2
    l_aas = floss(P.P[3], hat_aas, onehot(ts.X1aas_target),          scalefloss(P.P[3], ts.t, 1, 0.2f0)) / 10
    splits_loss = floss(P, hat_splits, ts.splits_target, ts.Xt.padmask .* ts.Xt.branchmask, scalefloss(P, ts.t, 1, 0.2f0))
    del_loss = floss(P.deletion_policy, hat_del, ts.del, ts.Xt.padmask .* ts.Xt.branchmask, scalefloss(P, ts.t, 1, 0.2f0))
    return l_loc, l_rot, l_aas, splits_loss, del_loss
end

#=
function mod_wrapper(t, Xₜ)
    Xtstate = MaskedState.(Xₜ.state, (Xₜ.groupings .< Inf,), (Xₜ.groupings .< Inf,))
    println(replace(DLProteinFormats.ints_to_aa(tensor(Xtstate[3])[:]), "X"=>"-"))
    resinds = similar(Xₜ.groupings) .= 1:size(Xₜ.groupings, 1)
    input_bundle = ([t]', (Xtstate[1], Xtstate[2], onehot(Xtstate[3])), Xₜ.groupings, resinds) |> device
    sc_frames, _ = model(input_bundle...)
    sc_frames, _ = model(input_bundle..., sc_frames = sc_frames)
    sc_frames, _ = model(input_bundle..., sc_frames = sc_frames)
    pred = model(input_bundle..., sc_frames = sc_frames) |> cpu
    state_pred = ContinuousState(values(translation(pred[1]))), ManifoldState(rotM, eachslice(cpu(values(linear(pred[1]))), dims=(3,4))), pred[2]
    return state_pred, pred[3], pred[4]
end
=#

function gen2prot(samp, chainids, resnums; name = "Gen", )
    d = Dict(zip(0:25,'A':'Z'))
    chain_letters = get.((d,), chainids, 'Z')
    ProteinStructure(name, Atom{eltype(tensor(samp[1]))}[], DLProteinFormats.unflatten(tensor(samp[1]), tensor(samp[2]), tensor(samp[3]), chain_letters, resnums)[1])
 end
export_pdb(path, samp, chainids, resnums) = ProteinChains.writepdb(path, gen2prot(samp, chainids, resnums))

function test_sample(path; numchains = 1, chainlength_dist = [1], steps = 0f0:0.005f0:1f0, only_sampled_masked = true)
    b = rand(findall(dat.len .< 1000))
    sampled = compoundstate.(dat[[b]])
    X1s = [s[1] for s in sampled]
    if only_sampled_masked
        counter = 0
        while length(X1s[1].flowmask) == sum(X1s[1].flowmask)
            println("Resampling because there are no masked chains")
            counter += 1
            b = rand(findall(dat.len .< 1000))
            sampled = compoundstate.(dat[[b]])
            X1s = [s[1] for s in sampled]
            if counter > 100
                println("Failed to sample a masked chain")
                break
            end
        end
    end
    hasnobreaks = [true]
    t = [0f0]
    bat = branching_bridge(P, X0sampler, X1s, t, 
                            coalescence_factor = 1.0, 
                            use_branching_time_prob = 0.0,
                            merger = BranchingFlows.canonical_anchor_merge,
                            length_mins = Poisson(X0_mean_length),
                            deletion_pad = deletion_pad,
                            X1_modifier = X1_modifier,
                        )
    X0 = bat.Xt
    @show size(X0.groupings)
    samp = gen(P, X0, mod_wrapper, steps)
    @show sum(tensor(samp.state[3]) .== 21)
    export_pdb(path, samp.state, samp.groupings, collect(1:length(samp.groupings)))
end


dat = load(PDBSimpleFlat);
#To prevent OOM, we now need to factor in that some low-t samples might have more elements than their X1 lengths:
len_lbs = max.(dat.len, length.(union.(dat.chainids)) .* per_chain_upper_X0_len) .* deletion_pad

uncond_model = Flux.loadmodel!(BranchChainV1(), JLD2.load("/home/murrellb/BFruns/runs/gentle_tune_of933211_maxisplitty1_2025-10-16_536734/maxisplitty_536734_epoch_3.jld", "model_state"));
model = CondBranchChainV1(merge(CondBranchChainV1().layers, uncond_model.layers)) |> device;
model.layers.mask_embedder.weight ./= 10;
model.layers.break_embedder.weight ./= 10;

sched = burnin_learning_schedule(0.00001f0, 0.000250f0, 1.05f0, 0.9999f0)
opt_state = Flux.setup(Muon(eta = sched.lr, fallback = x -> any(size(x) .== 21)), model)
Flux.MLDataDevices.Internal.unsafe_free!(x) = (Flux.fmapstructure(Flux.MLDataDevices.Internal.unsafe_free_internal!, x); return nothing)

struct BatchDataset{T}
    batchinds::T
end
Base.length(x::BatchDataset) = length(x.batchinds)
Base.getindex(x::BatchDataset, i) = training_prep(x.batchinds[i])
function batchloader(; device=identity, parallel=true)
    uncapped_l2b = length2batch(1500, 1.25)
    #batchinds = sample_batched_inds(dat,l2b = x -> min(uncapped_l2b(x), 100))
    batchinds = sample_batched_inds(len_lbs, dat.cluster, l2b = x -> min(uncapped_l2b(x), 100))
    @show length(batchinds)
    x = BatchDataset(batchinds)
    dataloader = Flux.DataLoader(x; batchsize=-1, parallel)
    return device(dataloader)
end

frameid = [1]
vidpath = nothing
vidprepath = "$(rundir)/vids/"

function mod_wrapper(t, Xₜ; frameid = frameid, recycles = 5)
    export_pdb(vidpath*"/Xt/$(string(frameid[1], pad = 4)).pdb", Xₜ.state, Xₜ.groupings, collect(1:length(Xₜ.groupings)))
    #Xtstate = MaskedState.(Xₜ.state, (Xₜ.groupings .< Inf,), (Xₜ.groupings .< Inf,))
    Xtstate = Xₜ.state
    println(replace(DLProteinFormats.ints_to_aa(tensor(Xtstate[3])[:]), "X"=>"-"), ":", frameid[1])
    if length(tensor(Xtstate[3])[:]) > 2000
        error("Chain too long")
    end
    resinds = similar(Xₜ.groupings) .= 1:size(Xₜ.groupings, 1)
    input_bundle = ([t]', Xₜ, Xₜ.groupings, resinds, [true]) |> device
    sc_frames, _ = model(input_bundle...)
    for _ in 1:recycles
        print(".")
        sc_frames, _ = model(input_bundle..., sc_frames = sc_frames)
    end
    pred = model(input_bundle..., sc_frames = sc_frames) |> cpu
    state_pred = ContinuousState(values(translation(pred[1]))), ManifoldState(rotM, eachslice(cpu(values(linear(pred[1]))), dims=(3,4))), pred[2]
    export_pdb(vidpath*"/X1hat/$(string(frameid[1], pad = 4)).pdb", (state_pred[1], state_pred[2], Xₜ.state[3]), Xₜ.groupings, collect(1:length(Xₜ.groupings)))
    frameid[1] += 1
    return state_pred, pred[3], pred[4]
end


textlog("$(rundir)/log.csv", ["epoch", "batch", "learning rate", "loss"])
for epoch in 1:7
    if epoch == 6
        sched = linear_decay_schedule(sched.lr, 0.000000001f0, 5800) 
    end
    for (i, ts) in enumerate(batchloader(; device = device))
        sc_frames = nothing
        if rand() < 0.5
            sc_frames, _ = model(ts.t', ts.Xt, ts.chainids, ts.resinds, ts.hasnobreaks)
        end
        l, grad = Flux.withgradient(model) do m
            frames, aa_logits, count_log, del_logit = m(ts.t', ts.Xt, ts.chainids, ts.resinds, ts.hasnobreaks, sc_frames = sc_frames)
            l_loc, l_rot, l_aas, l_splits, l_del = losses(P, (frames, aa_logits, count_log, del_logit), ts)
            @show l_loc, l_rot, l_aas, l_splits, l_del
            l_loc + l_rot + l_aas + l_splits + l_del
        end
        Flux.update!(opt_state, model, grad[1])
        (mod(i, 10) == 0) && Flux.adjust!(opt_state, next_rate(sched))
        textlog("$(rundir)/log.csv", [epoch, i, sched.lr, l])
        if mod(i, 2000) == 0
            for v in 1:5
                try
                    vidname = "e$(epoch)_b$(i)_samp$(v)"
                    vidpath = vidprepath*vidname
                    frameid = [1]
                    mkpath(vidpath*"/Xt")
                    mkpath(vidpath*"/X1hat")
                    steps = 0f0:0.005f0:1f0
                    test_sample(vidpath*"/X1hat/$(string(length(steps), pad = 4)).pdb", numchains = rand(1:4), chainlength_dist = [1], steps = steps)
                catch
                    println("Error in test_sample for samp $v")
                end
            end
            jldsave("$(rundir)/model_epoch_$(epoch)_batch_$(i).jld", model_state = Flux.state(cpu(model)), opt_state=cpu(opt_state))
        end
    end
    jldsave("$(rundir)/model_epoch_$(epoch).jld", model_state = Flux.state(cpu(model)), opt_state=cpu(opt_state))
end



unqcode = rand(1000000:9999999)
for v in 1:20
    vidname = "test_$(unqcode)_samp$(v)"
    vidpath = vidprepath*vidname
    frameid = [1]
    mkpath(vidpath*"/Xt")
    mkpath(vidpath*"/X1hat")
    steps = 0f0:0.005f0:1f0
    test_sample(vidpath*"/X1hat/$(string(length(steps), pad = 4)).pdb", numchains = rand(1:4), chainlength_dist = [1], steps = steps)
end
