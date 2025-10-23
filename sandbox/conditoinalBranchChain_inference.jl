using Pkg
Pkg.activate(".")
Pkg.add("Revise")

using Revise

Pkg.develop(path = "../")

#Pkg.add(["JLD2", "Flux", "CannotWaitForTheseOptimisers", "LearningSchedules", "DLProteinFormats"])
#Pkg.add(["CUDA", "cuDNN"])

#Pkg.develop(path="../../ForwardBackward.jl/")
#Pkg.develop(path="../../Flowfusion.jl/")


using DLProteinFormats, Flux, CannotWaitForTheseOptimisers, LearningSchedules, JLD2, Dates, BatchedTransformations, ProteinChains
using BranchingFlows, Flowfusion, Distributions, ForwardBackward, RandomFeatureMaps, InvariantPointAttention, Onion, StatsBase, Random
using DLProteinFormats: load, PDBSimpleFlat, batch_flatrecs, sample_batched_inds, length2batch
using CUDA

device!(1) #Because we have set CUDA_VISIBLE_DEVICES = GPUnum
device = gpu

rundir = "runs/condchain_tune_of_maxisplitty_$(Date(now()))_$(rand(100000:999999))"
mkpath("$(rundir)/samples")

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
            #frames = l.framemovers[i - l.depth + l.f_depth](frames, x, t = 1 .- (1 .- t) .* cmask)
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

#Use this for a loop-design model:
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

#Use this for a chain-design model:
function group_mask(numgroups)
    num_to_keep = rand(1:max(1,numgroups-1))
    groups_to_keep = sample(1:numgroups, num_to_keep, replace = false)
    return groups_to_keep
end
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

function compoundstate(rec; del_prob_dist = nothing) #<-Switching this to nothing so it errors if you don't set it in the call.
    L = length(rec.AAs)
    cmask = rand_mask(rec.chainids)
    breaks = nobreaks(rec.resinds, rec.chainids, cmask)
    X1locs = MaskedState(ContinuousState(rec.locs), cmask, cmask)
    X1rots = MaskedState(ManifoldState(rotM,eachslice(rec.rots, dims=3)), cmask, cmask)
    X1aas = MaskedState((DiscreteState(21, rec.AAs)), cmask, cmask)
    X1 = BranchingState((X1locs, X1rots, X1aas), rec.chainids, flowmask = cmask, branchmask = cmask) #<- .state, .groupings
    X1 = uniform_del_insertions(X1, rand(del_prob_dist))
    X1.state[3].S.state[X1.del] .= 21 #Set deleted discrete components to the dummy!
    return X1, breaks
end

function gen2prot(samp, chainids, resnums; name = "Gen", )
    d = Dict(zip(0:25,'A':'Z'))
    chain_letters = get.((d,), chainids, 'Z')
    ProteinStructure(name, Atom{eltype(tensor(samp[1]))}[], DLProteinFormats.unflatten(tensor(samp[1]), tensor(samp[2]), tensor(samp[3]), chain_letters, resnums)[1])
 end
export_pdb(path, samp, chainids, resnums) = ProteinChains.writepdb(path, gen2prot(samp, chainids, resnums))

function test_sample(path; numchains = 1, chainlength_dist = [1], steps = 0f0:0.005f0:1f0, only_sampled_masked = true)
    b = rand(findall(dat.len .< 1000))
    sampled = compoundstate.(dat[[b]], del_prob_dist = Uniform(0.2, 0.5))
    X1s = [s[1] for s in sampled]
    if only_sampled_masked
        counter = 0
        while length(X1s[1].flowmask) == sum(X1s[1].flowmask)
            println("Resampling because there are no masked chains")
            counter += 1
            b = rand(findall(dat.len .< 1000))
            sampled = compoundstate.(dat[[b]], del_prob_dist = Uniform(0.2, 0.5))
            X1s = [s[1] for s in sampled]
            if counter > 100
                println("Failed to sample a masked chain")
                break
            end
        end
    end
    hasnobreaks = [true]
    t = Uniform(0f0,0.0000000001f0)
    bat = branching_bridge(P, X0sampler, X1s, t, 
                            coalescence_factor = 1.0, 
                            use_branching_time_prob = 0.0,
                            merger = BranchingFlows.canonical_anchor_merge
                        )
    X0 = bat.Xt
    @show size(X0.groupings)
    samp = gen(P, X0, mod_wrapper, steps)
    @show sum(tensor(samp.state[3]) .== 21)
    export_pdb(path, samp.state, samp.groupings, collect(1:length(samp.groupings)))
end


dat = load(PDBSimpleFlat);

model = Flux.loadmodel!(CondBranchChainV1(), JLD2.load("model.jld", "model_state"));

#Weird globals for frame exporting:
frameid = [1]
vidpath = nothing
vidprepath = "$(rundir)/vids/"

function mod_wrapper(t, Xₜ; frameid = frameid, recycles = 5)
    export_pdb(vidpath*"/Xt/$(string(frameid[1], pad = 4)).pdb", Xₜ.state, Xₜ.groupings, collect(1:length(Xₜ.groupings)))
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
