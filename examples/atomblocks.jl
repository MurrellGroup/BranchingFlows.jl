#TODO: ADD SELF CONDITIONING - LOOK AT SIDE CHAIN MODEL.
using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path = "../")



using BranchingFlows
using Flux, Onion, RandomFeatureMaps, StatsBase, Plots, ForwardBackward, Flowfusion, Distributions, Dates
using CannotWaitForTheseOptimisers, LinearAlgebra, Random, Einops, LearningSchedules, JLD2, Random, DLProteinFormats
using DLProteinFormats: load, PDBFlatom169K

using CUDA
device!(1) #Because we have set CUDA_VISIBLE_DEVICES = GPUnum
devi = gpu

rundir = "../examples/AtomBlocks/test_$(Date(now()))_$(rand(100000:999999))/"
mkpath(rundir)

dat = load(PDBFlatom169K);
#Elements that occur more than 1000 times in the data:
#=
element_counts = countmap(vcat([map(x -> x.element, dat[i].atoms) for i in 1:length(dat)]...));
common_elements = sort([e for e in keys(element_counts) if element_counts[e] > 1450])
=#
const element_coding_dict = Dict(zip(Int8[0, 1, 6, 7, 8, 9, 11, 12, 15, 16, 17, 19, 20, 25, 26, 27, 28, 29, 30, 33, 34, 35, 48, 53, 74, 80], 5:30))
const backbone_coding_dict = Dict(DLProteinFormats.static" N  "4 => 1,
                            DLProteinFormats.static" CA "4 => 2, 
                            DLProteinFormats.static" C  "4 => 3, 
                            DLProteinFormats.static" O  "4 => 4)

#119th element means "too rare, was coded as other".
const element_from_atom_code_dict = Dict(vcat(collect(zip(values(element_coding_dict),keys(element_coding_dict))),[(1,7),(2,6),(3,6),(4,8),(31,119)]))
#1,2,3,4 are for NA, CA, C, O. The rest are the top 26 elements. Then 31 is other. 32 will be reserved for "masked".
function atom_code(atom)
    if atom.category == 1 && haskey(backbone_coding_dict, atom.atomname)
        return backbone_coding_dict[atom.atomname]
    else
        return get(element_coding_dict,atom.element,31)
    end
end
element_from_atom_code(code) = element_from_atom_code_dict[code]

#allatoms = vcat([atom_code.(dat[i].atoms) for i in 1:length(dat)]...);
#cmp = countmap(allatoms)

const masked_index = 32

function center_and_randrot(X::AbstractArray{T}, σ=one(T)) where T<:Number
    X = X .- mean(X, dims = 2)
    @assert size(X, 1) == 3
    Q, _ = qr(randn!(similar(X, 3, 3)))
    if det(Q) < 0
        Q[:,1] .*= -1
    end
    R = Q
    t = randn!(similar(X, 3)) * σ
    X′ = reshape(X, 3, :)
    Y′ = R * X′ .+ t
    Y = reshape(Y′, size(X))
    return Y
end



#=
for i in 1:10000
    prerec = dat[i].atoms;
    preresnames = map(x -> x.resname, prerec)
    rec = prerec[preresnames .!= "HOH"];
    xyz = stack(x -> x.coords, rec)
    atomnames = map(x -> x.atomname, rec)
    chainids = map(x -> x.chainid, rec)
    resnames = map(x -> x.resname, rec)
    resnums = map(x -> x.resnum, rec)
    cats = map(x -> x.category, rec)
    if 2 in cats
        println(i)
        break;
    end
end


prerec = dat[10].atoms;
preresnames = map(x -> x.resname, prerec)
rec = prerec[preresnames .!= "HOH"];
xyz = stack(x -> x.coords, rec)
atomnames = map(x -> x.atomname, rec)
chainids = map(x -> x.chainid, rec)
resnames = map(x -> x.resname, rec)
resnums = map(x -> x.resnum, rec)
cats = map(x -> x.category, rec)


#groupings needs to augment chainids with splits on resnames for anything not in category 1

outer_radius = 16.0
inner_radius = 15.0
center = mean(xyz, dims = 2)

distances = sqrt.(sum((xyz .- center).^2, dims = 1))[:]

outer_distfilt = distances .< outer_radius
resfilt = Set(unique(resnums[outer_distfilt]))

inner_distfilt = distances .> inner_radius
shell_distfilt = outer_distfilt .& inner_distfilt
shell_resfilt = Set(unique(resnums[shell_distfilt]))
inds = [r in resfilt for r in resnums];
fixed_inds = [r in shell_resfilt for r in resnums];
sum(fixed_inds)
sum(inds)
=#

function atom_ball(prerec; inner_radius = 14.5, outer_radius = 15.5, filterHOH = true)
    if filterHOH
        preresnames = map(x -> x.resname, prerec)
        rec = prerec[preresnames .!= "HOH"]
    else
        rec = prerec
    end
    resnums = map(x -> x.resnum, rec)
    xyz = stack(x -> x.coords, rec)
    center_ind = rand(1:size(xyz, 2))
    center = xyz[:,center_ind:center_ind]
    distances = sqrt.(sum((xyz .- center).^2, dims = 1))[:]
    outer_distfilt = distances .< outer_radius
    resfilt = Set(unique(resnums[outer_distfilt]))
    inds = [r in resfilt for r in resnums]
    outrec = rec[inds]
    outdists = distances[inds]
    outresnums = resnums[inds]
    inner_distfilt = outdists .> inner_radius
    shell_resfilt = Set(unique(outresnums[inner_distfilt]))
    shell_inds = [r in shell_resfilt for r in outresnums]
    return outrec, shell_inds
end

#@time ab,shel = atom_ball(dat[1].atoms)

#Key choice: Category 3: all one group. Otherwise how does the model decide how many little smol mols there are?
#Alternative would be one group per cat-3 resnum.
function groups(category, chainids, resnums)
    modified_resnums = copy(resnums)
    modified_resnums[category .== 1] .= 0
    modified_resnums[category .== 2] .= 0
    catchain = zip(chainids, category, modified_resnums)
    groups = union(catchain)
    group_dict = Dict(zip(groups, 1:length(groups)))
    groupings = [group_dict[c] for c in catchain]
    return groupings
end

function group_mask(numgroups)
    num_to_keep = rand(1:max(1,numgroups-1))
    groups_to_keep = sample(1:numgroups, num_to_keep, replace = false)
    return groups_to_keep
end

function rand_mask(chainids, atomcodes)
    l = length(chainids)
    unique_chains = unique(chainids)
    if rand() < 0.5
        chains_to_keep = group_mask(length(unique_chains))
        mask = falses(l)
        for ci in chains_to_keep
            mask[chainids .== ci] .= true
        end
    else
        mask = trues(l)
    end
    if rand() < 0.5
        p = rand() #Sometimes this will be close to 1, and then all backbones get fixed.
        for ci in  unique_chains
            if rand() < p
                ci_inds = findall(chainids .== ci)
                for ind in ci_inds
                    if atomcodes[ind] in 1:3 #N, CA, C (but not O - let that get placed)
                        mask[ind] = false
                    end
                end
            end
        end
    end
    if !any(mask)
        mask .= true
    end
    return mask
end

#Data draw:
function X1target()
    b = rand(1:length(dat))
    ab,shell = atom_ball(dat[b].atoms)
    chainids = map(x -> x.chainid, ab)
    resnums = map(x -> x.resnum, ab)
    cats = map(x -> x.category, ab)
    groupings = groups(cats, chainids, resnums)
    atomcodes = atom_code.(ab)
    mask = rand_mask(chainids, atomcodes)
    mask[shell] .= false
    xyz = center_and_randrot(stack(x -> x.coords, ab)) ./ 10 #Convert to nanometers
    n = length(atomcodes)
    masked_continuous = MaskedState(ContinuousState(xyz), mask, mask) #Note: must return a tuple of states.
    masked_discrete = MaskedState(DiscreteState(masked_index, atomcodes), mask, mask) #Note: must return a tuple of states.
    X1 = BranchingState((masked_continuous, masked_discrete), groupings, flowmask = mask, branchmask = mask)
    return X1
end

function safeX1target()
    X1 = X1target()
    if all(X1.branchmask .== false) || length(X1.branchmask) < 50 || length(X1.branchmask) > 1500
        return safeX1target()
    end
    return X1
end

#@time X1 = safeX1target()
#length(X1.branchmask)
#X1.branchmask |> sum

X0sampler(root) = (ContinuousState(randn(Float32,3,1)), DiscreteState(masked_index, [masked_index])) #Note: must return a tuple of states. Discrete states must start in the dummy.

distmat(p) = Onion.pairwise_sqeuclidean(permutedims(p, (2,1,3)), p)

function fixed_pairfeats(resinds, groups)
    chain_diffs = Float32.(Onion.batched_pairs(==, groups, groups))
    num_diffs = Float32.(Onion.batched_pairs(-, resinds, resinds)) .* chain_diffs
    pos_num_diffs = .- max.(num_diffs, 0)
    neg_num_diffs = .- max.(.-num_diffs, 0)
    return vcat(reshape(pos_num_diffs, 1, size(pos_num_diffs)...), reshape(neg_num_diffs, 1, size(neg_num_diffs)...), reshape(chain_diffs, 1, size(chain_diffs)...))
end

function pair_features(coords)
    o = rearrange(coords, (:d, :L, :B) --> (:d, 1, :L, :B)) .- rearrange(coords, (:d, :L, :B) --> (:d, :L, 1, :B)) #We don't need the other direction on these, because that is just the sign flip
    pos_o = .- max.(o, 0)
    neg_o = .- max.(.-o, 0)
    d = .- sqrt.(max.(distmat(coords), 1f-6))
    return vcat(pos_o, neg_o, reshape(d, 1, size(d)...))
end

#3+7

#=
"""
    TrainableRBF(n => m, [σ])

Maps `n`-dimensional data to `m` Gaussian radial basis responses with trainable
centers and isotropic radii per basis.

The optional `σ` controls the initialization scale and element type.

Examples

```jldoctest
julia> rbf = TrainableRBF(2 => 4, 1.0); # 4 bases in 2D

julia> rbf(rand(2, 3)) |> size # 3 samples
(4, 3)

julia> rbf(rand(2, 3, 5)) |> size # extra batch dim
(4, 3, 5)
```
"""
struct TrainableRBF{T<:Real, A<:AbstractMatrix{T}, V<:AbstractVector{T}}
    centers::A  # (n, m)
    radii::V    # (m,) isotropic per basis
end

Flux.@layer TrainableRBF trainable=(centers, radii)

TrainableRBF(dims::Pair{<:Integer, <:Integer}, σ::Real=1.0) = TrainableRBF(dims, float(σ))

function TrainableRBF((d1, d2)::Pair{<:Integer, <:Integer}, σ::AbstractFloat)
    isfinite(σ) || throw(ArgumentError("scale must be finite"))
    centers = randn(typeof(σ), d1, d2) * σ
    radii = fill(typeof(σ)(1), d2)
    return TrainableRBF(centers, radii)
end

function (rbf::TrainableRBF{T})(X::AbstractMatrix{T}) where T<:Real
    C = rbf.centers
    R = rbf.radii
    X2 = sum(abs2, X; dims = 1)                 # (1, N)
    C2 = sum(abs2, C; dims = 1)                 # (1, M)
    D2 = (-2 .* (C' * X)) .+ C2' .+ X2          # (M, N)
    σ = softplus.(R) .+ T(1e-6)
    denom = 2 .* reshape(σ .^ 2, :, 1)          # (M, 1)
    return exp.(-D2 ./ denom)                   # (M, N)
end

function (rbf::TrainableRBF{T})(X::AbstractArray{T}) where T<:Real
    X′ = reshape(X, size(X, 1), :)
    Y′ = rbf(X′)
    return reshape(Y′, :, size(X)[2:end]...)
end
=#


@eval Onion begin
    norm_and_ff(h, ffn, ffn_norm, cond) = h + ffn(ffn_norm(h, cond...))
    function (block::TransformerBlock)(
        x, xs...;
        cond=nothing, pair_feats=nothing,
        pair=block.pair_proj(pair_feats),
        kws...
    )
    print(".")
        cond = isnothing(cond) ? () : (cond,)
        h = x + block.attention(
            block.attention_norm(x, cond...), xs...;
            pair, kws...)
        #h = h + block.feed_forward(block.ffn_norm(h, cond...))
        h = Flux.Zygote.checkpointed(norm_and_ff, h, block.feed_forward, block.ffn_norm, cond)
        return h
    end
end

oldAdaLN(dim,cond_dim) = AdaLN(Flux.LayerNorm(dim), Dense(cond_dim, dim), Dense(cond_dim, dim))

struct Toy{L}
    layers::L
end
Flux.@layer Toy
function Toy(dim, depth; shift_depth = depth)
    nheads = 12
    head_dim = 32
    layers = (;
        depth = depth,
        shift_depth = shift_depth,
        loc_rff = RandomFourierFeatures(3 => 2dim, 1f0),
        loc_rff2 = RandomFourierFeatures(3 => 2dim, 0.1f0),
        loc_encoder = Dense(4dim => dim, bias=false),
        t_rff = RandomFourierFeatures(1 => 2dim, 1f0),
        #rbf = TrainableRBF(reshape([0.9:0.1:2.5; (1.65:0.5:4).^2;], 1, :) ./ 10, ones(22) .* 0.05),
        t_embed = Dense(2dim => dim, bias=false),
        d_encoder = Embedding(masked_index => dim),
        mask_embedder = Embedding(2 => dim),
        #rope = RoPE(head_dim, 1000),
        #pfs:10
        #transformers = [Onion.AdaTransformerBlock(dim, dim, nheads; head_dim = head_dim, qk_norm = true, g1_gate = Modulator(dim => nheads*head_dim), pair_proj = Dense(32=>nheads)) for _ in 1:depth],
        #transformers = [Onion.TransformerBlock(dim, nheads; head_dim = head_dim, attention_norm = oldAdaLN(dim, dim), ffn_norm = oldAdaLN(dim, dim), qk_norm = true, g1_gate = Modulator(dim => nheads*head_dim), pair_proj = Dense(32=>nheads, x -> -softplus(x))) for _ in 1:depth],
        transformers = [Onion.TransformerBlock(dim, nheads; head_dim = head_dim, attention_norm = oldAdaLN(dim, dim), ffn_norm = oldAdaLN(dim, dim), qk_norm = true, g1_gate = Modulator(dim => nheads*head_dim), pair_proj = Dense(10=>nheads, x -> -softplus(x))) for _ in 1:depth],
        loc_shifters = [Dense(dim => 3, bias=false) for _ in 1:shift_depth],
        count_decoder = Dense(dim => 1, bias=false),
        del_decoder = Dense(dim => 1, bias=false),
        d_decoder = Dense(dim => masked_index, bias=true),
    )
    return Toy(layers)
end
function (m::Toy)(t,preXt, resinds)
    groupings = preXt.groupings
    static_z = Flux.Zygote.@ignore fixed_pairfeats(resinds, groupings)
    l = m.layers
    Xt = preXt.state
    cmask = Flowfusion.getcmask(Xt[1])
    pmask = preXt.padmask
    locs = tensor(Xt[1])
    x = l.d_encoder(tensor(Xt[2])) + l.loc_encoder(vcat(l.loc_rff(locs),l.loc_rff2(locs))) .+ l.mask_embedder(cmask .+ 1)
    t_cond = l.t_embed(l.t_rff(reshape(zero(similar(tensor(Xt[1]), size(tensor(Xt[1]),3))) .+ t, 1, :))) #Because "gen" will pass a scalar t, but we train with each batch having its own t.
    #rope = l.rope[1:size(locs,2)]
    pair_feats = vcat(static_z, pair_features(locs)) #, l.rbf(rearrange(distmat(locs), einops"... -> 1 ...")))
    for i in 1:(l.depth - l.shift_depth)
        x = l.transformers[i](x; rope = identity, cond = t_cond, pair_feats = pair_feats, kpad_mask = pmask)
    end
    for i in 1:l.shift_depth
        x = l.transformers[i + l.depth - l.shift_depth](x; rope = identity, cond = t_cond, pair_feats = pair_feats, kpad_mask = pmask)
        #locs += l.loc_shifters[i](x) .* (1 .- Onion.glut(t, 3, 0) .* 0.95f0)
        locs += l.loc_shifters[i](x) .* (rearrange(cmask, (..) --> (1, ..))) .* (1.05f0 .- rearrange(t, (..) --> (1, 1, ..)))
        pair_feats = vcat(static_z, pair_features(locs)) #, l.rbf(rearrange(distmat(locs), einops"... -> 1 ...")))
    end
    return (locs, l.d_decoder(x)), l.count_decoder(x)[1,:,:], l.del_decoder(x)[1,:,:]
end

#OUmild:
#P = CoalescentFlow((OUBridgeExpVar(5f0, 10f0, 0.001f0, dec = -1f0), Flowfusion.DistNoisyInterpolatingDiscreteFlow()), Beta(1,2))
#model = Toy(256, 12, shift_depth = 6) |> devi

P = CoalescentFlow((OUBridgeExpVar(5f0, 10f0, 0.001f0, dec = -1f0), Flowfusion.DistNoisyInterpolatingDiscreteFlow()), Beta(1,2))
model = Toy(256, 10, shift_depth = 5) |> devi

for l in model.layers.transformers
    l.attention.wo.weight ./= 10
    l.feed_forward.w2.weight ./= 10
    l.pair_proj.weight .*= 0
    l.attention_norm.scale.weight .*= 0
    l.ffn_norm.scale.weight .*= 0
    l.attention_norm.shift.weight .*= 0
    l.ffn_norm.shift.weight .*= 0
end
for l in model.layers.loc_shifters
    l.weight .*= 0
end

#model = Flux.loadmodel!(Toy(256, 12, shift_depth = 6), JLD2.load("/home/murrellb/BranchingFlows.jl/examples/QM9/runOUmild_run2/model_state_10000.jld", "model_state")) |> devi

#Optimizer:
sched = burnin_learning_schedule(0.00001f0, 0.0001f0, 1.05f0, 0.999998f0)
opt_state = Flux.setup(Muon(eta = sched.lr, fallback = x -> size(x,1) .== 3), model)

#Flux.freeze!(opt_state.layers.rbf.centers)

function training_prep(; batch_size = 3)
    t = rand(Float32, batch_size)
    #bat = branching_bridge(P, X0sampler, [safeX1target() for _ in 1:batch_size], t, coalescence_factor = 1.0, use_branching_time_prob = 0.5, length_mins = 1, deletion_pad = 1.2);
    bat = branching_bridge(P, X0sampler, [safeX1target() for _ in 1:batch_size], t, coalescence_factor = 1.0, use_branching_time_prob = 0.5, length_mins = Poisson(10), deletion_pad = 1.2);
    resinds = zeros(Int, size(bat.Xt.groupings))
    resinds .= 1:size(bat.Xt.groupings, 1)
    (;t, Xt = bat.Xt, X1targets = bat.X1anchor, splits_target = bat.splits_target, del = bat.del, resinds)
end

function m_wrap(t,Xt)
    resinds = zeros(Int, size(Xt.groupings))
    resinds .= 1:size(Xt.groupings, 1)
    X1hat, hat_splits, hat_del = model(devi([t]),devi(Xt), devi(resinds))
    return (cpu(ContinuousState(X1hat[1])), cpu(X1hat[2])), cpu(hat_splits), cpu(hat_del) #<-Because no batch dim for discrete
end

function to_xyz(elements::AbstractVector, positions::AbstractMatrix)
    join("$e $x $y $z\n" for (e, (x, y, z)) in zip([DLProteinFormats.Flatom.number_to_element_symbol(el) for el in elements], eachcol(positions)))
end

Flux.MLDataDevices.Internal.unsafe_free!(x) = (Flux.fmapstructure(Flux.MLDataDevices.Internal.unsafe_free_internal!, x); return nothing)

iters = 500000
struct BatchDataset end
Base.length(x::BatchDataset) = iters
Base.getindex(x::BatchDataset, i) = training_prep()

function batchloader(; device=identity, parallel=true)
    x = BatchDataset()
    dataloader = Flux.DataLoader(x; batchsize=-1, parallel)
    return device(dataloader)
end

#ts = training_prep() |> devi
#X1hat, hat_splits, hat_del = model(ts.t, ts.Xt, ts.resinds)

backup_ts = nothing
for (i, ts) in enumerate(batchloader(; device = devi))
    backup_ts = ts
    if (size(ts.Xt.groupings, 1) > 1000) || (sum(ts.Xt.branchmask) == 0) || (sum(ts.Xt.flowmask) == 0) ||
        continue;
    end
    #if i == 5000
    #   Flux.thaw!(opt_state.layers.rbf.centers)
    #end
    if i == 450000
        sched = linear_decay_schedule(sched.lr, 0.000000001f0, 5000)
    end
    l,g = Flux.withgradient(model) do m
        X1hat, hat_splits, hat_del = m(ts.t,ts.Xt, ts.resinds)
        mse_loss = floss(P.P[1], X1hat[1], ts.X1targets[1], scalefloss(P.P[1], ts.t, 1, 0.2f0)) * 5
        d_loss = floss(P.P[2], X1hat[2], onehot(ts.X1targets[2]), scalefloss(P.P[2], ts.t, 1, 0.2f0)) / 3 #Add a floss wrapper that calls this onehot automatically.
        splits_loss = floss(P, hat_splits, ts.splits_target, ts.Xt.padmask, scalefloss(P, ts.t, 1, 0.2f0)) / 3
        del_loss = floss(P.deletion_policy, hat_del, ts.del, ts.Xt.padmask, scalefloss(P, ts.t, 1, 0.2f0)) / 3
        #if i % 1 == 0
            println("mse_loss: $mse_loss, d_loss: $d_loss, splits_loss: $splits_loss, del_loss: $del_loss, t: $(ts.t)")
        #end
        return mse_loss + d_loss + splits_loss + del_loss
    end
    Flux.update!(opt_state, model, g[1])
    if mod(i, 10) == 0
        GC.gc()
        CUDA.reclaim()
        Flux.adjust!(opt_state, next_rate(sched))
    end
    (i % 50 == 0) && println("i: $i; Loss: $l, eta: $(sched.lr)")
    #=
    if i % 5000 == 0
        frameid = [1]
        towrite = rundir*"batch$(string(i, pad = 5))"
        mkpath(towrite*"/X1hat")
        mkpath(towrite*"/Xt")
        function m_wrap(t,Xt; dir = towrite)
            X1hat, hat_splits, hat_del = model(devi([t]),devi(Xt)) |> cpu
            open(towrite*"/Xt/$(string(frameid[1], pad = 5)).xyz","a") do io
                println(io, to_xyz(Xt.state[2].S.state[:], tensor(Xt.state[1].S)[:,:,1]))
            end
            open(towrite*"/X1hat/$(string(frameid[1], pad = 5)).xyz","a") do io
                println(io, to_xyz(Xt.state[2].S.state[:], tensor(X1hat[1])[:,:,1]))
            end
            frameid[1] += 1
            return (ContinuousState(X1hat[1]), X1hat[2]), hat_splits, hat_del #<-Because no batch dim for discrete
        end
        X0 = branching_bridge(P, X0sampler, [X1target() for _ in 1:1], [0.0000000001f0], coalescence_factor = 1.0).Xt
        step_sched(t) = 1-(cos(t*pi)+1)/2
        custom_steps = step_sched.(0f0:0.001:1f0)
        samp = gen(P, X0, m_wrap, custom_steps)
        open(towrite*"/Xt/$(string(frameid[1], pad = 5)).xyz","a") do io
            println(io, to_xyz(samp.state[2].S.state[:], tensor(samp.state[1])[:,:,1]))
        end
        open(towrite*"/X1hat/$(string(frameid[1], pad = 5)).xyz","a") do io
            println(io, to_xyz(samp.state[2].S.state[:], tensor(samp.state[1])[:,:,1]))
        end
        println(to_xyz(samp.state[2].S.state[:], tensor(samp.state[1])[:,:,1]))
    end
    =#
    if mod(i, 25000) == 0
        jldsave(rundir*"model_state_v3_$(string(i, pad = 5)).jld", model_state = Flux.state(cpu(model)), opt_state=cpu(opt_state))
    end
end

#jldsave("../examples/qm9_BM_v1.jld", model_state = Flux.state(cpu(model)), opt_state=cpu(opt_state))



ts = training_prep() |> devi
X1hat, hat_splits, hat_del = model(ts.t,ts.Xt, ts.resinds)
mse_loss = floss(P.P[1], X1hat[1], ts.X1targets[1], scalefloss(P.P[1], ts.t, 1, 0.2f0)) * 2
d_loss = floss(P.P[2], X1hat[2], onehot(ts.X1targets[2]), scalefloss(P.P[2], ts.t, 1, 0.2f0)) / 3 #Add a floss wrapper that calls this onehot automatically.
splits_loss = floss(P, hat_splits, ts.splits_target, ts.Xt.padmask, scalefloss(P, ts.t, 1, 0.2f0)) / 3
del_loss = floss(P.deletion_policy, hat_del, ts.del, ts.Xt.padmask, scalefloss(P, ts.t, 1, 0.2f0)) / 3


f(a) = nothing
f(a::AbstractArray{T}) where T<:Real = isfinite(maximum(a)) ? print(maximum(a), " ") : error()#println(maximum(a))
f(a::AbstractArray) = f.(a)
f(a::NamedTuple) = f.(values(a))


ts = nothing
for i in 1:10000
    ts = training_prep() |> devi
    l,g = Flux.withgradient(model) do m
        X1hat, hat_splits, hat_del = m(ts.t,ts.Xt, ts.resinds)
        mse_loss = floss(P.P[1], X1hat[1], ts.X1targets[1], scalefloss(P.P[1], ts.t, 1, 0.2f0)) * 5
        d_loss = floss(P.P[2], X1hat[2], onehot(ts.X1targets[2]), scalefloss(P.P[2], ts.t, 1, 0.2f0)) / 3 #Add a floss wrapper that calls this onehot automatically.
        splits_loss = floss(P, hat_splits, ts.splits_target, ts.Xt.padmask, scalefloss(P, ts.t, 1, 0.2f0)) / 3
        del_loss = floss(P.deletion_policy, hat_del, ts.del, ts.Xt.padmask, scalefloss(P, ts.t, 1, 0.2f0)) / 3
        #if i % 1 == 0
            println("mse_loss: $mse_loss, d_loss: $d_loss, splits_loss: $splits_loss, del_loss: $del_loss")
        #end
        return mse_loss + d_loss + splits_loss + del_loss
    end
    f(g[1].layers);
end


X1hat, hat_splits, hat_del = model(ts.t,ts.Xt, ts.resinds)



mapfoldl(maximum, g[1].layers)


step_sched(t) = 1-(cos(t*pi)+1)/2
#Exporting trajectories:
for i in 2:50
    frameid = [1]
    towrite = "../examples/QM9/runOUmild_run2_resume/aftertraining/samp_$(string(i, pad = 5))"
    mkpath(towrite*"/X1hat")
    mkpath(towrite*"/Xt")
    function m_wrap(t,Xt; dir = towrite)
        X1hat, hat_splits, hat_del = model(devi([t]),devi(Xt)) |> cpu
        open(towrite*"/Xt/$(string(frameid[1], pad = 5)).xyz","a") do io
            println(io, to_xyz(Xt.state[2].S.state[:], tensor(Xt.state[1].S)[:,:,1]))
        end
        open(towrite*"/X1hat/$(string(frameid[1], pad = 5)).xyz","a") do io
            println(io, to_xyz(Xt.state[2].S.state[:], tensor(X1hat[1])[:,:,1]))
        end
        frameid[1] += 1
        return (ContinuousState(X1hat[1]), X1hat[2]), hat_splits, hat_del #<-Because no batch dim for discrete
    end
    X0 = branching_bridge(P, X0sampler, [X1target() for _ in 1:1], [0.0000000001f0], coalescence_factor = 1.0).Xt
    custom_steps = step_sched.(0f0:0.001:1f0)
    samp = gen(P, X0, m_wrap, custom_steps)
    open(towrite*"/Xt/$(string(frameid[1], pad = 5)).xyz","a") do io
        println(io, to_xyz(samp.state[2].S.state[:], tensor(samp.state[1])[:,:,1]))
    end
    open(towrite*"/X1hat/$(string(frameid[1], pad = 5)).xyz","a") do io
        println(io, to_xyz(samp.state[2].S.state[:], tensor(samp.state[1])[:,:,1]))
    end
end








for i in 1:1
    towrite = "../examples/QM9/runOUmild_run2_resume/forcing/"
        mkpath(towrite)
            id = "samp_$(string(i, pad = 5))"
    function m_wrap(t,Xt; dir = towrite)
            X1hat, hat_splits, hat_del = model(devi([t]),devi(Xt)) |> cpu
            @show t, size(hat_splits)
            if 0.0 < mean(t) < 0.5
                hat_splits .+= 1
            end
            if length(hat_splits) > 100
                hat_splits .= -100
            end
        return (ContinuousState(X1hat[1]), X1hat[2]), hat_splits, hat_del #<-Because no batch dim for discrete
            end
    X0 = branching_bridge(P, X0sampler, [X1target() for _ in 1:1], [0.0000000001f0], coalescence_factor = 1.0).Xt
        custom_steps = step_sched.(0f0:0.001:1f0)
            samp = gen(P, X0, m_wrap, custom_steps)
                open(towrite*id*".xyz","a") do io
        println(io, to_xyz(samp.state[2].S.state[:], tensor(samp.state[1])[:,:,1]))
            end
end