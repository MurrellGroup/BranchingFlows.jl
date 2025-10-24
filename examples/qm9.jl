#Branching Flows demo
using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path = "../")

using BranchingFlows
using Flux, Onion, RandomFeatureMaps, StatsBase, Plots, ForwardBackward, Flowfusion, ForwardBackward, Distributions, CannotWaitForTheseOptimisers, LinearAlgebra, Random, Einops, LearningSchedules, JLD2

using CUDA
device!(0) #Because we have set CUDA_VISIBLE_DEVICES = GPUnum
devi = gpu

using Serialization
molecules = deserialize("qm9.jls")





bond_lengths = [
    # C–C bonds
    ("C–C (single)", 1.54),
    ("C=C (double)", 1.34),
    ("C≡C (triple)", 1.20),
    ("C–C (aromatic)", 1.39),

    # C–H bonds
    ("C–H (sp3)", 1.09),
    ("C–H (sp2)", 1.08),
    ("C–H (sp)", 1.06),

    # C–N bonds
    ("C–N (single)", 1.47),
    ("C=N (double)", 1.28),
    ("C≡N (triple)", 1.16),

    # C–O bonds
    ("C–O (single)", 1.43),
    ("C=O (double)", 1.23),
    ("C–O (aromatic/partial double)", 1.36),

    # C–S bonds
    ("C–S (single)", 1.82),
    ("C=S (double)", 1.61),

    # C–halogen bonds
    ("C–F", 1.35),
    ("C–Cl", 1.77),
    ("C–Br", 1.94),
    ("C–I", 2.14),

    # N–H and O–H
    ("N–H", 1.01),
    ("O–H", 0.96),

    # N–O and N–N
    ("N–O (single)", 1.40),
    ("N=O (double)", 1.20),
    ("N–N (single)", 1.45),
    ("N=N (double)", 1.25),
    ("N≡N (triple)", 1.10),

    # O–O and S–S
    ("O–O (single)", 1.48),
    ("O=O (double)", 1.21),
    ("S–S", 2.05),
    ("S–O (single)", 1.63),
    ("S=O (double)", 1.43),

    # Metal–ligand bonds (approximate, vary by oxidation state and coordination)
    ("Fe–N (porphyrin)", 2.00),
    ("Fe–O (oxo complex)", 1.65),
    ("Fe–S (thiolate)", 2.25),
    ("Cu–N", 2.00),
    ("Cu–O", 1.95),
    ("Cu–S", 2.25),
    ("Ni–N", 1.90),
    ("Ni–O", 1.85),
    ("Ni–S", 2.13),
    ("Co–N", 1.96),
    ("Co–O", 1.88),
    ("Zn–N", 2.05),
    ("Zn–O", 1.95),
    ("Mn–O", 1.90),
    ("Cr–O (oxo)", 1.60),
    ("V=O (oxo)", 1.58),
    ("Mo=O (oxo)", 1.70),
    ("W=O (oxo)", 1.71),
    ("Pt–Cl", 2.30),
    ("Pd–Cl", 2.28),
    ("Ag–N", 2.20),
    ("Ag–O", 2.10)
]

sort(last.(bond_lengths))

centers = collect(0.9:0.05:2.4)








using Random

# Return a permutation p such that hydrogens ('H') are placed
# immediately after their nearest non-hydrogen, ordered by distance
# (ties broken randomly).
function reorder_indices(coords, elements)
    N = length(elements)
    is_h = elements .== 'H'
    non_idx = findall(!, is_h)
    h_idx = findall(identity, is_h)

    # Map: anchor_index => Vector of (hydrogen_index, squared_distance)
    anchor_to_hs = Dict(j => Vector{Tuple{Int,Float64}}() for j in non_idx)

    # Assign each hydrogen to its nearest non-hydrogen anchor (random tie-break)
    tol = 1e-12
    for h in h_idx
        xh = @view coords[:, h]
        best_d = Inf
        best_js = Int[]
        for j in non_idx
            d = sum(abs2, (@view coords[:, j]) .- xh)
            if d + tol < best_d
                best_d = d
                empty!(best_js); push!(best_js, j)
            elseif abs(d - best_d) ≤ tol
                push!(best_js, j)
            end
        end
        anchor = length(best_js) == 1 ? best_js[1] : rand(rng, best_js)
        push!(anchor_to_hs[anchor], (h, best_d))
    end

    # Build permutation: keep non-H order; insert their H's sorted by distance
    p = Int[]
    ε = 1e-9
    for j in non_idx
        push!(p, j)
        hs = get(anchor_to_hs, j, Tuple{Int,Float64}[])
        if !isempty(hs)
            # Random tie-break by adding tiny jitter, then sort ascending by distance
            jittered = [(h, d + ε * rand()) for (h, d) in hs]
            sort!(jittered, by = x -> x[2])
            append!(p, first.(jittered))
        end
    end
    return p
end

# Convenience: apply permutation and return reordered arrays
function reorder(mol)
    p = reorder_indices(mol.coords, mol.elements)
    return (; coords = mol.coords[:, p], elements = mol.elements[p])
end

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

vocabulary = unique(vcat((mol.elements for mol in molecules)...))
vocab_dict = Dict(name => i for (i, name) in enumerate(vocabulary))
reverse_vocab_dict = Dict(i => name for (i, name) in enumerate(vocabulary))
masked_index = length(vocabulary)+1
reverse_vocab_dict[masked_index] = 'X'

#Data draw:
function X1target()
    mol = reorder(rand(molecules))
    elements = [vocab_dict[el] for el in mol.elements]
    coords = center_and_randrot(mol.coords)
    n = length(elements)
    masked_continuous = MaskedState(ContinuousState(coords), trues(n), trues(n)) #Note: must return a tuple of states.
    masked_discrete = MaskedState(DiscreteState(masked_index, elements), trues(n), trues(n)) #Note: must return a tuple of states.
    #return BranchingState((masked_continuous, masked_discrete), 1 .+ (elements .== vocab_dict['H'])) #Second argument is "groupings"
    X1 = BranchingState((masked_continuous, masked_discrete), ones(Int,n)) #Second argument is "groupings"
    X1 = uniform_del_insertions(X1, 1.0)
    return X1
end

X0sampler(root) = (ContinuousState(randn(Float32,3,1)), DiscreteState(masked_index, [masked_index])) #Note: must return a tuple of states. Discrete states must start in the dummy.

distmat(p) = clamp.(Onion.pairwise_sqeuclidean(permutedims(p, (2,1,3)), p), 0.01f0, 10000f0) .- 0.01f0 #This is because sometimes near-zero numbers snap to -1 or 1 when run through decay.
decay(d) = (sign.(d) ./ (1 .+ abs.(d.^2) ./ 1))
function pair_features(coords)
    o = rearrange(coords, (:d, :L, :B) --> (:d, 1, :L, :B)) .- rearrange(coords, (:d, :L, :B) --> (:d, :L, 1, :B)) #We don't need the other direction on these, because that is just the sign flip
    d = distmat(coords)
    e1 = rearrange(1.443f0 .* softplus.(.- (d)), (..) --> (1, ..))
    e2 = rearrange(1.443f0 .* softplus.(.- (d ./ 5)), (..) --> (1, ..))
    e3 = rearrange(1.443f0 .* softplus.(.- (d ./ 15)), (..) --> (1, ..))
    return vcat(decay(o .* 6), decay(o .* 3), decay(o), decay(o ./ 3), decay(o ./ 6), e1, e2, e3)
end

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
    σ = softplus.(R) .+ oftype(T, 1e-6)
    denom = 2 .* reshape(σ .^ 2, :, 1)          # (M, 1)
    return exp.(-D2 ./ denom)                   # (M, N)
end

function (rbf::TrainableRBF{T})(X::AbstractArray{T}) where T<:Real
    X′ = reshape(X, size(X, 1), :)
    Y′ = rbf(X′)
    return reshape(Y′, :, size(X)[2:end]...)
end
=#

struct Toy{L}
    layers::L
end
Flux.@layer Toy
function Toy(dim, depth; shift_depth = depth)
    nheads = 12
    head_dim = 64
    layers = (;
        depth = depth,
        shift_depth = shift_depth,
        loc_rff = RandomFourierFeatures(3 => 2dim, 1f0),
        loc_rff2 = RandomFourierFeatures(3 => 2dim, 0.1f0),
        loc_encoder = Dense(4dim => dim, bias=false),
        t_rff = RandomFourierFeatures(1 => 2dim, 1f0),
        t_embed = Dense(2dim => dim, bias=false),
        d_encoder = Embedding(6 => dim),
        rope = RoPE(head_dim, 1000),
        transformers = [Onion.AdaTransformerBlock(dim, dim, nheads; head_dim = head_dim, qk_norm = true, g1_gate = Modulator(dim => nheads*head_dim), pair_proj = Dense(18=>nheads)) for _ in 1:depth],
        loc_shifters = [Dense(dim => 3, bias=false) for _ in 1:shift_depth],
        count_decoder = Dense(dim => 1, bias=false),
        del_decoder = Dense(dim => 1, bias=false),
        d_decoder = Dense(dim => 6, bias=true),
    )
    return Toy(layers)
end
function (m::Toy)(t,preXt)
    l = m.layers
    Xt = preXt.state
    lmask = Flowfusion.getlmask(Xt[1])
    locs = tensor(Xt[1])
    x = l.d_encoder(tensor(Xt[2])) + l.loc_encoder(vcat(l.loc_rff(locs),l.loc_rff2(locs)))
    t_cond = l.t_embed(l.t_rff(reshape(zero(similar(tensor(Xt[1]), size(tensor(Xt[1]),3))) .+ t, 1, :))) #Because "gen" will pass a scalar t, but we train with each batch having its own t.
    rope = l.rope[1:size(locs,2)]
    pair_feats = pair_features(locs)
    for i in 1:(l.depth - l.shift_depth)
        #x = l.transformers[i](x; rope=x->l.rope(x, locs), cond = t_cond, pair_feats = pair_feats, kpad_mask = lmask)
        x = l.transformers[i](x; rope, cond = t_cond, pair_feats = pair_feats, kpad_mask = lmask)
    end
    for i in 1:l.shift_depth
        x = l.transformers[i + l.depth - l.shift_depth](x; rope, cond = t_cond, pair_feats = pair_feats, kpad_mask = lmask)
        locs += l.loc_shifters[i](x) .* (1 .- Onion.glut(t, 3, 0) .* 0.95f0)
        if i < l.shift_depth
            pair_feats = pair_features(locs)
        end
    end
    return (locs, l.d_decoder(x)), l.count_decoder(x)[1,:,:], l.del_decoder(x)[1,:,:]
end

#P = CoalescentFlow((OUFlow(25f0, 100f0, 0.001f0, -2f0), Flowfusion.DistNoisyInterpolatingDiscreteFlow()), Beta(1,2)) #Extreme schedule.
P = CoalescentFlow((BrownianMotion(0.01f0), Flowfusion.DistNoisyInterpolatingDiscreteFlow()), Beta(1,1), SequentialUniform())

model = Toy(256, 12, shift_depth = 6) |> devi

for l in model.layers.transformers
    l.attention.wo.weight ./= length(model.layers.transformers)
    l.feed_forward.w2.weight ./= length(model.layers.transformers)
end
for l in model.layers.loc_shifters
    l.weight ./= 10
end

#Optimizer:
sched = burnin_learning_schedule(0.0001f0, 0.005f0, 1.15f0, 0.9995f0)
opt_state = Flux.setup(Muon(eta = sched.lr, fallback = x -> (size(x,1) .== 3 || size(x,2) .== 6 || size(x,2) .== 6 || size(x,1) .== 1)), model)

 
function training_prep(; batch_size = 256)
    t = rand(Float32, batch_size)
    bat = branching_bridge(P, X0sampler, [X1target() for _ in 1:batch_size], t, coalescence_factor = 1.0);
    (;t, Xt = bat.Xt, X1targets = bat.X1anchor, splits_target = bat.splits_target, del = bat.del)
end

function m_wrap(t,Xt)
    X1hat, hat_splits, hat_del = model(devi([t]),devi(Xt))
    return (cpu(ContinuousState(X1hat[1])), cpu(X1hat[2])), cpu(hat_splits), cpu(hat_del) #<-Because no batch dim for discrete
end

function to_xyz(elements::AbstractVector, positions::AbstractMatrix)
    join("$e $x $y $z\n" for (e, (x, y, z)) in zip([reverse_vocab_dict[el] for el in elements], eachcol(positions)))
end


Flux.MLDataDevices.Internal.unsafe_free!(x) = (Flux.fmapstructure(Flux.MLDataDevices.Internal.unsafe_free_internal!, x); return nothing)

iters = 50000
struct BatchDataset end
Base.length(x::BatchDataset) = iters
Base.getindex(x::BatchDataset, i) = training_prep()

function batchloader(; device=identity, parallel=true)
    x = BatchDataset()
    dataloader = Flux.DataLoader(x; batchsize=-1, parallel)
    return device(dataloader)
end

for (i, ts) in enumerate(batchloader(; device = devi))
    if i == 45000
        sched = linear_decay_schedule(sched.lr, 0.000000001f0, 500)
    end
    l,g = Flux.withgradient(model) do m
        X1hat, hat_splits, hat_del = m(ts.t,ts.Xt)
        mse_loss = floss(P.P[1], X1hat[1], ts.X1targets[1], scalefloss(P.P[1], ts.t, 1, 0.2f0)) * 2
        d_loss = floss(P.P[2], X1hat[2], onehot(ts.X1targets[2]), scalefloss(P.P[2], ts.t, 1, 0.2f0)) / 3 #Add a floss wrapper that calls this onehot automatically.
        splits_loss = floss(P, hat_splits, ts.splits_target, ts.Xt.padmask, scalefloss(P, ts.t, 1, 0.2f0)) / 3
        del_loss = floss(P.deletion_policy, hat_del, ts.del, ts.Xt.padmask, scalefloss(P, ts.t, 1, 0.2f0)) / 3
        if i % 50 == 0
            println("mse_loss: $mse_loss, d_loss: $d_loss, splits_loss: $splits_loss, del_loss: $del_loss")
        end
        return mse_loss + d_loss + splits_loss + del_loss
    end
    Flux.update!(opt_state, model, g[1])
    if mod(i, 10) == 0
        Flux.adjust!(opt_state, next_rate(sched))
    end
    (i % 50 == 0) && println("i: $i; Loss: $l, eta: $(sched.lr)")
    if i % 1000 == 0
        X0 = BranchingFlows.BranchingState(BranchingFlows.regroup([[X0sampler(nothing) for _ in 1:1]]), [1 ;;]) #Note: You MUST get the batch dimension back in. The model will need it, and the sampler assumes it.
        samp = gen(P, X0, m_wrap, 0f0:0.0005f0:1f0)
        println(to_xyz(samp.state[2].S.state[:], tensor(samp.state[1])[:,:,1]))
    end
end

#jldsave("../examples/qm9_BM_v1.jld", model_state = Flux.state(cpu(model)), opt_state=cpu(opt_state))





#Exporting trajectories:
frameid = [1]
towrite = "../examples/QM9/samp2/"
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
samp = gen(P, X0, m_wrap, 0f0:0.001f0:1f0)
open(towrite*"/Xt/$(string(frameid[1], pad = 5)).xyz","a") do io
    println(io, to_xyz(samp.state[2].S.state[:], tensor(samp.state[1])[:,:,1]))
end
open(towrite*"/X1hat/$(string(frameid[1], pad = 5)).xyz","a") do io
    println(io, to_xyz(samp.state[2].S.state[:], tensor(samp.state[1])[:,:,1]))
end

