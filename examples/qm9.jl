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
molecules = deserialize("../examples/qm9.jls")

function center_and_randrot(X::AbstractArray{T}, σ::T=zero(T)) where T<:Number
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
    mol = rand(molecules)
    elements = [vocab_dict[el] for el in mol.elements]
    coords = center_and_randrot(mol.coords)
    n = length(elements)
    masked_continuous = MaskedState(ContinuousState(coords), trues(n), trues(n)) #Note: must return a tuple of states.
    masked_discrete = MaskedState(DiscreteState(masked_index, elements), trues(n), trues(n)) #Note: must return a tuple of states.
    #return BranchingState((masked_continuous, masked_discrete), 1 .+ (elements .== vocab_dict['H'])) #Second argument is "groupings"
    return BranchingState((masked_continuous, masked_discrete), ones(Int,n)) #Second argument is "groupings"
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
        #rope = MultidimRoPE(theta=100f0),
        transformers = [Onion.AdaTransformerBlock(dim, dim, nheads; head_dim = head_dim, qk_norm = true, g1_gate = Modulator(dim => nheads*head_dim), pair_proj = Dense(18=>nheads)) for _ in 1:depth],
        loc_shifters = [Dense(dim => 3, bias=false) for _ in 1:shift_depth],
        count_decoder = Dense(dim => 1, bias=false),
        d_decoder = Dense(dim => 6, bias=true),
    )
    return Toy(layers)
end
function (m::Toy)(t,Xt)
    l = m.layers
    lmask = Flowfusion.getlmask(Xt[1])
    #=
    if isnothing(lmask)
        pmask = 0
    else
        pmask = Flux.Zygote.@ignore self_att_padding_mask(lmask)
    end
    =#
    locs = tensor(Xt[1])
    x = l.d_encoder(tensor(Xt[2])) + l.loc_encoder(vcat(l.loc_rff(locs),l.loc_rff2(locs)))
    t_cond = l.t_embed(l.t_rff(reshape(zero(similar(tensor(Xt[1]), size(tensor(Xt[1]),3))) .+ t, 1, :))) #Because "gen" will pass a scalar t, but we train with each batch having its own t.
    rope = l.rope[1:size(locs,2)]
    pair_feats = pair_features(locs)
    for i in 1:(l.depth - l.shift_depth)
        #x = l.transformers[i](x; rope=x->l.rope(x, locs), cond = t_cond, pair_feats = pair_feats, kpad_mask = lmask)
        x = l.transformers[i](x; rope, cond = t_cond, pair_feats = pair_feats, kpad_mask = lmask)
        #x = l.transformers[i](x; cond = t_cond, pair_feats = pair_feats, kpad_mask = lmask)
    end
    for i in 1:l.shift_depth
        #x = l.transformers[i + l.depth - l.shift_depth](x; rope=x->l.rope(x, locs), cond = t_cond, pair_feats = pair_feats, kpad_mask = lmask)
        x = l.transformers[i + l.depth - l.shift_depth](x; rope, cond = t_cond, pair_feats = pair_feats, kpad_mask = lmask)
        #x = l.transformers[i + l.depth - l.shift_depth](x; cond = t_cond, pair_feats = pair_feats, kpad_mask = lmask)
        locs += l.loc_shifters[i](x)
        if i < l.shift_depth
            pair_feats = pair_features(locs)
        end
    end
    return (locs, l.d_decoder(x)), l.count_decoder(x)[1,:,:]
end

#P = CoalescentFlow((OUFlow(20f0, 20f0, 0.1f0, -2f0), Flowfusion.DistNoisyInterpolatingDiscreteFlow()), Beta(1,2)) #Note: base process must be tuple
#P = CoalescentFlow((OUFlow(10f0, 5f0, 0.1f0, -2f0), Flowfusion.DistNoisyInterpolatingDiscreteFlow()), Beta(1,2)) #Extreme schedule.
#P = CoalescentFlow((OUFlow(25f0, 100f0, 0.001f0, -2f0), Flowfusion.DistNoisyInterpolatingDiscreteFlow()), Beta(1,2), last_to_nearest_coalescence()) #Extreme schedule.
P = CoalescentFlow((OUFlow(25f0, 100f0, 0.001f0, -2f0), Flowfusion.DistNoisyInterpolatingDiscreteFlow()), Beta(1,2)) #Extreme schedule.

model = Toy(256, 12, shift_depth = 6) |> devi

for l in model.layers.transformers
    l.attention.wo.weight ./= length(model.layers.transformers)
    l.feed_forward.w2.weight ./= length(model.layers.transformers)
end
for l in model.layers.loc_shifters
    l.weight ./= 10
end

@eval Onion begin
    function (rope::MultidimRoPE)(x::AbstractArray, positions::AbstractArray)
        x_4d = glut(x, 4, 3)
        pos_3d = glut(positions, 3, 2)
        D, S, H, B = size(x_4d)
        d_coords, S_pos, B_pos = size(pos_3d)
        @assert S == S_pos && B == B_pos "Sequence length or batch size mismatch between x and positions"
        num_pairs = D ÷ 2
        freqs = @ignore_derivatives 1.0f0 ./ (rope.theta .^ (like(0:2:D-1, x, Float32)[1:num_pairs] ./ D))
        pos_indices = @ignore_derivatives mod1.(like(1:num_pairs, x), d_coords)
        selected_pos = pos_3d[pos_indices, :, :]
        angles = reshape(freqs, num_pairs, 1, 1) .* selected_pos
        cos_vals = cos.(angles)
        sin_vals = sin.(angles)
        cos_vals = reshape(cos_vals, num_pairs, S, 1, B)
        sin_vals = reshape(sin_vals, num_pairs, S, 1, B)
        x1 = x_4d[1:D÷2, :, :, :]
        x2 = x_4d[D÷2+1:end, :, :, :]
        rotated_x = vcat(
            x1 .* cos_vals .- x2 .* sin_vals,
            x2 .* cos_vals .+ x1 .* sin_vals
        )
        return reshape(rotated_x, size(x))
    end
end



#Optimizer:
sched = burnin_learning_schedule(0.0001f0, 0.005f0, 1.15f0, 0.9995f0)
opt_state = Flux.setup(Muon(eta = sched.lr, fallback = x -> (size(x,1) .== 3 || size(x,2) .== 6 || size(x,2) .== 6 || size(x,1) .== 1)), model)





function training_prep()
    t = rand(Float32, 100) .* 0.999f0 .+ 0.0005f0 #To stop Gaussain combine errors - I should really write a shortcircuit for this...
    #For very low temp, distance_weighted_coalescence doesn't coalesce back down to one element. Sounds like a bug that could cause a generator mismatch?
    #bat = branching_bridge(P, X0sampler, [X1target() for _ in 1:100], t, coalescence_factor = 1.0, merger = BranchingFlows.canonical_anchor_merge, coalescence_policy = distance_weighted_coalescence(state_index=1, temperature=0.05, squared=true));
    bat = branching_bridge(P, X0sampler, [X1target() for _ in 1:100], t, coalescence_factor = 1.0);
    splits_target = bat.splits_target
    Xt = bat.Xt.state
    X1targets = bat.X1anchor
    (;t, Xt, X1targets, splits_target, padmask = bat.padmask)
end

function m_wrap(t,Xt)
    X1hat, hat_splits = model(devi([t]),devi(Xt.state))
    return (cpu(ContinuousState(X1hat[1])), cpu(softmax(X1hat[2]))), cpu(hat_splits) #<-Because no batch dim for discrete
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

#for i in 1:iters
for (i, ts) in enumerate(batchloader(; device = devi))
    l,g = Flux.withgradient(model) do m
        X1hat, hat_splits = m(ts.t,ts.Xt)
        mse_loss = floss(P.P[1], X1hat[1], ts.X1targets[1], scalefloss(P.P[1], ts.t, 1, 0.2f0)) * 2
        d_loss = floss(P.P[2], X1hat[2], onehot(ts.X1targets[2]), scalefloss(P.P[2], ts.t, 1, 0.2f0)) / 3 #Add a floss wrapper that calls this onehot automatically.
        splits_loss = floss(P, hat_splits, ts.splits_target, ts.padmask, scalefloss(P, ts.t, 1, 0.2f0)) / 3
        if i % 50 == 0
            println("mse_loss: $mse_loss, d_loss: $d_loss, splits_loss: $splits_loss")
        end
        return mse_loss + d_loss + splits_loss
    end
    Flux.update!(opt_state, model, g[1])
    if mod(i, 10) == 0
        #eta = max(eta - orig_eta/1000, 0.0000000001)
        Flux.adjust!(opt_state, next_rate(sched))
    end
    (i % 50 == 0) && println("i: $i; Loss: $l, eta: $(sched.lr)")
    if i % 1000 == 0
        X0 = BranchingFlows.BranchingState(BranchingFlows.regroup([[X0sampler(nothing) for _ in 1:1]]), [1 ;;]) #Note: You MUST get the batch dimension back in. The model will need it, and the sampler assumes it.
        samp = gen(P, X0, m_wrap, 0f0:0.001f0:1f0)
        println(to_xyz(samp.state[2].state[:], tensor(samp.state[1])[:,:,1]))
    end
end

#jldsave("../examples/qm9_50k_batches.jld", model_state = Flux.state(cpu(model)), opt_state=cpu(opt_state))



X0 = BranchingFlows.BranchingState(BranchingFlows.regroup([[X0sampler(nothing) for _ in 1:1]]), [1 ;;]) #Note: You MUST get the batch dimension back in. The model will need it, and the sampler assumes it.
paths = Tracker()
samp = gen(P, X0, m_wrap, 0f0:0.001f0:1f0, tracker = paths)
println(to_xyz(samp.state[2].state[:], tensor(samp.state[1])[:,:,1]))

samptens = tensor(samp.state[1])[:,:,1]
sampelems = samp.state[2].state[:]
pl = scatter3d(samptens[1,:], samptens[2,:], samptens[3,:], label = :none, marker_z = sampelems .- 1, msw = 0, ms = 7, cmap = :rainbow, colorbar = false)
for i in 1:size(samptens,2)
    for j in 1:size(samptens,2)
        #If the distance between any two non-hydrogens is less than 1.4, plot a line between them
        if sampelems[i] != vocab_dict['H'] && sampelems[j] != vocab_dict['H']
            if sqrt(sum((samptens[:,i] .- samptens[:,j]) .^ 2)) < 1.97
                plot!([samptens[1,i], samptens[1,j]], [samptens[2,i], samptens[2,j]], [samptens[3,i], samptens[3,j]], label = :none, color = "black", linewidth = 2.0, alpha = 0.6)
            end
        end
    end
    #Then plot a line from each hydrogen to the nearest non-hydrogen
    if sampelems[i] == vocab_dict['H']
        non_h = sampelems .!= vocab_dict['H']
        nearest = argmin(sqrt.(sum((samptens[:,i] .- samptens[:,non_h]) .^ 2, dims = 1)[:]))
        plot!([samptens[1,i], samptens[1,nearest]], [samptens[2,i], samptens[2,nearest]], [samptens[3,i], samptens[3,nearest]], label = :none, color = "black", linewidth = 1.5, alpha = 0.6)
    end
end
savefig(pl, "../examples/qm9_sample_$(rand(1000001:9999999)).pdf")
println(to_xyz(samp.state[2].state[:], tensor(samp.state[1])[:,:,1]))
