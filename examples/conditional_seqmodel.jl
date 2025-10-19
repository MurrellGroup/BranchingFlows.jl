#Note, the cmask in X1target, the coalescence_factor, group_mins, and X0 state all need to be selected together to switch between a conditional and unconditional model.

#Branching Flows demo
using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path = "../")

using BranchingFlows
using Flux, Onion, RandomFeatureMaps, StatsBase, Plots, ForwardBackward, Flowfusion, ForwardBackward, Distributions, CannotWaitForTheseOptimisers, LearningSchedules

using CUDA
device!(0) #Because we have set CUDA_VISIBLE_DEVICES = GPUnum
devi = gpu

AAs_withmask = "ACDEFGHIKLMNPQRSTVWY#"
AAs = collect(AAs_withmask[1:end-1])
tok_dict = Dict(AAs[i] => i for i in 1:length(AAs))
reverse_tok_dict = Dict(i => AAs_withmask[i] for i in 1:length(AAs_withmask))
dummy_token = length(AAs_withmask)
tokenize(seq,tok_dict) = [tok_dict[c] for c in seq]
untokenize(inds,reverse_tok_dict) = join([reverse_tok_dict[i] for i in inds])

#Read data, remove X-containing sequences, and adding start and end tokens
data = readlines("abs.txt")
data = [d for d in data if !(occursin("X", d))]

function rand_mask(l)
    if rand() < 0.5
        return trues(l)
    end
    mask = falses(l)
    for _ in 1:(1+rand(Poisson(rand()*6)))
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

function X1target()
    seq = rand(data)
    cmask = rand_mask(length(seq))
    discrete_pts = tokenize(seq,tok_dict)
    disc_state = MaskedState(DiscreteState(dummy_token, discrete_pts), cmask, cmask) #Note: must return a tuple of states.
    X1 = BranchingState((disc_state,), ones(Int,length(seq)), branchmask = cmask, flowmask = cmask) #Second argument is "groupings"
    X1 = uniform_del_insertions(X1, 0.3)
    #X1.state[1].S.state[X1.del] .= dummy_token #Optional: Set deleted discrete components to the dummy!
    return X1
end

X0sampler(root) = (DiscreteState(dummy_token, [dummy_token]),)

struct Toy{L}
    layers::L
end
Flux.@layer Toy
function Toy(dim, depth)
    nheads = 16
    head_dim = 32
    layers = (;
        mask_encoder = Embedding(2 => dim),
        t_rff = RandomFourierFeatures(1 => dim, 1f0),
        t_embed = Dense(dim => dim, bias=false),
        rope = RoPE(head_dim, 1000),
        d_encoder = Embedding(dummy_token => dim),
        transformers = [Onion.AdaTransformerBlock(dim, dim, nheads; head_dim, qk_norm = true) for _ in 1:depth],
        d_decoder = Dense(dim => dummy_token, bias=false),
        count_decoder = Dense(dim => 1, bias=false),
        del_decoder = Dense(dim => 1, bias=false),
    )
    return Toy(layers)
end
function (m::Toy)(t,Xt)
    l = m.layers
    seqs = tensor(Xt.state[1])
    x = l.d_encoder(seqs) .+ l.mask_encoder(Xt.flowmask .+ 1)
    t_cond = l.t_embed(l.t_rff(reshape(zero(similar(x, size(seqs,2))) .+ t, 1, :)))
    rope = l.rope[1:size(seqs,1)]
    for layer in l.transformers
        x = layer(x; rope, cond = t_cond, kpad_mask = Xt.padmask)
    end
    return (l.d_decoder(x), ), l.count_decoder(x)[1,:,:], l.del_decoder(x)[1,:,:]
end

P = CoalescentFlow((Flowfusion.DistNoisyInterpolatingDiscreteFlow(D1=Beta(1.5,1.5)),), Beta(1,2), SequentialUniform())

model = Toy(512, 12) |> devi
model.layers.t_embed.weight ./= 12
for l in model.layers.transformers
    l.attention.wo.weight ./= 12
    l.feed_forward.w2.weight ./= 12
end
#Optimizer:
sched = burnin_learning_schedule(0.000001f0, 0.00050f0, 1.05f0, 0.99995f0)
opt_state = Flux.setup(Muon(eta = sched.lr), model)

function training_prep(; batch_size = 32)
    t = Uniform(0f0,1f0)
    bat = branching_bridge(P, X0sampler, [X1target() for _ in 1:batch_size], t, coalescence_factor = 1.0, use_branching_time_prob = 0.5)
    return (;t = bat.t, Xt = bat.Xt, X1targets = bat.X1anchor, splits_target = bat.splits_target, del = bat.del)
end

Flux.MLDataDevices.Internal.unsafe_free!(x) = (Flux.fmapstructure(Flux.MLDataDevices.Internal.unsafe_free_internal!, x); return nothing)

iters = 150000
struct BatchDataset end
Base.length(x::BatchDataset) = iters
Base.getindex(x::BatchDataset, i) = training_prep()

function batchloader(; device=identity, parallel=true)
    x = BatchDataset()
    dataloader = Flux.DataLoader(x; batchsize=-1, parallel)
    return device(dataloader)
end

function m_wrap(t,Xt)
    X1hat, hat_splits, hat_del = model(devi([t]),devi(Xt))
    println(replace(untokenize(cpu(Xt.state[1].S.state), reverse_tok_dict), "#" => "-"))
    return (cpu(X1hat[1]),), cpu(hat_splits), cpu(hat_del) #<-Because no batch dim for discrete
end

tim = time()
for (i, ts) in enumerate(batchloader(; device = devi))
    (i % 100 == 0) && println("Batch $i, time: $(time() - tim)")
    l,g = Flux.withgradient(model) do m
        X1hat, hat_splits, hat_del = m(ts.t,ts.Xt)        
        d_loss = floss(P.P[1], X1hat[1], onehot(ts.X1targets[1]), scalefloss(P.P[1], ts.t, 1, 0.2f0)) / 3 #Add a floss wrapper that calls this onehot automatically.
        splits_loss = floss(P, hat_splits, ts.splits_target, ts.Xt.padmask .* ts.Xt.branchmask, scalefloss(P, ts.t, 1, 0.2f0))
        del_loss = floss(P.deletion_policy, hat_del, ts.del, ts.Xt.padmask .* ts.Xt.branchmask, scalefloss(P, ts.t, 1, 0.2f0))
        if i % 100 == 0
            println("d_loss: $d_loss, splits_loss: $splits_loss, del_loss: $del_loss")
        end
        return d_loss + splits_loss + del_loss
    end
    Flux.update!(opt_state, model, g[1])
    if mod(i, 10) == 0
        Flux.adjust!(opt_state, next_rate(sched))
    end
    (i % 100 == 0) && println("i: $i; Loss: $l, eta: $(sched.lr)")
    if i % 500 == 0
        for _ in 1:5
            try
                X0 = branching_bridge(P, X0sampler, [X1target() for _ in 1:1], [0.000000001f0], coalescence_factor = 1.0, use_branching_time_prob = 0.0).Xt
                samp = gen(P, X0, m_wrap, 0f0:0.005f0:1f0)
                println(replace(untokenize(cpu(samp.state[1].S.state), reverse_tok_dict), "#" => "-"))
                println("\n\n\n")
            catch
            end
        
        end
    end
    tim = time()
end

