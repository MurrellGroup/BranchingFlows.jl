using Pkg
Pkg.activate(".")
#Pkg.add(["Flowfusion", "ForwardBackward", "Flux", "Onion", "RandomFeatureMaps", "Distributions", "StatsBase", "Plots", "CUDA", "cuDNN", "CannotWaitForTheseOptimisers"])
Pkg.add("https://github.com/MurrellGroup/BranchingFlows.jl")
Pkg.develop(path = "../")

using BranchingFlows, Flowfusion, ForwardBackward #BF & Friends
using Flux, Onion, RandomFeatureMaps, CannotWaitForTheseOptimisers #DL frameworks, Layers, and Optimizers
using Distributions, StatsBase, Plots #Stats & Plotting

using CUDA
devi = gpu

#Synthetic Data:
xs = -1:0.001:1
f(x) = x+sin(3x)
len() = rand(MixtureModel([Poisson(4), Poisson(20)], [0.5, 0.5])) + 1
bounds() = (-rand(),rand())
function X1target()
    bnds = bounds()
    n = len()
    xpts = bnds[1]:(bnds[2]-bnds[1])/n:bnds[2]
    ypts = f.(xpts)
    discrete_pts = ones(Int, n)
    odd_or_even = (rand() < 0.5) + 1
    discrete_pts[odd_or_even:2:end] .= 2
    masked_continuous = MaskedState(ContinuousState(Float32.(vcat(xpts',ypts'))), trues(n), trues(n)) #Note: must return a tuple of states.
    masked_discrete = MaskedState(DiscreteState(3, discrete_pts), trues(n), trues(n)) #Note: must return a tuple of states.
    X1 = BranchingState((masked_continuous, masked_discrete), ones(Int,n)) #Second argument is "groupings"
    return X1
end

X0sampler(root) = (ContinuousState([1,-1] .+ randn(Float32,2,1)), DiscreteState(3, [3])) #Note: must return a tuple of states. Discrete states must start in the dummy.

struct Toy{L}
    layers::L
end
Flux.@layer Toy
function Toy(dim, depth)
    nheads = 8
    head_dim = 32
    layers = (;
        loc_rff = RandomFourierFeatures(2 => 2dim, 1f0),
        loc_rff2 = RandomFourierFeatures(2 => 2dim, 0.1f0),
        t_rff = RandomFourierFeatures(1 => 4dim, 1f0),
        t_embed = Dense(4dim => dim, bias=false),
        loc_encoder = Dense(4dim+2 => dim, bias=false),
        d_encoder = Embedding(3 => dim),
        rope = RoPE(head_dim, 1000),
        transformers = [Onion.AdaTransformerBlock(dim, dim, nheads; head_dim, qk_norm = true) for _ in 1:depth],
        loc_decoder = Dense(dim => 2, bias=false),
        count_decoder = Dense(dim => 1, bias=false),
        del_decoder = Dense(dim => 1, bias=false),
        d_decoder = Dense(dim => 3, bias=false),
    )
    return Toy(layers)
end
function (m::Toy)(t,bState)
    Xt = bState.state
    l = m.layers
    locs = tensor(Xt[1])
    x = l.loc_encoder(vcat(l.loc_rff(locs),l.loc_rff2(locs),locs)) + l.d_encoder(tensor(Xt[2]))
    t_cond = l.t_embed(l.t_rff(reshape(zero(similar(tensor(Xt[1]), size(tensor(Xt[1]),3))) .+ t, 1, :)))
    rope = l.rope[1:size(locs,2)]
    for layer in l.transformers
        x = layer(x; rope, cond = t_cond, kpad_mask = bState.padmask)
    end
    return (l.loc_decoder(x), l.d_decoder(x)), l.count_decoder(x)[1,:,:], l.del_decoder(x)[1,:,:]
end

P = CoalescentFlow((BrownianMotion(0.05f0), Flowfusion.DistNoisyInterpolatingDiscreteFlow()), Beta(1,2)) #Good for viz
#P = CoalescentFlow((OUFlow(25f0, 100f0, 0.001f0, -2f0), Flowfusion.DistNoisyInterpolatingDiscreteFlow()), Beta(1,3)) #Extreme, but works well!

model = Toy(256, 8) |> devi
for l in model.layers.transformers
    l.attention.wo.weight ./= 10
    l.feed_forward.w2.weight ./= 10
end

#Optimizer:
orig_eta = eta = 0.01
opt_state = Flux.setup(Muon(eta = orig_eta), model)

for i in 1:2000
    t = Uniform(0f0,1f0)
    ts = branching_bridge(P, X0sampler, [X1target() for _ in 1:64], t, use_branching_time_prob = 0.5, deletion_pad = 1.5) |> devi
    l,g = Flux.withgradient(model) do m
        X1hat, hat_splits, hat_del = m(ts.t,ts.Xt)        
        mse_loss = floss(P.P[1], X1hat[1], ts.X1anchor[1], scalefloss(P.P[1], ts.t, 1, 0.2f0)) * 10
        d_loss = floss(P.P[2], X1hat[2], onehot(ts.X1anchor[2]), scalefloss(P.P[2], ts.t, 1, 0.2f0)) / 3 
        splits_loss = floss(P, hat_splits, ts.splits_target, ts.Xt.padmask, scalefloss(P, ts.t, 1, 0.2f0))
        del_loss = floss(P.deletion_policy, hat_del, ts.del, ts.Xt.padmask, scalefloss(P, ts.t, 1, 0.2f0))
        if i % 10 == 0
            println(i, ", mse_loss: $mse_loss, d_loss: $d_loss, splits_loss: $splits_loss, del_loss: $del_loss")
        end
        return mse_loss + d_loss + splits_loss + del_loss
    end
    Flux.update!(opt_state, model, g[1])
    if i > 1500 - 1000
        eta = max(eta - orig_eta/1000, 0.0000000001)
        Flux.adjust!(opt_state, eta)
    end
    tim = time()
end

#Wrap the model to move the state 
function m_wrap(t,Xt)
    X1hat, hat_splits, hat_del = model(devi([t]),devi(Xt))
    return (cpu(ContinuousState(X1hat[1])), cpu(X1hat[2])), cpu(hat_splits), cpu(hat_del)
end

#This will draw one sample.
X0 = branching_bridge(P, X0sampler, [X1target() for _ in 1:1], 0.0f0).Xt
samp = gen(P, X0, m_wrap, 0f0:0.001f0:1f0)
tensor(samp.state[1]), tensor(samp.state[2])

#This will draw 5 samples, and viz them:
for i in 1:5
    #A hack for creating an X0 state:
    X0 = branching_bridge(P, X0sampler, [X1target() for _ in 1:1], 0.0f0).Xt
    paths = Tracker()
    samp = gen(P, X0, m_wrap, 0f0:0.001f0:1f0, tracker = paths)
    pl = plot(colorbar = false, size = (400,350))
    for p in paths.xt
        s = tensor(p[1].state[1])
        scatter!(s[1,:], s[2,:], label = :none, marker_z = (1:size(s,2))./size(s,2), msw = 0, ms = 0.65, cmap = :rainbow)
    end
    endsamp = tensor(samp.state[1])[:,:,1]
    scatter!(endsamp[1,:], endsamp[2,:], label = "Sampled X1", c = samp.state[2].S.state[:] .- 1, msw = 0, ms = 3)
    zerotens = tensor(X0.state[1])
    scatter!(zerotens[1,:], zerotens[2,:], label = "X0", color = "green", msw = 0, ms = 4)
    plot!(xs, f.(xs), color=:black, label = "f")
    pl
    savefig(pl, "$i.pdf")
end
