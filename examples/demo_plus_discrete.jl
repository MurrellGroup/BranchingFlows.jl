#Branching Flows demo
using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path = "../")

using BranchingFlows
using Flux, Onion, RandomFeatureMaps, StatsBase, Plots, ForwardBackward, Flowfusion, ForwardBackward, Distributions, CannotWaitForTheseOptimisers

using CUDA
device!(0) #Because we have set CUDA_VISIBLE_DEVICES = GPUnum
devi = gpu

#Data draw:
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
    return BranchingState((masked_continuous, masked_discrete), ones(Int,n)) #Second argument is "groupings"
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
        loc_rff = RandomFourierFeatures(2 => dim, 1f0),
        t_rff = RandomFourierFeatures(1 => dim, 1f0),
        t_embed = Dense(dim => dim, bias=false),
        loc_encoder = Dense(dim+2 => dim, bias=false),
        d_encoder = Embedding(3 => dim),
        rope = RoPE(head_dim, 1000),
        transformers = [Onion.AdaTransformerBlock(dim, dim, nheads; head_dim, qk_norm = true) for _ in 1:depth],
        loc_decoder = Dense(dim => 2, bias=false),
        count_decoder = Dense(dim => 1, bias=false),
        d_decoder = Dense(dim => 3, bias=false),
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
    x = l.loc_encoder(vcat(l.loc_rff(locs),locs)) + l.d_encoder(tensor(Xt[2]))
    t_cond = l.t_embed(l.t_rff(reshape(zero(similar(tensor(Xt[1]), size(tensor(Xt[1]),3))) .+ t, 1, :))) #Because "gen" will pass a scalar t, but we train with each batch having its own t.
    rope = l.rope[1:size(locs,2)]
    for layer in l.transformers
        x = layer(x; rope, cond = t_cond, kpad_mask = lmask)
    end
    return (l.loc_decoder(x), l.d_decoder(x)), l.count_decoder(x)[1,:,:]
end

#Note: base process must be tuple (for now)
#P = CoalescentFlow((OUFlow(10f0, 5f0, 0.01f0, -2f0), Flowfusion.DistNoisyInterpolatingDiscreteFlow()), Beta(2,2)) #This is a good flow to use - gets around the terminal mean quickly.
#P = CoalescentFlow((Deterministic(), Flowfusion.DistNoisyInterpolatingDiscreteFlow()), Beta(1,2)) #Deterministic hurts the model.
P = CoalescentFlow((BrownianMotion(0.05f0), Flowfusion.DistNoisyInterpolatingDiscreteFlow()), Beta(1,2)) #For seeing the splits the clearest.
#P = CoalescentFlow((OUFlow(10f0, 0.2f0, 0.01f0, -2f0), Flowfusion.DistNoisyInterpolatingDiscreteFlow()), Beta(1,5)) #Extreme schedule.

model = Toy(384, 5) |> devi

for l in model.layers.transformers
    l.attention.wo.weight ./= 10
    l.feed_forward.w2.weight ./= 10
end

#Optimizer:
orig_eta = eta = 0.01
opt_state = Flux.setup(Muon(eta = orig_eta), model)

iters = 1500
for i in 1:iters
    t = rand(Float32, 60)
    bat = branching_bridge(P, X0sampler, [X1target() for _ in 1:60], t, coalescence_factor = 1.0, merger = BranchingFlows.select_anchor_merge);
    splits_target = bat.splits_target
    Xt = bat.Xt.state
    X1targets = bat.X1anchor
    t, Xt, X1targets, splits_target, padmask = (t, Xt, X1targets, splits_target, bat.padmask) |> devi
    l,g = Flux.withgradient(model) do m
        X1hat, hat_splits = m(t,Xt)        
        mse_loss = floss(P.P[1], X1hat[1], X1targets[1], scalefloss(P.P[1], t, 1, 0.2f0)) * 10
        d_loss = floss(P.P[2], X1hat[2], onehot(X1targets[2]), scalefloss(P.P[2], t, 1, 0.2f0)) / 3 #Add a floss wrapper that calls this onehot automatically.
        splits_loss = floss(P, hat_splits, splits_target, padmask, scalefloss(P, t, 1, 0.2f0)) / 5
        if i % 10 == 0
            println("mse_loss: $mse_loss, d_loss: $d_loss, splits_loss: $splits_loss")
        end
        return mse_loss + d_loss + splits_loss
    end
    Flux.update!(opt_state, model, g[1])
    if i > iters - 1000
        eta = max(eta - orig_eta/1000, 0.0000000001)
        Flux.adjust!(opt_state, eta)
    end
    (i % 10 == 0) && println("i: $i; Loss: $l, eta: $eta")
end

function m_wrap(t,Xt)
    X1hat, hat_splits = model(devi([t]),devi(Xt.state))
    return (cpu(ContinuousState(X1hat[1])), cpu(softmax(X1hat[2]))), cpu(hat_splits) #<-Because no batch dim for discrete
end

for _ in 1:10
    X0 = BranchingFlows.BranchingState(BranchingFlows.regroup([[X0sampler(nothing) for _ in 1:1]]), [1 ;;]) #Note: You MUST get the batch dimension back in. The model will need it, and the sampler assumes it.
    paths = Tracker()
    samp = gen(P, X0, m_wrap, 0f0:0.005f0:1f0, tracker = paths)
    pl = plot(colorbar = false, size = (400,350))
    for p in paths.xt
        s = tensor(p[1].state[1])
        scatter!(s[1,:], s[2,:], label = :none, marker_z = (1:size(s,2))./size(s,2), msw = 0, ms = 0.75, cmap = :rainbow)
    end
    endsamp = tensor(samp.state[1])[:,:,1]
    plot!(endsamp[1,:], endsamp[2,:], label = :none, color = "red")
    scatter!(endsamp[1,:], endsamp[2,:], label = "Sampled X1", c = samp.state[2].state[:] .- 1, msw = 0, ms = 3)
    zerotens = tensor(X0.state[1])
    scatter!(zerotens[1,:], zerotens[2,:], label = "X0", color = "green", msw = 0, ms = 4)
    plot!(xs, f.(xs), color=:black, label = "f")
    pl
    savefig(pl, "../examples/coaltimeshift_$(rand(1000001:9999999)).pdf")
end

#Connecting with lines
X0 = BranchingFlows.BranchingState(BranchingFlows.regroup([[X0sampler(nothing) for _ in 1:1]]), [1 ;;]) #Note: You MUST get the batch dimension back in. The model will need it, and the sampler assumes it.
paths = Tracker()
samp = gen(P, X0, m_wrap, 0f0:0.001f0:1f0, tracker = paths)
pl = plot(colorbar = false, size = (400,350))
current_s = tensor(paths.xt[1][1].state[1])
for p in paths.xt
    s = tensor(p[1].state[1])
    if size(s,2) == size(current_s,2)
        for i in 1:size(s,2)
            plot!([current_s[1,i], s[1,i]], [current_s[2,i], s[2,i]], label = :none, line_z = i, cmap = :rainbow, linewidth = 0.2)
        end
    else
        scatter!(s[1,:], s[2,:], label = :none, marker_z = 1:size(s,2), msw = 0, ms = 0.75, cmap = :rainbow)
    end
    current_s = s
end
endsamp = tensor(samp.state[1])[:,:,1]
plot!(endsamp[1,:], endsamp[2,:], label = :none, color = "red", linestyle = :dot)
scatter!(endsamp[1,:], endsamp[2,:], label = "Sampled X1", c = samp.state[2].state[:] .- 1, msw = 0, ms = 3)
zerotens = tensor(X0.state[1])
scatter!(zerotens[1,:], zerotens[2,:], label = "X0", color = "green", msw = 0, ms = 4)
plot!(xs, f.(xs), color=:black, label = "f")
pl
savefig(pl, "../examples/connecting_lines_sample_$(rand(1000001:9999999)).pdf")


#Histogram check - note: needs a very long training run to converge to a good histogram
sizs = [length(gen(P, BranchingState(X0sampler(nothing), ones(Int,1)), m_wrap, 0f0:0.001f0:1f0).groupings) for _ in 1:1000];
pl = histogram([length(X1target().groupings) for _ in 1:50000], bins = 0:1:42, normalize = :probability, color = "blue", alpha = 0.5, 
label = "Target lengths", xlabel = "Length", ylabel = "Probability", size = (600,300), margins = 5Plots.mm)
histogram!(sizs, label = "Sampled lengths", bins = 0:1:42, alpha = 0.5, normalize = :probability, color = "red")
pl
savefig(pl, "length_matching_poisson_mix_betacoaltimes.pdf")

