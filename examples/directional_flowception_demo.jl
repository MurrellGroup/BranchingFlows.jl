@assert VERSION >= v"1.11"

using Pkg

env_or(primary, fallback, default) = get(ENV, primary, get(ENV, fallback, default))

if env_or("DIRECTIONAL_FLOWCEPTION_DEMO_SKIP_PKG", "FLOWCEPTION_DEMO_SKIP_PKG", "false") != "true"
    Pkg.activate(temp = true)
    Registry.add(RegistrySpec(url = "https://github.com/MurrellGroup/MurrellGroupRegistry"))
    Pkg.add([
        "Flowfusion",
        "ForwardBackward",
        "Flux",
        "RandomFeatureMaps",
        "Distributions",
        "StatsBase",
        "Plots",
        "CannotWaitForTheseOptimisers",
    ])
    Pkg.add(url = "https://github.com/MurrellGroup/ONIONop.jl.git", rev = "fix-flash-attention-padding")
    Pkg.add(url = "https://github.com/MurrellGroup/Onion.jl", rev = "proteins")
    Pkg.develop(path = joinpath(@__DIR__, ".."))
end

ENV["CUDA_VISIBLE_DEVICES"] = env_or("DIRECTIONAL_FLOWCEPTION_DEMO_CUDA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "1")
use_gpu = env_or("DIRECTIONAL_FLOWCEPTION_DEMO_NO_CUDA", "FLOWCEPTION_DEMO_NO_CUDA", "false") == "false"
if use_gpu
    Pkg.add(["CUDA", "cuDNN"])
    using CUDA, cuDNN
end

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
ENV["DISPLAY"] = get(ENV, "DISPLAY", "")

using BranchingFlows, Flowfusion, ForwardBackward
using Flux, RandomFeatureMaps, Onion
using CannotWaitForTheseOptimisers
using Distributions, StatsBase
using Plots

gr()

const dev = use_gpu ? gpu : cpu

function clear_old_directional_flowception_outputs()
    for name in readdir(@__DIR__)
        occursin(r"^directional_flowception_demo_.*\.(pdf|png)$", name) || continue
        rm(joinpath(@__DIR__, name); force = true)
    end
end

clear_old_directional_flowception_outputs()

xs = -1:0.001:1
f(x) = x + sin(3x)
len() = rand(MixtureModel([Poisson(4), Poisson(18)], [0.55, 0.45])) + 2
bounds() = (-rand(), rand())

function X1target()
    bnds = bounds()
    n = len()
    xpts = range(bnds[1], bnds[2], length = n)
    ypts = f.(xpts)
    discrete_pts = ones(Int, n)
    phase = (rand() < 0.5) + 1
    discrete_pts[phase:2:end] .= 2
    masked_continuous = MaskedState(ContinuousState(Float32.(vcat(xpts', ypts'))), trues(n), trues(n))
    masked_discrete = MaskedState(DiscreteState(3, discrete_pts), trues(n), trues(n))
    return FlowceptionState((masked_continuous, masked_discrete), ones(Int, n))
end

birth_sampler(_) = (
    ContinuousState(Float32[1, -1] .+ randn(Float32, 2, 1)),
    DiscreteState(3, [3]),
)

struct ToyDirectionalFlowception{L}
    layers::L
end

Flux.@layer ToyDirectionalFlowception

function ToyDirectionalFlowception(dim, depth)
    nheads = 8
    head_dim = dim ÷ nheads
    layers = (;
        loc_rff = RandomFourierFeatures(2 => 2dim, 1f0),
        loc_rff2 = RandomFourierFeatures(2 => 2dim, 0.1f0),
        local_t_rff = RandomFourierFeatures(1 => dim, 1f0),
        global_t_rff = RandomFourierFeatures(1 => dim, 1f0),
        global_t_embed = Dense(dim => dim, bias = false),
        loc_encoder = Dense(5dim + 4 => dim, bias = false),
        disc_encoder = Embedding(3 => dim),
        branch_encoder = Embedding(2 => dim),
        flow_encoder = Embedding(2 => dim),
        rope = RoPE(head_dim, 1000),
        transformers = [Onion.AdaTransformerBlock(dim, dim, nheads; head_dim, qk_norm = true) for _ in 1:depth],
        loc_decoder = Dense(dim => 2, bias = false),
        disc_decoder = Dense(dim => 3, bias = false),
        insertion_decoder = Dense(dim => 2),
    )
    return ToyDirectionalFlowception(layers)
end

function (m::ToyDirectionalFlowception)(t, X::FlowceptionState)
    Xt = X.state
    l = m.layers
    locs = tensor(Xt[1])
    disc = tensor(Xt[2])
    local_t = reshape(X.local_t, 1, size(X.local_t)...)
    branchmask = reshape(Float32.(X.branchmask), 1, size(X.branchmask)...)
    flowmask = reshape(Float32.(X.flowmask), 1, size(X.flowmask)...)
    x = l.loc_encoder(vcat(
        l.loc_rff(locs),
        l.loc_rff2(locs),
        locs,
        l.local_t_rff(local_t),
        branchmask,
        flowmask,
    )) +
        l.disc_encoder(disc) +
        l.branch_encoder(Int.(X.branchmask) .+ 1) +
        l.flow_encoder(Int.(X.flowmask) .+ 1)
    t_cond = l.global_t_embed(l.global_t_rff(reshape(t, 1, :)))
    rope = l.rope[1:size(locs, 2)]
    for layer in l.transformers
        x = layer(x; rope, cond = t_cond, kpad_mask = X.padmask)
    end
    return (l.loc_decoder(x), l.disc_decoder(x)), l.insertion_decoder(x)
end

function make_source(P::DirectionalFlowceptionFlow; nstart = 1, T = Float32)
    births = [birth_sampler(nothing) for _ in 1:nstart]
    mask = trues(nstart, 1)
    state = MaskedState.(Flowfusion.regroup([births]), (mask,), (mask,))
    return FlowceptionState(state, ones(Int, nstart, 1);
        local_t = zeros(T, nstart, 1),
        branchmask = copy(mask),
        flowmask = copy(mask),
        padmask = copy(mask),
    )
end

P = DirectionalFlowceptionFlow(
    (BrownianMotion(0.05f0), Flowfusion.DistNoisyInterpolatingDiscreteFlow()),
    birth_sampler,
    insertion_transform = x -> exp.(clamp.(x, -20, 6)),
)

model_dim = parse(Int, env_or("DIRECTIONAL_FLOWCEPTION_DEMO_DIM", "FLOWCEPTION_DEMO_DIM", "256"))
model_depth = parse(Int, env_or("DIRECTIONAL_FLOWCEPTION_DEMO_DEPTH", "FLOWCEPTION_DEMO_DEPTH", "8"))
model = ToyDirectionalFlowception(model_dim, model_depth) |> dev
for layer in model.layers.transformers
    layer.attention.wo.weight ./= 10
    layer.feed_forward.w2.weight ./= 10
end
model.layers.insertion_decoder.weight ./= 10
model.layers.insertion_decoder.bias .= -4f0

orig_eta = eta = parse(Float64, env_or("DIRECTIONAL_FLOWCEPTION_DEMO_LR", "FLOWCEPTION_DEMO_LR", "0.003"))
opt_state = Flux.setup(Muon(eta = orig_eta), model)

iters = parse(Int, env_or("DIRECTIONAL_FLOWCEPTION_DEMO_ITERS", "FLOWCEPTION_DEMO_ITERS", "2000"))
batchsize = parse(Int, env_or("DIRECTIONAL_FLOWCEPTION_DEMO_BATCH", "FLOWCEPTION_DEMO_BATCH", "64"))
warmdown_steps = parse(Int, env_or("DIRECTIONAL_FLOWCEPTION_DEMO_WARMDOWN_STEPS", "FLOWCEPTION_DEMO_WARMDOWN_STEPS", "1000"))
warmdown_start = max(iters - warmdown_steps, 0)

for i in 1:iters
    ts = directional_flowception_bridge(P, [X1target() for _ in 1:batchsize], Uniform(0f0, 2f0), nstart = 1) |> dev
    loss, (∇model,) = Flux.withgradient(model) do m
        X1hat, hat_insertions = m(ts.t, ts.Xt)
        global_t = reshape(ts.t, 1, :)
        loc_loss = floss(P.P[1], X1hat[1], ts.X1anchor[1], scalefloss(P.P[1], ts.Xt.local_t, 1, 0.2f0)) * 10
        disc_loss = floss(P.P[2], X1hat[2], onehot(ts.X1anchor[2]), scalefloss(P.P[2], ts.Xt.local_t, 1, 0.2f0)) / 3
        insertion_loss = floss(P, hat_insertions, ts.insertions_target, ts.Xt, scalefloss(P, global_t, 1, 0.2f0))
        if i % 25 == 0
            println("iter=$i loc_loss=$loc_loss disc_loss=$disc_loss insertion_loss=$insertion_loss")
        end
        return loc_loss + disc_loss + insertion_loss
    end
    Flux.update!(opt_state, model, ∇model)
    if i > warmdown_start
        global eta
        eta = max(eta - orig_eta / max(warmdown_steps, 1), 1e-10)
        Flux.adjust!(opt_state, eta)
    end
end

function model_wrap(t, Xt)
    X1hat, hat_insertions = model(dev([t]), dev(Xt))
    return (cpu(ContinuousState(X1hat[1])), cpu(X1hat[2])), cpu(hat_insertions)
end

nsamples = parse(Int, env_or("DIRECTIONAL_FLOWCEPTION_DEMO_NSAMPLES", "FLOWCEPTION_DEMO_NSAMPLES", "5"))
sample_dt = parse(Float32, env_or("DIRECTIONAL_FLOWCEPTION_DEMO_SAMPLE_DT", "FLOWCEPTION_DEMO_SAMPLE_DT", "0.002"))

for sample_ix in 1:nsamples
    X0 = make_source(P)
    paths = Tracker()
    samp = gen(P, X0, model_wrap, 0f0:sample_dt:2f0, tracker = paths)

    println("sample=$sample_ix final_length=$(size(tensor(samp.state[1]), 2))")
    println("sample=$sample_ix final_tokens=$(vec(samp.state[2].S.state[:, 1]))")

    pl = plot(colorbar = false, size = (400, 350))
    for p in paths.xt
        s = tensor(p[1].state[1])
        scatter!(pl, s[1, :], s[2, :], label = :none, marker_z = (1:size(s, 2)) ./ size(s, 2), msw = 0, ms = 0.65, cmap = :rainbow)
    end
    endsamp = tensor(samp.state[1])[:, :, 1]
    scatter!(pl, endsamp[1, :], endsamp[2, :], label = "Sampled X1", c = samp.state[2].S.state[:] .- 1, msw = 0, ms = 3)
    zerotens = tensor(X0.state[1])
    scatter!(pl, zerotens[1, :], zerotens[2, :], label = "X0", color = "green", msw = 0, ms = 4)
    plot!(pl, xs, f.(xs), color = :black, label = "f")

    outfile = joinpath(@__DIR__, "directional_flowception_demo_sample_$(sample_ix).pdf")
    savefig(pl, outfile)
    println("saved=$outfile")
end
