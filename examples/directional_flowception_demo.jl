@assert VERSION >= v"1.11"

using Pkg

env_or(primary, fallback, default) = get(ENV, primary, get(ENV, fallback, default))

const skip_pkg = env_or("DIRECTIONAL_FLOWCEPTION_DEMO_SKIP_PKG", "FLOWCEPTION_DEMO_SKIP_PKG", "false") == "true"

if !skip_pkg
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
    !skip_pkg && Pkg.add(["CUDA", "cuDNN"])
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

function parse_target_mode(kind::AbstractString)
    normalized = lowercase(strip(kind))
    if normalized in ("count", "counts", "legacy")
        return CountRevealTarget()
    elseif normalized in ("sparse", "next-slot", "next_slot")
        return SparseRevealTarget()
    elseif normalized in ("rb", "rao-blackwell", "rao_blackwell", "rao-blackwellized", "rao_blackwellized")
        return RaoBlackwellizedRevealTarget()
    end
    error("Unknown directional Flowception demo target mode `$kind`. Use `count`, `sparse`, or `rb`.")
end

function parse_reveal_order(kind::AbstractString, temperature)
    normalized = lowercase(strip(kind))
    if normalized in ("independent", "iid", "default")
        return IndependentRevealOrder()
    elseif normalized in ("reveal-order", "reveal_order", "ordered", "seeded", "centerout", "centreout")
        return SeededRevealOrder(
            temperature = temperature,
            target = parse_target_mode(env_or("DIRECTIONAL_FLOWCEPTION_DEMO_TARGET_MODE", "FLOWCEPTION_DEMO_TARGET_MODE", "rb")),
        )
    end
    error("Unknown directional Flowception demo reveal order `$kind`. Use `independent` or `reveal-order`.")
end

function clear_old_directional_flowception_outputs()
    for name in readdir(@__DIR__)
        occursin(r"^directional_flowception_demo_.*\.(pdf|png|mp4)$", name) || continue
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
    rope_length = parse(Int, env_or("DIRECTIONAL_FLOWCEPTION_DEMO_ROPE_LENGTH", "FLOWCEPTION_DEMO_ROPE_LENGTH", "4096"))
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
        rope = RoPE(head_dim, rope_length),
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
    total_time = parse(Float32, env_or("DIRECTIONAL_FLOWCEPTION_DEMO_TOTAL_TIME", "FLOWCEPTION_DEMO_TOTAL_TIME", "2")),
    reveal_order = parse_reveal_order(
        env_or("DIRECTIONAL_FLOWCEPTION_DEMO_REVEAL_ORDER", "FLOWCEPTION_DEMO_REVEAL_ORDER", "independent"),
        parse(Float32, env_or("DIRECTIONAL_FLOWCEPTION_DEMO_REVEAL_TEMPERATURE", "FLOWCEPTION_DEMO_REVEAL_TEMPERATURE", "0.25")),
    ),
)
nstart = parse(Int, env_or("DIRECTIONAL_FLOWCEPTION_DEMO_NSTART", "FLOWCEPTION_DEMO_NSTART", "1"))

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
    ts = directional_flowception_bridge(P, [X1target() for _ in 1:batchsize], Uniform(0f0, P.total_time), nstart = nstart) |> dev
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
animation_max_frames = parse(Int, env_or("DIRECTIONAL_FLOWCEPTION_DEMO_ANIMATION_MAX_FRAMES", "FLOWCEPTION_DEMO_ANIMATION_MAX_FRAMES", "300"))
animation_fps = parse(Int, env_or("DIRECTIONAL_FLOWCEPTION_DEMO_ANIMATION_FPS", "FLOWCEPTION_DEMO_ANIMATION_FPS", "20"))

function token_colors(X::FlowceptionState)
    return vec(X.state[2].S.state) .- 1
end

function final_points(X::FlowceptionState)
    return tensor(X.state[1])[:, :, 1]
end

function final_tokens(X::FlowceptionState)
    return vec(X.state[2].S.state[:, 1])
end

function curve_mse(X::FlowceptionState)
    pts = final_points(X)
    isempty(pts) && return Inf32
    finite = vec(all(isfinite, pts; dims = 1))
    any(finite) || return Inf32
    x = pts[1, finite]
    y = pts[2, finite]
    return mean((y .- f.(x)) .^ 2)
end

function alternating_violations(tokens::AbstractVector{<:Integer})
    length(tokens) <= 1 && return 0
    return count(identity, tokens[2:end] .== tokens[1:end-1])
end

function full_plot_limits(X0, path_states, X)
    xmin = minimum(xs)
    xmax = maximum(xs)
    ymin = minimum(f.(xs))
    ymax = maximum(f.(xs))

    pts = tensor(X0.state[1])
    xmin = min(xmin, minimum(pts[1, :]))
    xmax = max(xmax, maximum(pts[1, :]))
    ymin = min(ymin, minimum(pts[2, :]))
    ymax = max(ymax, maximum(pts[2, :]))
    for p in path_states
        pts = tensor(p[1].state[1])
        xmin = min(xmin, minimum(pts[1, :]))
        xmax = max(xmax, maximum(pts[1, :]))
        ymin = min(ymin, minimum(pts[2, :]))
        ymax = max(ymax, maximum(pts[2, :]))
    end
    pts = tensor(X.state[1])[:, :, 1]
    xmin = min(xmin, minimum(pts[1, :]))
    xmax = max(xmax, maximum(pts[1, :]))
    ymin = min(ymin, minimum(pts[2, :]))
    ymax = max(ymax, maximum(pts[2, :]))

    xpad = max((xmax - xmin) * 0.05, 0.05)
    ypad = max((ymax - ymin) * 0.05, 0.05)
    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)
end

function render_demo_plot(X0, X, path_prefix = nothing; current_time = nothing, total_time = nothing, xlims = nothing, ylims = nothing)
    pl = plot(colorbar = false, size = (400, 350), xlims = xlims, ylims = ylims)
    if !isnothing(path_prefix)
        for p in path_prefix
            s = tensor(p[1].state[1])
            scatter!(pl, s[1, :], s[2, :], label = :none, marker_z = (1:size(s, 2)) ./ size(s, 2), msw = 0, ms = 0.65, cmap = :rainbow)
        end
    end
    current = tensor(X.state[1])[:, :, 1]
    scatter!(pl, current[1, :], current[2, :], label = "Sampled X1", c = token_colors(X), msw = 0, ms = 3)
    zerotens = tensor(X0.state[1])
    scatter!(pl, zerotens[1, :], zerotens[2, :], label = "X0", color = "green", msw = 0, ms = 4)
    plot!(pl, xs, f.(xs), color = :black, label = "f")
    if !isnothing(current_time)
        plot!(pl, title = "absolute time = $(round(current_time; digits = 2)) / $(round(total_time; digits = 2))")
    end
    return pl
end

function animation_frame_indices(nstates, max_frames)
    nstates == 0 && return Int[]
    return unique(round.(Int, range(1, nstates, length = min(max(nstates, 1), max(max_frames, 1)))))
end

sample_curve_mse = Float32[]
sample_alt_violations = Int[]
sample_dummy_counts = Int[]

for sample_ix in 1:nsamples
    X0 = make_source(P; nstart)
    paths = Tracker()
    steps = collect(0f0:sample_dt:P.total_time)
    samp = gen(P, X0, model_wrap, steps, tracker = paths)

    pts = final_points(samp)
    toks = final_tokens(samp)
    mse = curve_mse(samp)
    alt_viol = alternating_violations(toks)
    dummy_count = count(==(3), toks)
    push!(sample_curve_mse, mse)
    push!(sample_alt_violations, alt_viol)
    push!(sample_dummy_counts, dummy_count)

    println("sample=$sample_ix final_length=$(size(pts, 2))")
    println("sample=$sample_ix final_tokens=$toks")
    println("sample=$sample_ix curve_mse=$mse alternating_violations=$alt_viol dummy_tokens=$dummy_count")

    xlims, ylims = full_plot_limits(X0, paths.xt, samp)
    pl = render_demo_plot(X0, samp, paths.xt; xlims, ylims)
    outfile = joinpath(@__DIR__, "directional_flowception_demo_sample_$(sample_ix).pdf")
    savefig(pl, outfile)
    println("saved=$outfile")

    frame_indices = animation_frame_indices(length(paths.xt), animation_max_frames)
    anim = Animation()
    for frame_ix in frame_indices
        frame_state = paths.xt[frame_ix][1]
        frame_time = steps[min(frame_ix + 1, length(steps))]
        frame(anim, render_demo_plot(X0, frame_state, view(paths.xt, 1:frame_ix); current_time = frame_time, total_time = P.total_time, xlims, ylims))
    end
    mp4file = joinpath(@__DIR__, "directional_flowception_demo_sample_$(sample_ix).mp4")
    mp4(anim, mp4file; fps = animation_fps, show_msg = false)
    println("saved=$mp4file")
end

println(
    "summary mean_curve_mse=$(mean(sample_curve_mse)) " *
    "mean_alternating_violations=$(mean(sample_alt_violations)) " *
    "mean_dummy_tokens=$(mean(sample_dummy_counts))",
)
