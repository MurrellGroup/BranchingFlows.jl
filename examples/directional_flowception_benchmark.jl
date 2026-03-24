@assert VERSION >= v"1.11"

using Pkg

env_or(primary, fallback, default) = get(ENV, primary, get(ENV, fallback, default))

const skip_pkg = env_or("DIRECTIONAL_FLOWCEPTION_BENCHMARK_SKIP_PKG", "DIRECTIONAL_FLOWCEPTION_DEMO_SKIP_PKG", "false") == "true"

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

ENV["CUDA_VISIBLE_DEVICES"] = env_or("DIRECTIONAL_FLOWCEPTION_BENCHMARK_CUDA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "1")
use_gpu = env_or("DIRECTIONAL_FLOWCEPTION_BENCHMARK_NO_CUDA", "DIRECTIONAL_FLOWCEPTION_DEMO_NO_CUDA", "false") == "false"
if use_gpu
    !skip_pkg && Pkg.add(["CUDA", "cuDNN"])
    using CUDA, cuDNN
end

using BranchingFlows, Flowfusion, ForwardBackward
using Flux, RandomFeatureMaps, Onion
using CannotWaitForTheseOptimisers
using Distributions, StatsBase, Random
using Dates, Serialization, Statistics

const dev = use_gpu ? gpu : cpu

function parse_reveal_order(kind::AbstractString, temperature)
    normalized = lowercase(strip(kind))
    if normalized in ("independent", "iid", "default")
        return IndependentRevealOrder()
    elseif normalized in ("reveal-order", "reveal_order", "ordered", "seeded", "centerout", "centreout")
        return SeededRevealOrder(temperature = temperature, target = parse_target_mode(env_or("DIRECTIONAL_FLOWCEPTION_BENCHMARK_TARGET_MODE", "DIRECTIONAL_FLOWCEPTION_DEMO_TARGET_MODE", "rb")))
    end
    error("Unknown reveal order `$kind`. Use `independent` or `reveal-order`.")
end

function parse_target_mode(kind::AbstractString)
    normalized = lowercase(strip(kind))
    if normalized in ("count", "counts", "legacy")
        return CountRevealTarget()
    elseif normalized in ("sparse", "next-slot", "next_slot")
        return SparseRevealTarget()
    elseif normalized in ("rb", "rao-blackwell", "rao_blackwell", "rao-blackwellized", "rao_blackwellized")
        return RaoBlackwellizedRevealTarget()
    end
    error("Unknown structured target mode `$kind`. Use `count`, `sparse`, or `rb`.")
end

const xs = -1:0.001:1
f(x) = x + sin(3x)
len_dist() = MixtureModel([Poisson(4), Poisson(18)], [0.55, 0.45])
len() = rand(len_dist()) + 2
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
    rope_length = parse(Int, env_or("DIRECTIONAL_FLOWCEPTION_BENCHMARK_ROPE_LENGTH", "DIRECTIONAL_FLOWCEPTION_DEMO_ROPE_LENGTH", "4096"))
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

function analytic_length_pmf(max_length::Int)
    d = len_dist()
    lengths = 2:max_length
    probs = [pdf(d, L - 2) for L in lengths]
    return lengths, probs
end

csv_escape(s::AbstractString) = "\"" * replace(s, "\"" => "\"\"") * "\""
join_values(v) = join(v, ';')

function write_samples_csv(path, rows)
    open(path, "w") do io
        println(io, "sample_id,length,curve_mse,alternating_violations,dummy_tokens,tokens,x_coords,y_coords")
        for row in rows
            tokens = csv_escape(join_values(row.tokens))
            xs_s = csv_escape(join_values(row.xcoords))
            ys_s = csv_escape(join_values(row.ycoords))
            println(io, "$(row.sample_id),$(row.length),$(row.curve_mse),$(row.alternating_violations),$(row.dummy_tokens),$tokens,$xs_s,$ys_s")
        end
    end
end

function write_target_length_csv(path, max_length::Int)
    lengths, probs = analytic_length_pmf(max_length)
    open(path, "w") do io
        println(io, "length,probability")
        for (L, p) in zip(lengths, probs)
            println(io, "$L,$p")
        end
    end
end

function write_summary_csv(path; kwargs...)
    keys_vec = collect(keys(kwargs))
    open(path, "w") do io
        println(io, join(string.(keys_vec), ","))
        println(io, join(string.(getindex.(Ref(kwargs), keys_vec)), ","))
    end
end

function model_wrap(model, t, Xt)
    X1hat, hat_insertions = model(dev([t]), dev(Xt))
    return (cpu(ContinuousState(X1hat[1])), cpu(X1hat[2])), cpu(hat_insertions)
end

const experiment_name = env_or("DIRECTIONAL_FLOWCEPTION_BENCHMARK_NAME", "DIRECTIONAL_FLOWCEPTION_BENCHMARK_LABEL", "directional_flowception_benchmark")
const out_root = joinpath(@__DIR__, "..", "runs")
mkpath(out_root)
const run_dir = joinpath(out_root, experiment_name * "_" * string(Dates.Date(Dates.now())) * "_" * string(rand(100000:999999)))
mkpath(run_dir)

const seed = parse(Int, env_or("DIRECTIONAL_FLOWCEPTION_BENCHMARK_SEED", "DIRECTIONAL_FLOWCEPTION_DEMO_SEED", "1234"))
Random.seed!(seed)
use_gpu && CUDA.seed!(seed)

const total_time = parse(Float32, env_or("DIRECTIONAL_FLOWCEPTION_BENCHMARK_TOTAL_TIME", "DIRECTIONAL_FLOWCEPTION_DEMO_TOTAL_TIME", "10"))
const reveal_temperature = parse(Float32, env_or("DIRECTIONAL_FLOWCEPTION_BENCHMARK_REVEAL_TEMPERATURE", "DIRECTIONAL_FLOWCEPTION_DEMO_REVEAL_TEMPERATURE", "0.75"))
const reveal_kind = env_or("DIRECTIONAL_FLOWCEPTION_BENCHMARK_REVEAL_ORDER", "DIRECTIONAL_FLOWCEPTION_DEMO_REVEAL_ORDER", "independent")
const target_mode = env_or("DIRECTIONAL_FLOWCEPTION_BENCHMARK_TARGET_MODE", "DIRECTIONAL_FLOWCEPTION_DEMO_TARGET_MODE", "rb")
const nstart = parse(Int, env_or("DIRECTIONAL_FLOWCEPTION_BENCHMARK_NSTART", "DIRECTIONAL_FLOWCEPTION_DEMO_NSTART", "1"))
const model_dim = parse(Int, env_or("DIRECTIONAL_FLOWCEPTION_BENCHMARK_DIM", "DIRECTIONAL_FLOWCEPTION_DEMO_DIM", "256"))
const model_depth = parse(Int, env_or("DIRECTIONAL_FLOWCEPTION_BENCHMARK_DEPTH", "DIRECTIONAL_FLOWCEPTION_DEMO_DEPTH", "8"))
const iters = parse(Int, env_or("DIRECTIONAL_FLOWCEPTION_BENCHMARK_ITERS", "DIRECTIONAL_FLOWCEPTION_DEMO_ITERS", "10000"))
const batchsize = parse(Int, env_or("DIRECTIONAL_FLOWCEPTION_BENCHMARK_BATCH", "DIRECTIONAL_FLOWCEPTION_DEMO_BATCH", "64"))
const warmdown_steps = parse(Int, env_or("DIRECTIONAL_FLOWCEPTION_BENCHMARK_WARMDOWN_STEPS", "DIRECTIONAL_FLOWCEPTION_DEMO_WARMDOWN_STEPS", "3000"))
const lr = parse(Float64, env_or("DIRECTIONAL_FLOWCEPTION_BENCHMARK_LR", "DIRECTIONAL_FLOWCEPTION_DEMO_LR", "0.005"))
const nsamples = parse(Int, env_or("DIRECTIONAL_FLOWCEPTION_BENCHMARK_NSAMPLES", "DIRECTIONAL_FLOWCEPTION_DEMO_NSAMPLES", "1000"))
const sample_steps = parse(Int, env_or("DIRECTIONAL_FLOWCEPTION_BENCHMARK_SAMPLE_STEPS", "DIRECTIONAL_FLOWCEPTION_DEMO_SAMPLE_STEPS", "1000"))
const warmdown_start = max(iters - warmdown_steps, 0)

println("run_dir=$run_dir")
println("settings seed=$seed reveal_order=$reveal_kind target_mode=$target_mode reveal_temperature=$reveal_temperature total_time=$total_time nstart=$nstart iters=$iters warmdown_steps=$warmdown_steps batch=$batchsize lr=$lr nsamples=$nsamples sample_steps=$sample_steps")

P = DirectionalFlowceptionFlow(
    (BrownianMotion(0.05f0), Flowfusion.DistNoisyInterpolatingDiscreteFlow()),
    birth_sampler,
    insertion_transform = x -> exp.(clamp.(x, -20, 6)),
    total_time = total_time,
    reveal_order = parse_reveal_order(reveal_kind, reveal_temperature),
)

model = ToyDirectionalFlowception(model_dim, model_depth) |> dev
for layer in model.layers.transformers
    layer.attention.wo.weight ./= 10
    layer.feed_forward.w2.weight ./= 10
end
model.layers.insertion_decoder.weight ./= 10
model.layers.insertion_decoder.bias .= -4f0

orig_eta = lr
eta_ref = Ref(Float64(lr))
opt_state = Flux.setup(Muon(eta = orig_eta), model)

open(joinpath(run_dir, "config.txt"), "w") do io
    println(io, "seed=$seed")
    println(io, "reveal_order=$reveal_kind")
    println(io, "target_mode=$target_mode")
    println(io, "reveal_temperature=$reveal_temperature")
    println(io, "total_time=$total_time")
    println(io, "nstart=$nstart")
    println(io, "iters=$iters")
    println(io, "warmdown_steps=$warmdown_steps")
    println(io, "batchsize=$batchsize")
    println(io, "lr=$lr")
    println(io, "nsamples=$nsamples")
    println(io, "sample_steps=$sample_steps")
end

open(joinpath(run_dir, "train_log.csv"), "w") do io
    println(io, "iter,loc_loss,disc_loss,insertion_loss,total_loss")
end

for i in 1:iters
    ts = directional_flowception_bridge(P, [X1target() for _ in 1:batchsize], Uniform(0f0, P.total_time), nstart = nstart) |> dev
    loss, (∇model,) = Flux.withgradient(model) do m
        X1hat, hat_insertions = m(ts.t, ts.Xt)
        global_t = reshape(ts.t, 1, :)
        loc_loss = floss(P.P[1], X1hat[1], ts.X1anchor[1], scalefloss(P.P[1], ts.Xt.local_t, 1, 0.2f0)) * 10
        disc_loss = floss(P.P[2], X1hat[2], onehot(ts.X1anchor[2]), scalefloss(P.P[2], ts.Xt.local_t, 1, 0.2f0)) / 3
        insertion_loss = floss(P, hat_insertions, ts.insertions_target, ts.Xt, scalefloss(P, global_t, 1, 0.2f0))
        total_loss = loc_loss + disc_loss + insertion_loss
        if i == 1 || i % 50 == 0
            println("iter=$i loc_loss=$loc_loss disc_loss=$disc_loss insertion_loss=$insertion_loss total_loss=$total_loss")
        end
        return (; val = total_loss, loc_loss, disc_loss, insertion_loss)
    end
    Flux.update!(opt_state, model, ∇model)
    if i > warmdown_start
        eta_ref[] = max(eta_ref[] - orig_eta / max(warmdown_steps, 1), 1e-10)
        Flux.adjust!(opt_state, eta_ref[])
    end
    open(joinpath(run_dir, "train_log.csv"), "a") do io
        println(io, "$(i),$(loss.loc_loss),$(loss.disc_loss),$(loss.insertion_loss),$(loss.val)")
    end
end

serialize(joinpath(run_dir, "model.jls"), Flux.state(cpu(model)))

sample_rows = NamedTuple[]
sample_curve_mse = Float32[]
sample_alt_violations = Int[]
sample_dummy_counts = Int[]
sample_lengths = Int[]
steps = collect(range(0f0, P.total_time, length = sample_steps + 1))

for sample_ix in 1:nsamples
    X0 = make_source(P; nstart)
    samp = gen(P, X0, (t, Xt) -> model_wrap(model, t, Xt), steps)
    pts = final_points(samp)
    toks = final_tokens(samp)
    mse = curve_mse(samp)
    alt_viol = alternating_violations(toks)
    dummy_count = count(==(3), toks)
    push!(sample_curve_mse, mse)
    push!(sample_alt_violations, alt_viol)
    push!(sample_dummy_counts, dummy_count)
    push!(sample_lengths, length(toks))
    push!(sample_rows, (
        sample_id = sample_ix,
        length = length(toks),
        curve_mse = mse,
        alternating_violations = alt_viol,
        dummy_tokens = dummy_count,
        tokens = collect(toks),
        xcoords = vec(pts[1, :]),
        ycoords = vec(pts[2, :]),
    ))
    if sample_ix == 1 || sample_ix % 100 == 0
        println("sample=$sample_ix length=$(length(toks)) curve_mse=$mse alternating_violations=$alt_viol dummy_tokens=$dummy_count")
    end
end

write_samples_csv(joinpath(run_dir, "samples.csv"), sample_rows)
write_target_length_csv(joinpath(run_dir, "target_length_pmf.csv"), max(maximum(sample_lengths), 64))
write_summary_csv(joinpath(run_dir, "summary.csv");
    mean_curve_mse = mean(sample_curve_mse),
    mean_alternating_violations = mean(sample_alt_violations),
    mean_dummy_tokens = mean(sample_dummy_counts),
    mean_length = mean(sample_lengths),
    std_length = std(Float32.(sample_lengths)),
    min_length = minimum(sample_lengths),
    max_length = maximum(sample_lengths),
    nsamples = nsamples,
)
