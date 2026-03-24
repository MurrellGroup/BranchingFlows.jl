using Dates
using Statistics

const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
const RUNS_ROOT = joinpath(PROJECT_ROOT, "runs")

function latest_run_dir(prefix::AbstractString)
    isdir(RUNS_ROOT) || error("Runs directory not found: $RUNS_ROOT")
    matches = filter(name -> startswith(name, prefix * "_"), readdir(RUNS_ROOT))
    isempty(matches) && error("No run directories found for prefix `$prefix` in $RUNS_ROOT")
    mtimes = map(name -> stat(joinpath(RUNS_ROOT, name)).mtime, matches)
    return joinpath(RUNS_ROOT, matches[argmax(mtimes)])
end

function parse_scalar_csv(path::AbstractString)
    lines = readlines(path)
    length(lines) >= 2 || error("Expected header and value rows in $path")
    header = split(strip(lines[1]), ',')
    values = split(strip(lines[2]), ',')
    length(header) == length(values) || error("Mismatched header/value columns in $path")
    return Dict(header[i] => values[i] for i in eachindex(header))
end

function parse_samples_csv(path::AbstractString)
    lengths = Int[]
    curve_mse = Float64[]
    alternating_violations = Float64[]
    dummy_tokens = Float64[]
    open(path, "r") do io
        readline(io)
        for line in eachline(io)
            fields = split(line, ','; limit = 6)
            length(fields) >= 5 || error("Malformed samples row in $path: $line")
            push!(lengths, parse(Int, fields[2]))
            push!(curve_mse, parse(Float64, fields[3]))
            push!(alternating_violations, parse(Float64, fields[4]))
            push!(dummy_tokens, parse(Float64, fields[5]))
        end
    end
    return (; lengths, curve_mse, alternating_violations, dummy_tokens)
end

function parse_target_length_pmf(path::AbstractString)
    pmf = Dict{Int, Float64}()
    open(path, "r") do io
        readline(io)
        for line in eachline(io)
            isempty(strip(line)) && continue
            L, p = split(strip(line), ',')
            pmf[parse(Int, L)] = parse(Float64, p)
        end
    end
    return pmf
end

function empirical_length_pmf(lengths::AbstractVector{<:Integer})
    pmf = Dict{Int, Float64}()
    invn = 1 / length(lengths)
    for L in lengths
        pmf[L] = get(pmf, L, 0.0) + invn
    end
    return pmf
end

function total_variation(p::Dict{Int, Float64}, q::Dict{Int, Float64})
    support = union(keys(p), keys(q))
    return 0.5 * sum(abs(get(p, k, 0.0) - get(q, k, 0.0)) for k in support)
end

function pmf_mean(pmf::Dict{Int, Float64})
    return sum(k * v for (k, v) in pmf)
end

function pmf_std(pmf::Dict{Int, Float64})
    μ = pmf_mean(pmf)
    return sqrt(sum((k - μ)^2 * v for (k, v) in pmf))
end

function rank_map(rows, metric::Symbol)
    order = sortperm(rows; by = row -> getfield(row, metric))
    ranks = Dict{String, Int}()
    for (rank, idx) in enumerate(order)
        ranks[rows[idx].label] = rank
    end
    return ranks
end

function build_row(label::String, prefix::String)
    run_dir = latest_run_dir(prefix)
    summary = parse_scalar_csv(joinpath(run_dir, "summary.csv"))
    samples = parse_samples_csv(joinpath(run_dir, "samples.csv"))
    target_pmf = parse_target_length_pmf(joinpath(run_dir, "target_length_pmf.csv"))
    empirical_pmf = empirical_length_pmf(samples.lengths)
    return (
        label = label,
        prefix = prefix,
        run_dir = run_dir,
        nsamples = length(samples.lengths),
        mean_curve_mse = mean(samples.curve_mse),
        mean_alternating_violations = mean(samples.alternating_violations),
        mean_dummy_tokens = mean(samples.dummy_tokens),
        mean_length = mean(samples.lengths),
        std_length = std(Float64.(samples.lengths)),
        min_length = minimum(samples.lengths),
        max_length = maximum(samples.lengths),
        target_mean_length = pmf_mean(target_pmf),
        target_std_length = pmf_std(target_pmf),
        length_tv_distance = total_variation(empirical_pmf, target_pmf),
        length_mean_abs_error = abs(mean(samples.lengths) - pmf_mean(target_pmf)),
        length_std_abs_error = abs(std(Float64.(samples.lengths)) - pmf_std(target_pmf)),
        summary_mean_curve_mse = parse(Float64, summary["mean_curve_mse"]),
        summary_mean_alternating_violations = parse(Float64, summary["mean_alternating_violations"]),
        summary_mean_dummy_tokens = parse(Float64, summary["mean_dummy_tokens"]),
    )
end

function choose_repaired_mode(rows)
    repaired = filter(row -> row.label in ("seeded_sparse", "seeded_rb"), rows)
    metrics = (:length_tv_distance, :mean_curve_mse, :mean_alternating_violations, :mean_dummy_tokens)
    rank_maps = Dict(metric => rank_map(rows, metric) for metric in metrics)
    scored = map(repaired) do row
        total_rank = sum(rank_maps[metric][row.label] for metric in metrics)
        improved_metrics = count(metric -> getfield(row, metric) < getfield(only(filter(r -> r.label == "seeded_count", rows)), metric), metrics)
        return merge(row, (total_rank = total_rank, improved_metrics = improved_metrics))
    end
    sort!(scored; by = row -> (row.total_rank, row.length_tv_distance, row.mean_curve_mse, row.mean_alternating_violations))
    return first(scored)
end

function write_comparison_csv(path, rows)
    header = [
        "label",
        "run_dir",
        "nsamples",
        "length_tv_distance",
        "length_mean_abs_error",
        "length_std_abs_error",
        "mean_curve_mse",
        "mean_alternating_violations",
        "mean_dummy_tokens",
        "mean_length",
        "std_length",
        "min_length",
        "max_length",
    ]
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join((
                row.label,
                row.run_dir,
                row.nsamples,
                row.length_tv_distance,
                row.length_mean_abs_error,
                row.length_std_abs_error,
                row.mean_curve_mse,
                row.mean_alternating_violations,
                row.mean_dummy_tokens,
                row.mean_length,
                row.std_length,
                row.min_length,
                row.max_length,
            ), ","))
        end
    end
end

function write_decision(path, best_repaired, should_resume::Bool)
    selected_target_mode = best_repaired.label == "seeded_sparse" ? "sparse" : "rb"
    open(path, "w") do io
        println(io, "selected_target_mode=$selected_target_mode")
        println(io, "selected_label=$(best_repaired.label)")
        println(io, "selected_run_dir=$(best_repaired.run_dir)")
        println(io, "selected_length_tv_distance=$(best_repaired.length_tv_distance)")
        println(io, "selected_mean_curve_mse=$(best_repaired.mean_curve_mse)")
        println(io, "selected_mean_alternating_violations=$(best_repaired.mean_alternating_violations)")
        println(io, "selected_mean_dummy_tokens=$(best_repaired.mean_dummy_tokens)")
        println(io, "selected_total_rank=$(best_repaired.total_rank)")
        println(io, "selected_improved_metrics=$(best_repaired.improved_metrics)")
        println(io, "should_resume=$(should_resume)")
    end
end

rows = [
    build_row("seeded_count", "directional_flowception_seeded_count_target"),
    build_row("independent_count", "directional_flowception_independent_count_target"),
    build_row("seeded_sparse", "directional_flowception_seeded_sparse_target"),
    build_row("seeded_rb", "directional_flowception_seeded_rb_target"),
]

metrics = (:length_tv_distance, :mean_curve_mse, :mean_alternating_violations, :mean_dummy_tokens)
rank_maps = Dict(metric => rank_map(rows, metric) for metric in metrics)
rows = map(rows) do row
    merge(row, (total_rank = sum(rank_maps[metric][row.label] for metric in metrics),))
end
sort!(rows; by = row -> (row.total_rank, row.length_tv_distance, row.mean_curve_mse, row.mean_alternating_violations))

timestamp = string(Dates.Date(Dates.now())) * "_" * string(rand(100000:999999))
out_dir = joinpath(RUNS_ROOT, "directional_flowception_benchmark_comparison_" * timestamp)
mkpath(out_dir)
write_comparison_csv(joinpath(out_dir, "comparison.csv"), rows)

best_repaired = choose_repaired_mode(rows)
seeded_count = only(filter(row -> row.label == "seeded_count", rows))
should_resume =
    best_repaired.improved_metrics >= 3 &&
    best_repaired.length_tv_distance < seeded_count.length_tv_distance &&
    best_repaired.mean_curve_mse < seeded_count.mean_curve_mse
write_decision(joinpath(out_dir, "decision.txt"), best_repaired, should_resume)

println("comparison_dir=$out_dir")
for row in rows
    println(
        "label=$(row.label) total_rank=$(row.total_rank) length_tv_distance=$(row.length_tv_distance) " *
        "mean_curve_mse=$(row.mean_curve_mse) mean_alternating_violations=$(row.mean_alternating_violations) " *
        "mean_dummy_tokens=$(row.mean_dummy_tokens)"
    )
end
selected_target_mode = best_repaired.label == "seeded_sparse" ? "sparse" : "rb"
println(
    "selected_label=$(best_repaired.label) selected_target_mode=$selected_target_mode " *
    "should_resume=$(should_resume)"
)
