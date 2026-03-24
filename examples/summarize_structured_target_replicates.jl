using Dates
using Statistics

const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
const RUNS_ROOT = joinpath(PROJECT_ROOT, "runs")

function latest_run_dir(prefix::AbstractString)
    matches = filter(name -> startswith(name, prefix * "_"), readdir(RUNS_ROOT))
    isempty(matches) && error("No run directories found for prefix `$prefix`.")
    mtimes = map(name -> stat(joinpath(RUNS_ROOT, name)).mtime, matches)
    return joinpath(RUNS_ROOT, matches[argmax(mtimes)])
end

function exact_prefix_run_dir(prefix::AbstractString)
    candidates = filter(readdir(RUNS_ROOT)) do name
        startswith(name, prefix * "_") || return false
        suffix = name[length(prefix) + 2:end]
        return occursin(r"^\d{4}-\d{2}-\d{2}_\d+$", suffix)
    end
    isempty(candidates) && error("No exact run directories found for prefix `$prefix`.")
    mtimes = map(name -> stat(joinpath(RUNS_ROOT, name)).mtime, candidates)
    return joinpath(RUNS_ROOT, candidates[argmax(mtimes)])
end

function parse_scalar_csv(path::AbstractString)
    lines = readlines(path)
    length(lines) >= 2 || error("Expected header and value rows in $path")
    header = split(strip(lines[1]), ',')
    values = split(strip(lines[2]), ',')
    return Dict(header[i] => values[i] for i in eachindex(header))
end

function row_for(prefix::String, label::String; exact_prefix::Bool = false)
    run_dir = exact_prefix ? exact_prefix_run_dir(prefix) : latest_run_dir(prefix)
    summary = parse_scalar_csv(joinpath(run_dir, "summary.csv"))
    return (
        label = label,
        run_dir = run_dir,
        mean_curve_mse = parse(Float64, summary["mean_curve_mse"]),
        mean_alternating_violations = parse(Float64, summary["mean_alternating_violations"]),
        mean_dummy_tokens = parse(Float64, summary["mean_dummy_tokens"]),
        mean_length = parse(Float64, summary["mean_length"]),
        std_length = parse(Float64, summary["std_length"]),
    )
end

function aggregate_rows(label::String, rows)
    return (
        label = label,
        curve_mse_mean = mean(getfield.(rows, :mean_curve_mse)),
        curve_mse_std = std(getfield.(rows, :mean_curve_mse)),
        alt_mean = mean(getfield.(rows, :mean_alternating_violations)),
        alt_std = std(getfield.(rows, :mean_alternating_violations)),
        dummy_mean = mean(getfield.(rows, :mean_dummy_tokens)),
        dummy_std = std(getfield.(rows, :mean_dummy_tokens)),
        length_mean = mean(getfield.(rows, :mean_length)),
        length_std = std(getfield.(rows, :mean_length)),
        sample_length_std_mean = mean(getfield.(rows, :std_length)),
        sample_length_std_std = std(getfield.(rows, :std_length)),
    )
end

targets = Dict(
    "seeded_sparse" => [
        row_for("directional_flowception_seeded_sparse_target", "seeded_sparse_run1"; exact_prefix = true),
        row_for("directional_flowception_seeded_sparse_target_rerun2", "seeded_sparse_run2"),
    ],
    "seeded_rb" => [
        row_for("directional_flowception_seeded_rb_target", "seeded_rb_run1"; exact_prefix = true),
        row_for("directional_flowception_seeded_rb_target_rerun2", "seeded_rb_run2"),
    ],
)

timestamp = string(Dates.Date(Dates.now())) * "_" * string(rand(100000:999999))
out_dir = joinpath(RUNS_ROOT, "directional_flowception_structured_replicates_" * timestamp)
mkpath(out_dir)

open(joinpath(out_dir, "replicates.csv"), "w") do io
    println(io, "label,run_dir,mean_curve_mse,mean_alternating_violations,mean_dummy_tokens,mean_length,std_length")
    for rows in values(targets)
        for row in rows
            println(io, join((row.label, row.run_dir, row.mean_curve_mse, row.mean_alternating_violations, row.mean_dummy_tokens, row.mean_length, row.std_length), ","))
        end
    end
end

aggregates = [aggregate_rows(label, rows) for (label, rows) in sort(collect(targets); by = first)]
open(joinpath(out_dir, "aggregate.csv"), "w") do io
    println(io, "label,curve_mse_mean,curve_mse_std,alt_mean,alt_std,dummy_mean,dummy_std,length_mean,length_std,sample_length_std_mean,sample_length_std_std")
    for row in aggregates
        println(io, join((row.label, row.curve_mse_mean, row.curve_mse_std, row.alt_mean, row.alt_std, row.dummy_mean, row.dummy_std, row.length_mean, row.length_std, row.sample_length_std_mean, row.sample_length_std_std), ","))
    end
end

println("replicate_summary_dir=$out_dir")
for row in aggregates
    println("label=$(row.label) curve_mse_mean=$(row.curve_mse_mean) curve_mse_std=$(row.curve_mse_std) alt_mean=$(row.alt_mean) alt_std=$(row.alt_std) length_mean=$(row.length_mean) length_std=$(row.length_std)")
end
