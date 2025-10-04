"""
Abstract supertype for all coalescence selection policies.

Implementors should provide:
- `select_coalescence(policy, nodes; time=nothing) => Union{Nothing,Tuple{Int,Int}}`
- `max_coalescences(policy, nodes) => Int`

Optional stateful hooks (no-ops by default):
- `init!(policy, nodes)`
- `update!(policy, nodes, i, j, new_index)`
"""
abstract type CoalescencePolicy end

"""
Coalescence policies merge sequentially adjacent elements, and thus depend on the sequential ordering of `nodes`.

Models using `SequentialCoalescencePolicy` may exploit sequence-order features
(e.g., positional encodings). Caller must ensure ordering is meaningful.
If the model does not have sequence-order features, then a `SequentialCoalescencePolicy` may still make sense
if sequential order happens to relate to some other proximity property, otherwise it is just a weird kind of random coalescence.
"""
abstract type SequentialCoalescencePolicy <: CoalescencePolicy end

"""
Coalescence policies that merge elements that may not be sequentially adjacent.

Models using `NonSequentialCoalescencePolicy` must not rely on sequence order
signals (e.g., avoid RoPE/positional encodings in the conditioning).
"""
abstract type NonSequentialCoalescencePolicy <: CoalescencePolicy end

# Optional hooks for stateful policies
init!(policy, nodes) = nothing
update!(policy, nodes, i, j, new_index) = nothing

"""
Compute an upper bound on the number of coalescences possible under free merging
within groups. Used as a default for non-sequential policies.
"""
function groupwise_max_coalescences(nodes)
    counts = Dict{Int,Int}()
    for n in nodes
        if n.free
            counts[n.group] = get(counts, n.group, 0) + 1
        end
    end
    s = 0
    for c in values(counts)
        s += max(0, c - 1)
    end
    return s
end

"""
    select_coalescence(policy::CoalescencePolicy, nodes; time=nothing)

Return either `(i, j)` (indices into `nodes` to coalesce, with `i<j` recommended)
or `nothing` if no valid coalescence is available at this time. The `time` keyword
is provided so policies can condition on the current event time if desired.
"""

"""
    max_coalescences(policy::CoalescencePolicy, nodes)

Return an integer upper bound on how many coalescence events are possible from
the current `nodes` configuration. Used to sample the event count.
"""

# Sequential neighbor-uniform policy (default)
struct SequentialUniform <: SequentialCoalescencePolicy end

"""
    select_coalescence(::SequentialUniform, nodes; time=nothing)

Uniformly choose one sequentially-adjacent free pair `(i, i+1)` within the same
group. Returns `nothing` if no such pair exists.
"""
function select_coalescence(::SequentialUniform, nodes; time=nothing)
    idx = Int[]
    for i in 1:length(nodes)-1
        if nodes[i].free && nodes[i+1].free && nodes[i].group == nodes[i+1].group
            push!(idx, i)
        end
    end
    isempty(idx) && return nothing
    i = rand(idx)
    return (i, i+1)
end

"""
    max_coalescences(::SequentialUniform, nodes)

Count the number of eligible sequentially-adjacent pairs.
"""
function max_coalescences(::SequentialUniform, nodes)
    c = 0
    for i in 1:length(nodes)-1
        c += (nodes[i].free && nodes[i+1].free && nodes[i].group == nodes[i+1].group)
    end
    return c
end

"""
    sequential_pairs(nodes)

Iterator over all sequentially-adjacent eligible pairs `(i, i+1)` within groups.
"""
sequential_pairs(nodes) = ((i, i+1) for i in 1:length(nodes)-1
    if nodes[i].free && nodes[i+1].free && nodes[i].group == nodes[i+1].group)

"""
    all_intragroup_pairs(nodes)

Iterator over all eligible intragroup pairs `(i, j)` with `i<j`.
"""
all_intragroup_pairs(nodes) = Iterators.flatten(
    (( (i, j) for j in (i+1):length(nodes)
        if nodes[i].free && nodes[j].free && nodes[i].group == nodes[j].group )
        for i in 1:length(nodes))
)

"""
    WeightedPairs(weight, pairs)

A non-sequential coalescence policy that samples a pair with probability
proportional to a user-supplied nonnegative `weight(nodes, i, j)`. Candidate
pairs are produced by `pairs(nodes)` (e.g., `all_intragroup_pairs`, kNN, graph
edges). Sampling is done in one pass via Efraimidis–Spirakis, so no dense
probability vectors are materialized.

Notes:
- As a `NonSequentialCoalescencePolicy`, models using this policy should not
  rely on sequence order signals.
- Ensure `pairs(nodes)` only yields valid free intragroup pairs.
"""
struct WeightedPairs{F,G} <: NonSequentialCoalescencePolicy
    weight::F      # (nodes, i, j) -> nonnegative weight
    pairs::G       # nodes -> iterator of (i, j)
end

function _weighted_choice(pairs_iter, weight, nodes)
    best = nothing
    best_key = -Inf
    for (i, j) in pairs_iter
        w = weight(nodes, i, j)
        if w > 0
            k = rand()^(1 / w)
            if k > best_key
                best = (i, j)
                best_key = k
            end
        end
    end
    return best
end

"""
    select_coalescence(P::WeightedPairs, nodes; time=nothing)
"""
function select_coalescence(P::WeightedPairs, nodes; time=nothing)
    return _weighted_choice(P.pairs(nodes), P.weight, nodes)
end

"""
    max_coalescences(::WeightedPairs, nodes)

Defaults to a groupwise upper bound.
"""
max_coalescences(::WeightedPairs, nodes) = groupwise_max_coalescences(nodes)


# ------------------------------------------------------------
# Standard/coalescence policy constructors and utilities
# ------------------------------------------------------------

"""
    distance_weighted_coalescence(; state_index=1, temperature=1.0, squared=false, pairs=all_intragroup_pairs)

Convenience constructor for a non-sequential distance-weighted policy. For each
candidate pair `(i,j)`, compute Euclidean distance `d` between the
`state_index` component of their `node_data` (flattened). Sample pairs with
weights proportional to `exp(-d / temperature)` if `squared=false`, or
`exp(-d^2 / (2*temperature^2))` if `squared=true`.

Fallback: if all weights underflow to zero for the current candidates, the
policy deterministically chooses the pair with the smallest distance.

As `temperature → 0+`, behavior approaches nearest-first.

Notes:
- Choose `state_index` to reference a continuous Euclidean component.
- `pairs` controls candidate enumeration and is called each event.
"""
function distance_weighted_coalescence(; state_index::Int=1, temperature::Real=1.0, squared::Bool=false, pairs=all_intragroup_pairs)
    return DistanceWeighted(Int(state_index), float(temperature), Bool(squared), pairs)
end

"""
    DistanceWeighted(state_index, temperature, squared, pairs)

Concrete non-sequential policy used by `distance_weighted_coalescence`.
"""
struct DistanceWeighted{G} <: NonSequentialCoalescencePolicy
    state_index::Int
    temperature::Float64
    squared::Bool
    pairs::G
end

max_coalescences(::DistanceWeighted, nodes) = groupwise_max_coalescences(nodes)

function select_coalescence(P::DistanceWeighted, nodes; time=nothing)
    # Extractor to Float64 vector to reduce underflow
    _vec_from_node(idx) = begin
        data = nodes[idx].node_data
        s = data isa Tuple ? data[P.state_index] : data
        vec(Float64.(tensor(s)))
    end
    # Enumerate candidates, accumulate weights and track argmin distance
    totalw = 0.0
    best_pair = nothing
    best_d = Inf
    candidates = Tuple{Int,Int,Float64,Float64}[]  # (i,j,d,weight)
    for (i, j) in P.pairs(nodes)
        xi = _vec_from_node(i); xj = _vec_from_node(j)
        if P.squared
            d2 = sum((xi .- xj) .^ 2)
            w = exp(-d2 / (2 * (P.temperature^2)))
            d = sqrt(d2)
            push!(candidates, (i, j, d, w))
            totalw += w
            if d < best_d
                best_d = d; best_pair = (i, j)
            end
        else
            d = sqrt(sum((xi .- xj) .^ 2))
            w = exp(-d / P.temperature)
            push!(candidates, (i, j, d, w))
            totalw += w
            if d < best_d
                best_d = d; best_pair = (i, j)
            end
        end
    end
    isempty(candidates) && return nothing
    if totalw <= 0.0 || !isfinite(totalw)
        return best_pair
    end
    # Roulette-wheel selection using accumulated weights
    u = rand() * totalw
    acc = 0.0
    for (i, j, d, w) in candidates
        acc += w
        if acc >= u
            return (i, j)
        end
    end
    # Fallback to nearest if numerical noise prevented selection
    return best_pair
end

"""
    BalancedSequential(; alpha=1.0)

Sequential policy that prefers coalescing smaller adjacent clusters first,
encouraging balanced trees while respecting sequence adjacency. For an eligible
adjacent pair `(i,i+1)`, the sampling weight is `(w_i + w_{i+1})^(-alpha)`
where `w_k = nodes[k].weight`. Set `alpha=0` to recover uniform sequential
selection; larger `alpha` increases preference for smaller clusters.
"""
struct BalancedSequential <: SequentialCoalescencePolicy
    alpha::Float64
end

BalancedSequential(; alpha::Real=1.0) = BalancedSequential(float(alpha))

function select_coalescence(p::BalancedSequential, nodes; time=nothing)
    α = p.alpha
    α < 0 && throw(ArgumentError("alpha must be >= 0"))
    weight = (nodes, i, j) -> (nodes[i].weight + nodes[j].weight)^(-α)
    return _weighted_choice(sequential_pairs(nodes), weight, nodes)
end

function max_coalescences(::BalancedSequential, nodes)
    return max_coalescences(SequentialUniform(), nodes)
end

"""
    CorrelatedSequential(; boost=5.0, radius=1)

Sequential policy that increases the chance of coalescing near the most recent
coalescence location. Pairs `(i,i+1)` within `radius` of the last coalescence
index get their weight multiplied by `boost` (>= 1). Base weights are uniform.
"""
mutable struct CorrelatedSequential <: SequentialCoalescencePolicy
    boost::Float64
    radius::Int
    last::Union{Nothing,Int}
end

CorrelatedSequential(; boost::Real=5.0, radius::Integer=1) = CorrelatedSequential(float(boost), Int(radius), nothing)

function init!(p::CorrelatedSequential, nodes)
    p.last = nothing
    return nothing
end

function select_coalescence(p::CorrelatedSequential, nodes; time=nothing)
    base_pairs = sequential_pairs(nodes)
    if p.last === nothing
        # Uniform over sequential pairs on first event
        idx = Int[]
        for (i, j) in base_pairs
            push!(idx, i)
        end
        isempty(idx) && return nothing
        i = rand(idx)
        return (i, i+1)
    end
    last = p.last
    w = (nodes, i, j) -> (abs(i - last) <= p.radius ? p.boost : 1.0)
    return _weighted_choice(sequential_pairs(nodes), w, nodes)
end

function update!(p::CorrelatedSequential, nodes, i, j, new_index)
    p.last = new_index
    return nothing
end





