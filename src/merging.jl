"""
Abstract supertype for all coalescence selection policies.

Implementors should provide:
- `select_coalescence(policy, nodes) => Union{Nothing,Tuple{Int,Int}}`
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

select_coalescence(coalescence_policy, nodes, group_mins) = error("Group mins not (yet) implemented for this policy.")
select_coalescence(coalescence_policy, nodes, group_mins::Nothing) = select_coalescence(coalescence_policy, nodes)


"""
    should_append_on_split(policy::CoalescencePolicy) -> Bool

Indicates whether forward-time split insertions should be appended to the end of
the sequence (`true`), or inserted adjacently at the split location (`false`).
Defaults to `false`.
"""
should_append_on_split(::CoalescencePolicy) = false

"""
Compute an upper bound on the number of coalescences possible under branchable merging
within groups. Used as a default for non-sequential policies.
"""
function groupwise_max_coalescences(nodes)
    counts = Dict{Int,Int}()
    for n in nodes
        if n.branchable
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
    select_coalescence(policy::CoalescencePolicy, nodes)

Return either `(i, j)` (indices into `nodes` to coalesce, with `i<j` recommended)
or `nothing` if no valid coalescence is available at this time.
"""

"""
    max_coalescences(policy::CoalescencePolicy, nodes)

Return an integer upper bound on how many coalescence events are possible from
the current `nodes` configuration. Used to sample the event count.
"""

# Sequential neighbor-uniform policy (default)
struct SequentialUniform <: SequentialCoalescencePolicy end

"""
    select_coalescence(::SequentialUniform, nodes)

Uniformly choose one sequentially-adjacent branchable pair `(i, i+1)` within the same
group. Returns `nothing` if no such pair exists.
"""
function select_coalescence(::SequentialUniform, nodes, group_mins::Nothing)
    idx = Int[]
    for i in 1:length(nodes)-1
        if nodes[i].branchable && nodes[i+1].branchable && nodes[i].group == nodes[i+1].group
            push!(idx, i)
        end
    end
    isempty(idx) && return nothing
    i = rand(idx)
    return (i, i+1)
end

function select_coalescence(::SequentialUniform, nodes, group_mins::Dict{Int,Int})
    idx = Int[]
    group_sizes = Dict{Int,Int}()
    for n in nodes
        if n.branchable
            if haskey(group_sizes, n.group)
                group_sizes[n.group] += 1
            else
                group_sizes[n.group] = 1
            end
        end
    end
    for i in 1:length(nodes)-1
        if nodes[i].branchable && nodes[i+1].branchable && nodes[i].group == nodes[i+1].group && group_sizes[nodes[i].group] > group_mins[nodes[i].group]
            push!(idx, i)
        end
    end
    isempty(idx) && return nothing
    i = rand(idx)
    return (i, i+1)
end


#Other options for group_mins are:
#Int, which will control the per-segment minimum (eg. if you want two tokens per segment).
#UnivariateDistribution, if you want a random minimum per segment.

#Untested:
function select_coalescence(::SequentialUniform, nodes, group_mins::Int)
    idx = Int[]
    block_group_sizes = Dict{Int,Int}()
    #This must do a first pass and track adjacent blocks of branchable tokens, and count how many are in each block.
    current_block = 1
    for i in 1:length(nodes)-1
        if nodes[i].branchable && nodes[i+1].branchable && nodes[i].group == nodes[i+1].group
            if haskey(block_group_sizes, current_block)
                block_group_sizes[current_block] += 1
            else
                block_group_sizes[current_block] = 1
            end
        else
            current_block += 1
        end
    end
    #...then do a second block and push to the list if they're above the minimum.
    current_block = 1
    for i in 1:length(nodes)-1
        if nodes[i].branchable && nodes[i+1].branchable && nodes[i].group == nodes[i+1].group
            if block_group_sizes[current_block] > (group_mins - 1)
                push!(idx, i)
            end
        else
            current_block += 1
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
        c += (nodes[i].branchable && nodes[i+1].branchable && nodes[i].group == nodes[i+1].group)
    end
    return c
end

"""
    sequential_pairs(nodes)

Iterator over all sequentially-adjacent eligible pairs `(i, i+1)` within groups.
"""
sequential_pairs(nodes) = ((i, i+1) for i in 1:length(nodes)-1
    if nodes[i].branchable && nodes[i+1].branchable && nodes[i].group == nodes[i+1].group)

"""
    all_intragroup_pairs(nodes)

Iterator over all eligible intragroup pairs `(i, j)` with `i<j`.
"""
all_intragroup_pairs(nodes) = Iterators.flatten(
    (( (i, j) for j in (i+1):length(nodes)
        if nodes[i].branchable && nodes[j].branchable && nodes[i].group == nodes[j].group )
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
- Ensure `pairs(nodes)` only yields valid branchable intragroup pairs.
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
    select_coalescence(P::WeightedPairs, nodes)
"""
function select_coalescence(P::WeightedPairs, nodes)
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

function select_coalescence(P::DistanceWeighted, nodes)
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

function select_coalescence(p::BalancedSequential, nodes)
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

CorrelatedSequential(; boost::Real=15.0, radius::Integer=3) = CorrelatedSequential(float(boost), Int(radius), nothing)

function init!(p::CorrelatedSequential, nodes)
    p.last = nothing
    return nothing
end

function select_coalescence(p::CorrelatedSequential, nodes)
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

"""
    max_coalescences(::CorrelatedSequential, nodes)

Count eligible sequentially-adjacent pairs (same as `SequentialUniform`).
"""
function max_coalescences(::CorrelatedSequential, nodes)
    return max_coalescences(SequentialUniform(), nodes)
end


#=

#=
Policies that don't split elements in-place (like LastToNearest) need modifications to the splits_target. See note below.
Importantly, they make appear to work on a demo whose spatial structure is aligned with the sequence order.
=#

"""
    last_to_nearest_coalescence(; state_index=1)

Sequential policy that always coalesces the last element in the sequence into
the nearest other element (same group, branchable) under Euclidean distance computed
from the `state_index` component of `node_data`.

Used in tandem with append-on-split insertion so that newly split elements are
placed at the end, matching the backward coalescence assumption.
"""
last_to_nearest_coalescence(; state_index::Int=1) = LastToNearest(Int(state_index))

"""
    LastToNearest(state_index)

Concrete policy for last-to-nearest coalescence.
"""
struct LastToNearest <: SequentialCoalescencePolicy
    state_index::Int
end

should_append_on_split(::LastToNearest) = true

function select_coalescence(P::LastToNearest, nodes)
    L = length(nodes)
    L < 2 && return nothing
    # Collect branchable indices by group
    groups = Dict{Int,Vector{Int}}()
    for idx in 1:L
        n = nodes[idx]
        if n.branchable
            g = n.group
            v = get!(groups, g, Int[])
            push!(v, idx)
        end
    end
    # Only consider groups with at least two branchable nodes
    cand = [(g, idxs) for (g, idxs) in groups if length(idxs) >= 2]
    isempty(cand) && return nothing
    # Sample a group proportional to its number of branchable elements
    tot = 0
    for (g, idxs) in cand; tot += length(idxs); end
    u = rand(1:tot)
    acc = 0
    chosen_g = nothing
    chosen_idxs = Int[]
    for (g, idxs) in cand
        acc += length(idxs)
        if u <= acc
            chosen_g = g
            chosen_idxs = idxs
            break
        end
    end
    # Take the last branchable index in the chosen group
    last_idx = maximum(chosen_idxs)
    # Find nearest other branchable in same group
    data_last = nodes[last_idx].node_data
    s_last = data_last isa Tuple ? data_last[P.state_index] : data_last
    x_last = vec(Float64.(tensor(s_last)))
    best_j = nothing
    best_d = Inf
    for j in chosen_idxs
        j == last_idx && continue
        data_j = nodes[j].node_data
        s_j = data_j isa Tuple ? data_j[P.state_index] : data_j
        x_j = vec(Float64.(tensor(s_j)))
        d2 = sum((x_last .- x_j) .^ 2)
        if d2 < best_d
            best_d = d2
            best_j = j
        end
    end
    best_j === nothing && return nothing
    i, j = min(best_j, last_idx), max(best_j, last_idx)
    return (i, j)
end

function max_coalescences(::LastToNearest, nodes)
    return max_coalescences(SequentialUniform(), nodes)
end

#=
Important note:
We have an issue with the relationship between the coalescence policy and the split rate process.
When coalescence is randomly distributed across sites, like the SequentialUniform policy, predicting the expected number of terminal descendents works.
There, the expected number of descendents gets reweighted by the coalescence hazard distribution to ensure matching.
But for anything besides a uniform distribution, like the LastToNearest policy (and possibly the neighbour policies),
knowing the expected number of terminal descendents is not sufficient to know the current outgoing rate.

One solution would be to just work with the outgoing rate directly, but that might be harder for the model to learn?

Is there a way to keep X1 prediction, but still allow non-uniform coalescence policies?

Ah ha! LastToNearest worked in the linear case because the tree that arose was ladder-like, and so the terminal count *was* sufficient in that case.
At the root, the count included all descendents. Then after the first split, the lower branch expected zero, and the upper branch expected N-1, etc.

Can we change how the bridge is set up to ensure that X1 prediction over endstates still works?
Think through QM9 with a ring. We have to first generate the descendant of each element in the ring in turn,
all the while the base of the ring is outputting zero. Then when the ring is done, the base output must spike again.

We can probably do this by tweaking the splits_target differently for each policy. How can we do this in general though?
There is some hidden assumption in the uniform case that connects the split rate to the coalescence sample times.
I guess one option would be for force the bridge to explicitly construct the coalescence rates and sample those directly,
which would force a connection to the splits_target?

Or we can work through the maths for each process and figure out what the correct splits target should be.
=#

=#