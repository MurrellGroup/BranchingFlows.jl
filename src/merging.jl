#Note: Many of these are untested. The only one used at scaled is the default: SequentialUniform

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

# Optional hooks for stateful policies
init!(policy, nodes) = nothing
update!(policy, nodes, i, j, new_index) = nothing

# Optional hook to reorder the forest (roots and/or child order) after merges
# Default: no-op
reorder_forest(policy, nodes) = nodes

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

function _block_sizes(nodes)
    sizes = Int[]
    current = 0
    for i in eachindex(nodes)
        if nodes[i].branchable && (i == 1 || nodes[i - 1].branchable && nodes[i - 1].group == nodes[i].group)
            current += 1
        elseif nodes[i].branchable
            current = 1
        else
            current = 0
        end
        push!(sizes, current)
    end
    reverse_current = 0
    reverse_sizes = similar(sizes)
    for i in reverse(eachindex(nodes))
        if nodes[i].branchable && (i == lastindex(nodes) || nodes[i + 1].branchable && nodes[i + 1].group == nodes[i].group)
            reverse_current += 1
        elseif nodes[i].branchable
            reverse_current = 1
        else
            reverse_current = 0
        end
        reverse_sizes[i] = reverse_current
    end
    branchable = [node.branchable ? 1 : 0 for node in nodes]
    return sizes .+ reverse_sizes .- branchable
end

function _group_sizes(nodes)
    sizes = Dict{Int, Int}()
    for node in nodes
        if node.branchable
            sizes[node.group] = get(sizes, node.group, 0) + 1
        end
    end
    return sizes
end

function _eligible_sequential_pairs(nodes, group_mins)
    pairs = Tuple{Int, Int}[]
    group_sizes = _group_sizes(nodes)
    block_sizes = group_mins isa Int ? _block_sizes(nodes) : nothing
    for i in 1:(length(nodes) - 1)
        left = nodes[i]
        right = nodes[i + 1]
        if !(left.branchable && right.branchable && left.group == right.group)
            continue
        end
        allowed = if isnothing(group_mins)
            true
        elseif group_mins isa Dict{Int, Int}
            group_sizes[left.group] > group_mins[left.group]
        elseif group_mins isa Int
            block_sizes[i] > group_mins
        else
            error("Unsupported group_mins=$(typeof(group_mins)) for this sequential policy.")
        end
        allowed && push!(pairs, (i, i + 1))
    end
    return pairs
end

function _weighted_choice(pairs, weights)
    isempty(pairs) && return nothing
    total = sum(weights)
    total <= 0 && return pairs[rand(1:length(pairs))]
    cutoff = rand() * total
    running = 0.0
    for (pair, weight) in zip(pairs, weights)
        running += weight
        if running >= cutoff
            return pair
        end
    end
    return last(pairs)
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
    pairs = collect(sequential_pairs(nodes))
    weights = Float64[(nodes[i].weight + nodes[j].weight)^(-α) for (i, j) in pairs]
    return _weighted_choice(pairs, weights)
end

function max_coalescences(::BalancedSequential, nodes)
    return max_coalescences(SequentialUniform(), nodes)
end

"""
    RichGetRicherSequential(; alpha = 1.0)

Sequential policy that prefers coalescing larger adjacent clusters first. For
an eligible adjacent pair `(i, i+1)`, the sampling weight is
`(w_i + w_{i+1})^alpha` where `w_k = nodes[k].weight`. Set `alpha = 0` to
recover uniform sequential selection; larger `alpha` increases preference for
already-large clusters.
"""
struct RichGetRicherSequential <: SequentialCoalescencePolicy
    alpha::Float64
end

RichGetRicherSequential(; alpha::Real = 1.0) = RichGetRicherSequential(float(alpha))

function _select_weighted_pair(policy::RichGetRicherSequential, nodes, group_mins)
    pairs = _eligible_sequential_pairs(nodes, group_mins)
    isempty(pairs) && return nothing
    policy.alpha < 0 && throw(ArgumentError("alpha must be >= 0"))
    weights = Float64[(nodes[i].weight + nodes[j].weight)^policy.alpha for (i, j) in pairs]
    return _weighted_choice(pairs, weights)
end

select_coalescence(policy::RichGetRicherSequential, nodes, group_mins::Nothing) = _select_weighted_pair(policy, nodes, group_mins)
select_coalescence(policy::RichGetRicherSequential, nodes, group_mins) = _select_weighted_pair(policy, nodes, group_mins)
max_coalescences(::RichGetRicherSequential, nodes) = max_coalescences(SequentialUniform(), nodes)

euclidean_distance(x, y) = sqrt(sum(abs2, x .- y))

"""
    SequentialProximity(; extractor = identity, distance = euclidean_distance,
                          tie_atol = 0.0, tie_rtol = 0.0)

Sequential policy that chooses the eligible adjacent pair minimizing a
user-supplied distance between extracted per-node features. Ties within the
specified tolerances are broken uniformly at random.
"""
struct SequentialProximity{F, D} <: SequentialCoalescencePolicy
    extractor::F
    distance::D
    tie_atol::Float64
    tie_rtol::Float64
end

function SequentialProximity(;
    extractor = identity,
    distance = euclidean_distance,
    tie_atol::Real = 0.0,
    tie_rtol::Real = 0.0,
)
    return SequentialProximity(extractor, distance, float(tie_atol), float(tie_rtol))
end

function _closest_pairs(policy::SequentialProximity, nodes, group_mins)
    pairs = _eligible_sequential_pairs(nodes, group_mins)
    isempty(pairs) && return Tuple{Int, Int}[], Float64[]
    dists = Float64[]
    for (i, j) in pairs
        xi = policy.extractor(nodes[i].node_data)
        xj = policy.extractor(nodes[j].node_data)
        push!(dists, float(policy.distance(xi, xj)))
    end
    return pairs, dists
end

function _select_closest_pair(policy::SequentialProximity, nodes, group_mins)
    pairs, dists = _closest_pairs(policy, nodes, group_mins)
    isempty(pairs) && return nothing
    best = minimum(dists)
    tied = [pairs[i] for i in eachindex(pairs) if isapprox(dists[i], best; atol = policy.tie_atol, rtol = policy.tie_rtol)]
    return tied[rand(1:length(tied))]
end

select_coalescence(policy::SequentialProximity, nodes, group_mins::Nothing) = _select_closest_pair(policy, nodes, group_mins)
select_coalescence(policy::SequentialProximity, nodes, group_mins) = _select_closest_pair(policy, nodes, group_mins)
max_coalescences(::SequentialProximity, nodes) = max_coalescences(SequentialUniform(), nodes)

"""
    SequentialDeepLineage(; lambda = 1.0, min_count = 2,
                           trunk_target_sampler = () -> 1 + rand(Poisson(lambda)))

Stateful sequential policy that first allows a few trunk lineages to emerge,
then only allows merges that involve an already-deep lineage. For each group,
`init!` samples a target trunk count. Once the number of branchable nodes with
`weight >= min_count` reaches that target, the policy filters eligible adjacent
pairs down to those where at least one node already has `weight >= min_count`.
Among the remaining pairs, selection is uniform.
"""
mutable struct SequentialDeepLineage{F} <: SequentialCoalescencePolicy
    trunk_target_sampler::F
    min_count::Int
    target_trunks::Dict{Int, Int}
end

function SequentialDeepLineage(;
    lambda::Real = 1.0,
    min_count::Int = 2,
    trunk_target_sampler = () -> 1 + rand(Poisson(float(lambda))),
)
    min_count < 1 && throw(ArgumentError("min_count must be >= 1"))
    return SequentialDeepLineage(trunk_target_sampler, min_count, Dict{Int, Int}())
end

function init!(policy::SequentialDeepLineage, nodes)
    empty!(policy.target_trunks)
    for node in nodes
        if node.branchable && !haskey(policy.target_trunks, node.group)
            policy.target_trunks[node.group] = policy.trunk_target_sampler()
        end
    end
    return nothing
end

function _deep_lineage_pairs(policy::SequentialDeepLineage, nodes, group_mins)
    pairs = _eligible_sequential_pairs(nodes, group_mins)
    isempty(pairs) && return pairs
    deep_counts = Dict{Int, Int}()
    for node in nodes
        if node.branchable && node.weight >= policy.min_count
            deep_counts[node.group] = get(deep_counts, node.group, 0) + 1
        end
    end
    filtered = Tuple{Int, Int}[]
    for pair in pairs
        i, j = pair
        group = nodes[i].group
        deep_active = get(deep_counts, group, 0) >= get(policy.target_trunks, group, 1)
        if !deep_active || nodes[i].weight >= policy.min_count || nodes[j].weight >= policy.min_count
            push!(filtered, pair)
        end
    end
    return isempty(filtered) ? pairs : filtered
end

function _select_deep_lineage_pair(policy::SequentialDeepLineage, nodes, group_mins)
    pairs = _deep_lineage_pairs(policy, nodes, group_mins)
    isempty(pairs) && return nothing
    return pairs[rand(1:length(pairs))]
end

select_coalescence(policy::SequentialDeepLineage, nodes, group_mins::Nothing) = _select_deep_lineage_pair(policy, nodes, group_mins)
select_coalescence(policy::SequentialDeepLineage, nodes, group_mins) = _select_deep_lineage_pair(policy, nodes, group_mins)
max_coalescences(::SequentialDeepLineage, nodes) = max_coalescences(SequentialUniform(), nodes)
