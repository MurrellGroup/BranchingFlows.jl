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
    allows_cross_group_merge(policy::CoalescencePolicy) -> Bool

Trait indicating whether a coalescence policy may merge adjacent elements from
different groups (subject to its own constraints). Defaults to `false`.
"""
allows_cross_group_merge(::CoalescencePolicy) = false

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
    weight = (nodes, i, j) -> (nodes[i].weight + nodes[j].weight)^(-α)
    return _weighted_choice(sequential_pairs(nodes), weight, nodes)
end

function max_coalescences(::BalancedSequential, nodes)
    return max_coalescences(SequentialUniform(), nodes)
end

"""
    WithinThenBetween

Sequential policy that:
- merges sequentially adjacent elements within the same group (as usual);
- additionally allows merging across adjacent different groups, but only when
  both elements are singleton “islands” (neither has any same-group neighbor).
"""
struct WithinThenBetween <: SequentialCoalescencePolicy end

allows_cross_group_merge(::WithinThenBetween) = true

# Helper: whether nodes[i] has no adjacent same-group neighbor
function _is_island(nodes, i::Int)
    gi = nodes[i].group
    if i > 1 && nodes[i-1].branchable && nodes[i-1].group == gi
        return false
    end
    if i < length(nodes) && nodes[i+1].branchable && nodes[i+1].group == gi
        return false
    end
    return true
end

function _wtb_candidates(nodes)
    idx = Int[]
    for i in 1:length(nodes)-1
        a = nodes[i]; b = nodes[i+1]
        if !(a.branchable && b.branchable)
            continue
        end
        if a.group == b.group
            push!(idx, i)
        else
            if _is_island(nodes, i) && _is_island(nodes, i+1)
                push!(idx, i)
            end
        end
    end
    return idx
end

function select_coalescence(::WithinThenBetween, nodes, group_mins::Nothing)
    idx = _wtb_candidates(nodes)
    isempty(idx) && return nothing
    i = rand(idx)
    return (i, i+1)
end

function select_coalescence(::WithinThenBetween, nodes, group_mins::Dict{Int,Int})
    group_sizes = Dict{Int,Int}()
    for n in nodes
        if n.branchable
            group_sizes[n.group] = get(group_sizes, n.group, 0) + 1
        end
    end
    idx = Int[]
    for i in _wtb_candidates(nodes)
        ga = nodes[i].group
        gb = nodes[i+1].group
        if ga == gb
            if get(group_sizes, ga, 0) > get(group_mins, ga, 0)
                push!(idx, i)
            end
        else
            if get(group_sizes, ga, 0) > get(group_mins, ga, 0) &&
               get(group_sizes, gb, 0) > get(group_mins, gb, 0)
                push!(idx, i)
            end
        end
    end
    isempty(idx) && return nothing
    i = rand(idx)
    return (i, i+1)
end

# For Int minima, mirror SequentialUniform semantics for within-group blocks.
# Cross-group merges are allowed only if both sides are islands and group_mins ≤ 1.
function select_coalescence(::WithinThenBetween, nodes, group_mins::Int)
    idx = Int[]
    block_edges = Dict{Int,Int}()
    current_block = 1
    for i in 1:length(nodes)-1
        if nodes[i].branchable && nodes[i+1].branchable && nodes[i].group == nodes[i+1].group
            block_edges[current_block] = get(block_edges, current_block, 0) + 1
        else
            current_block += 1
        end
    end
    block_size = b -> get(block_edges, b, 0) + 1
    current_block = 1
    for i in 1:length(nodes)-1
        a = nodes[i]; b = nodes[i+1]
        if a.branchable && b.branchable && a.group == b.group
            if block_size(current_block) > group_mins
                push!(idx, i)
            end
        elseif a.branchable && b.branchable
            if _is_island(nodes, i) && _is_island(nodes, i+1) && group_mins <= 1
                push!(idx, i)
            end
            current_block += 1
        else
            current_block += 1
        end
    end
    isempty(idx) && return nothing
    i = rand(idx)
    return (i, i+1)
end

function max_coalescences(::WithinThenBetween, nodes)
    c = 0
    for i in 1:length(nodes)-1
        a = nodes[i]; b = nodes[i+1]
        if !(a.branchable && b.branchable)
            continue
        end
        if a.group == b.group
            c += 1
        else
            if _is_island(nodes, i) && _is_island(nodes, i+1)
                c += 1
            end
        end
    end
    return c
end

