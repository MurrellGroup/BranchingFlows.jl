struct Deletion end

"""
    CoalescentFlow{Proc,D,F,Pol,Delp,Dh} <: Process

Branching/coalescent flow wrapper that composes an underlying Flowfusion process
`P` with stochastic coalescence (backward) / splitting (forward) events, and
optional deletions governed by a time-hazard distribution.

Fields:
- `P::Proc`: underlying Flowfusion process (or tuple of processes) used for
  bridging between coalescence times.
- `branch_time_dist::D`: distribution on [0, 1] controlling split/coalescence
  event timing. Used both to sample absolute split times (for forests) and to
  scale small-step split Poisson rates via a truncated pdf at the current time.
- `split_transform::F`: elementwise map that converts model-predicted event
  logits into nonnegative intensities for split counts during forward-time
  simulation (default: `x -> exp.(clamp.(x, -100, 11))`).
- `coalescence_policy::Pol`: policy controlling which elements coalesce at each
  event; see policies in `merging.jl`.
- `deletion_policy::Delp`: tag for deletion behavior (placeholder hook).
- `deletion_time_dist::Dh`: distribution on [0, 1] specifying the deletion
  hazard via its pdf/cdf; used exactly over intervals during conditional
  path sampling and via a small-step hazard approximation during `step`.

Constructors:
- `CoalescentFlow(P, branch_time_dist)` uses the default `split_transform`,
  `SequentialUniform()` policy, and `Uniform(0, 1)` deletion-time distribution.
- `CoalescentFlow(P, branch_time_dist, policy)` as above with a custom policy.
- `CoalescentFlow(P, branch_time_dist, policy, deletion_time_dist)` as above
  with a custom deletion-time distribution.

Notes:
- Sequential policies assume sequence order is meaningful. Coalescences/splits
  are further constrained within groups (see `BranchingState.groupings`).
"""
struct CoalescentFlow{Proc,D,F,Pol,Delp,Dh} <: Process
    P::Proc
    branch_time_dist::D
    split_transform::F
    coalescence_policy::Pol
    deletion_policy::Delp
    deletion_time_dist::Dh
end
CoalescentFlow(P, branch_time_dist) = CoalescentFlow(P, branch_time_dist, x -> exp.(clamp.(x, -100, 11)), SequentialUniform(), Deletion(), Uniform(0, 1))
CoalescentFlow(P, branch_time_dist, policy) = CoalescentFlow(P, branch_time_dist, x -> exp.(clamp.(x, -100, 11)), policy, Deletion(), Uniform(0, 1))
CoalescentFlow(P, branch_time_dist, policy, deletion_time_dist) = CoalescentFlow(P, branch_time_dist, x -> exp.(clamp.(x, -100, 11)), policy, Deletion(), deletion_time_dist)


"""
    BranchingState(state, groupings)

Holds a batched state (or tuple of states) together with a matrix of group IDs
(`groupings::AbstractMatrix{<:Integer}`) with shape `(L, b)`, where `L` is the
sequence length and `b` is batch size. Elements only coalesce/split within the
same group.

Fields:
- `state`: a `MaskedState` or a tuple of `MaskedState`s representing the
  element-wise process state(s) at the current time.
- `groupings::AbstractMatrix{<:Integer}`: per-position group id.
- `del::AbstractMatrix{Bool}`: marks indices corresponding to to-be-deleted
  leaves in the conditional path.
- `ids::AbstractMatrix{Int}`: element ids for tracking; merged internal nodes
  produced during forest construction use id `0`.
- `branchmask::AbstractMatrix{Bool}`: where `true`, splits/deletions are
  permitted; where `false`, they are suppressed.
- `flowmask::AbstractMatrix{Bool}`: where `true`, the base process updates via
  `Flowfusion.step`; where `false`, state is held fixed.
- `padmask::AbstractMatrix{Bool}`: marks valid (unpadded) positions.

Convenience constructor:
    BranchingState(state, groupings; del=zeros(Bool,...), ids=1:size(groupings,1),
                   branchmask=ones(Bool,...), flowmask=ones(Bool,...),
                   padmask=ones(Bool,...))
"""
struct BranchingState{A,B,C,D,E,F,G} <: State
    state::A     #Flow state, or tuple of flow states
    groupings::B 
    del::C
    ids::D
    branchmask::E
    flowmask::F
    padmask::G
end

BranchingState(state, groupings; del =  zeros(Bool, size(groupings)), ids = 1:size(groupings, 1), branchmask = ones(Bool, size(groupings)), flowmask = ones(Bool, size(groupings)), padmask = ones(Bool, size(groupings))) = BranchingState(state, groupings, del, ids, branchmask, flowmask, padmask)

Base.copy(Xₜ::BranchingState) = deepcopy(Xₜ)

Flowfusion.resolveprediction(a, Xₜ::BranchingState) = a

"""
    uniform_del_insertions(X1::BranchingState, del_p)

Duplicate elements of `X1` with independent probability `del_p`, only among
positions where `flowmask & branchmask` are `true`. Each duplication inserts a
single copy either before or after the original (chosen uniformly), and exactly
one of the two (original or duplicate) is marked for deletion.

Returns a new `BranchingState` with:
- duplicated `state` entries, rewrapped as `MaskedState` with
  `flowmask` as both cond/label masks,
- updated `groupings`, `ids` (copied), `branchmask`, `flowmask`, `padmask`,
- `del` flags corresponding to the element selected for deletion per event.
"""


function uniform_del_insertions(X1::BranchingState, del_p) #X1 must be a BranchingState
    l = length(X1.groupings)
    elements = Flowfusion.element.((X1.state,), 1:l)
    #cmask, lmask = Flowfusion.getcmask(X1.state[1]), Flowfusion.getlmask(X1.state[1])
    fmask, pmask, bmask = X1.flowmask, X1.padmask, X1.branchmask
    del = (rand(l) .< del_p) .& fmask .& bmask
    new_indices = zeros(Int, l+sum(del))
    del_indices = zeros(Bool, l+sum(del))
    ind = 0
    for i in 1:l
        if del[i]
            new_indices[ind += 1] = i
            new_indices[ind += 1] = i
            if rand() < 0.5
                del_indices[ind] = true
            else
                del_indices[ind - 1] = true
            end
        else
            new_indices[ind += 1] = i
        end
    end
    return BranchingState(MaskedState.(regroup(elements[new_indices]), (fmask[new_indices],), (fmask[new_indices],)), X1.groupings[new_indices], del_indices, X1.ids[new_indices], bmask[new_indices], fmask[new_indices], pmask[new_indices])
end

export uniform_del_insertions



"""
    fixedcount_del_insertions(X1::BranchingState, num_events)

Insert exactly `num_events` duplication events into `X1`, sampling targets
uniformly with replacement from elements where `flowmask & branchmask` are
true. Each event inserts a duplicate either before or after the original
(chosen uniformly at random). For each event, exactly one of the two involved
indices (original or the newly inserted duplicate) is marked for deletion. If
multiple events hit the same original, only one can mark the original as
deleted; subsequent events allocate their deletion to their respective
duplicates.

Returns a new `BranchingState` constructed analogously to
`uniform_del_insertions`.
"""
function fixedcount_del_insertions(X1::BranchingState, num_events)
    l = length(X1.groupings)
    num_events <= 0 && return X1
    elements = Flowfusion.element.((X1.state,), 1:l)
    fmask, pmask, bmask = X1.flowmask, X1.padmask, X1.branchmask
    eligible = findall(fmask .& bmask)
    isempty(eligible) && return X1

    # Track per-index event allocations and which duplicates receive deletion
    before_flags = [Bool[] for _ in 1:l]
    after_flags = [Bool[] for _ in 1:l]
    orig_del = falses(l)

    for _ in 1:num_events
        i = eligible[rand(1:length(eligible))]
        if rand() < 0.5
            # insert before
            if rand(Bool) && !orig_del[i]
                orig_del[i] = true
                push!(before_flags[i], false)
            else
                push!(before_flags[i], true)
            end
        else
            # insert after
            if rand(Bool) && !orig_del[i]
                orig_del[i] = true
                push!(after_flags[i], false)
            else
                push!(after_flags[i], true)
            end
        end
    end

    new_indices = Vector{Int}(undef, l + num_events)
    del_indices = falses(l + num_events)
    ind = 0
    for i in 1:l
        # duplicates before
        for flag in before_flags[i]
            ind += 1
            new_indices[ind] = i
            del_indices[ind] = flag
        end
        # original
        ind += 1
        new_indices[ind] = i
        del_indices[ind] = orig_del[i]
        # duplicates after
        for flag in after_flags[i]
            ind += 1
            new_indices[ind] = i
            del_indices[ind] = flag
        end
    end

    return BranchingState(
        MaskedState.(regroup(elements[new_indices]), (fmask[new_indices],), (fmask[new_indices],)),
        X1.groupings[new_indices],
        del_indices,
        X1.ids[new_indices],
        bmask[new_indices],
        fmask[new_indices],
        pmask[new_indices],
    )
end

export fixedcount_del_insertions

#Not quite as tested as the others.
"""
    group_fixedcount_del_insertions(X1::BranchingState, group_num_events)

Insert a fixed number of duplication events per group as specified by the
dictionary `group_num_events`, which maps a group index (as found in
`X1.groupings`) to the number of events for that group.

Behavior per event matches `fixedcount_del_insertions`:
- Target indices are sampled uniformly with replacement among elements where
  `flowmask & branchmask` are true and whose group equals the requested group.
- Each event inserts a duplicate either before or after the original (chosen
  uniformly at random).
- For each event, exactly one of the two involved indices (original or newly
  inserted duplicate) is marked for deletion. If multiple events hit the same
  original, only one can mark the original as deleted; subsequent events
  allocate their deletion to their respective duplicates.

Returns a new `BranchingState` constructed analogously to
`uniform_del_insertions`.
"""
function group_fixedcount_del_insertions(X1::BranchingState, group_num_events)
    l = length(X1.groupings)
    # Quick exit if there are no positive requests
    has_any = false
    for (_, n) in group_num_events
        if n > 0
            has_any = true
            break
        end
    end
    has_any || return X1

    elements = Flowfusion.element.((X1.state,), 1:l)
    fmask, pmask, bmask = X1.flowmask, X1.padmask, X1.branchmask

    # Eligibility overall and by group (computed on-the-fly)
    eligible = findall(fmask .& bmask)
    isempty(eligible) && return X1

    # Track per-index event allocations and which duplicates receive deletion
    before_flags = [Bool[] for _ in 1:l]
    after_flags = [Bool[] for _ in 1:l]
    orig_del = falses(l)

    actual_events = 0
    for (g, n) in group_num_events
        n <= 0 && continue
        # Indices eligible for this specific group
        eligible_g = Int[]
        for i in eligible
            if X1.groupings[i] == g
                push!(eligible_g, i)
            end
        end
        isempty(eligible_g) && continue

        for _ in 1:n
            i = eligible_g[rand(1:length(eligible_g))]
            if rand() < 0.5
                # insert before
                if rand(Bool) && !orig_del[i]
                    orig_del[i] = true
                    push!(before_flags[i], false)
                else
                    push!(before_flags[i], true)
                end
            else
                # insert after
                if rand(Bool) && !orig_del[i]
                    orig_del[i] = true
                    push!(after_flags[i], false)
                else
                    push!(after_flags[i], true)
                end
            end
            actual_events += 1
        end
    end

    actual_events == 0 && return X1

    new_indices = Vector{Int}(undef, l + actual_events)
    del_indices = falses(l + actual_events)
    ind = 0
    for i in 1:l
        # duplicates before
        for flag in before_flags[i]
            ind += 1
            new_indices[ind] = i
            del_indices[ind] = flag
        end
        # original
        ind += 1
        new_indices[ind] = i
        del_indices[ind] = orig_del[i]
        # duplicates after
        for flag in after_flags[i]
            ind += 1
            new_indices[ind] = i
            del_indices[ind] = flag
        end
    end

    return BranchingState(
        MaskedState.(regroup(elements[new_indices]), (fmask[new_indices],), (fmask[new_indices],)),
        X1.groupings[new_indices],
        del_indices,
        X1.ids[new_indices],
        bmask[new_indices],
        fmask[new_indices],
        pmask[new_indices],
    )
end

export group_fixedcount_del_insertions

#This is a vestigial wrapper now - we will drop it and work directly.
split_target(P::CoalescentFlow, t, splits) = splits == 0 ? oftype(t, 0.0) : oftype(t, splits)


"""
Absolute time of the next split on this lineage.

H  : distribution for F_H on [0,1] with `cdf` and `quantile`
W  : descendant count at the next node, m = W - 1 must be ≥ 1
t0 : current absolute time in [0,1)

Implements: draw E ~ Exp(1), set S* = (1 - cdf(H,t0)) * exp(-E/m),
return quantile(H, 1 - S*).
"""
function next_split_time(H, W, t0::T) where T
    @assert 0.0 ≤ t0 < 1.0 "t0 must be in [0,1)."
    @assert W ≥ 2 "Need W ≥ 2 so m = W - 1 ≥ 1."
    m  = W - 1
    S0 = 1 - cdf(H, t0)
    @assert S0 > 0 "No support beyond t0: cdf(H, t0) = 1."
    E = rand(Exponential())
    S_star = S0 * exp(-E / m)
    p = 1 - S_star
    t = quantile(H, p)
    return T(clamp(t, t0, 1))
end

"""
    sample_split_times!(P::CoalescentFlow, node::FlowNode, t0; collection=nothing)

Assign absolute split times to all internal nodes in the subtree rooted at
`node` using `next_split_time(P.branch_time_dist, node.weight, t0)` whenever
`node.weight > 1`. Recurses into children with the newly assigned time as the
lower bound.

If `collection` is provided (a vector), pushes each sampled absolute time into
it. The collected times are not sorted.
"""
function sample_split_times!(P::CoalescentFlow, node::FlowNode, t0::T; collection = nothing) where T
    if node.weight > 1
        nextsplit = next_split_time(P.branch_time_dist, node.weight, t0)
        node.time = nextsplit
        if !isnothing(collection)
            push!(collection, nextsplit)
        end
        for child in node.children
            sample_split_times!(P, child, nextsplit; collection = collection)
        end
    end
end


"""
    sample_forest(P::CoalescentFlow, elements;
                  groupings=zeros(Int, length(elements)),
                  branchable=ones(Bool, length(elements)),
                  flowable=ones(Bool, length(elements)),
                  deleted=zeros(Bool, length(elements)),
                  ids=1:length(elements),
                  T=Float32,
                  coalescence_factor=1.0,
                  merger=canonical_anchor_merge,
                  coalescence_policy=P.coalescence_policy,
                  group_mins=nothing)

Sample a coalescent forest over `elements` with per-element `groupings` and
boolean `branchable` flags. Returns `(forest_nodes, coal_times)` where
`forest_nodes` is a vector of `FlowNode` roots (one per surviving group block),
and `coal_times` is a vector of absolute split times sampled across the forest
(not sorted).

Arguments:
- `flowable`: marks elements whose state should be bridged (non-flowables are
  emitted as fixed segments during conditional path sampling).
- `deleted`: marks which leaves correspond to to-be-deleted elements at t=1.
- `ids`: element ids to carry through the forest (merged internal nodes get id
  `0`; merged nodes are always flowable and never deleted).
- `coalescence_factor`: Binomial parameter scaling the maximum possible number
  of coalescences computed by the policy. May be a numeric in [0,1] or a
  `UnivariateDistribution` (sampled once per call). `1.0` collapses each group
  to one root; `0.0` yields no coalescences.
- `merger`: function `merger(left_state, right_state, w_left, w_right)` used to
  build anchor states for internal nodes; see `canonical_anchor_merge` and
  `select_anchor_merge`.
- `coalescence_policy`: chooses which pair to coalesce at each event; see
  `merging.jl` for available policies and their semantics.
- `group_mins`: forwarded to `select_coalescence`. Supported forms depend on
  the active policy. For `SequentialUniform`, supported values are:
  `nothing` (no per-group minimum), `Dict{Int,Int}` (fixed per-group minima),
  or `Int` (uniform minimum for contiguous branchable blocks per group).
"""
function sample_forest(P::CoalescentFlow, elements::AbstractVector; 
        groupings = zeros(Int, length(elements)), 
        branchable = ones(Bool, length(elements)), 
        flowable = ones(Bool, length(elements)),
        deleted = zeros(Bool, length(elements)),
        ids = 1:length(elements),
        T = Float32, 
        coalescence_factor = 1.0,
        merger = canonical_anchor_merge,
        coalescence_policy = P.coalescence_policy,
        group_mins = nothing, #This will not allow any merges when there are fewer than this many elements for each group. Needs to be a dictionary that maps group indices to minimum sizes.
        )
    nodes = FlowNode[FlowNode(T(1), elements[i], 1, groupings[i], branchable[i], deleted[i], ids[i], flowable[i]) for i = 1:length(elements)]
    init!(coalescence_policy, nodes)
    max_merge_count = max_coalescences(coalescence_policy, nodes)
    if coalescence_factor isa UnivariateDistribution
        coalescence_factor = rand(coalescence_factor)
    end
    sampled_merges = rand(Binomial(max_merge_count, coalescence_factor))
    for _ in 1:sampled_merges
        pair = select_coalescence(coalescence_policy, nodes, group_mins)
        pair === nothing && break
        i, j = pair
        if i > j
            i, j = j, i
        end
        left, right = nodes[i], nodes[j]
        if !(left.group == right.group || allows_cross_group_merge(coalescence_policy))
            throw(ArgumentError("Selected merge across groups ($(left.group), $(right.group))) is not permitted by policy."))
        end
        @assert left.branchable && right.branchable
        #Merged nodes can never be deleted. Merged nodes get an id of 0. Merged nodes are always flowable.
        merged_group = left.group
        merged = mergenodes(left, right, T(0), merger(left.node_data, right.node_data, left.weight, right.weight), left.weight + right.weight, merged_group, true, false, 0, true)
        nodes[i] = merged
        deleteat!(nodes, j)
        update!(coalescence_policy, nodes, i, j, i)
    end
    # Allow policy to reorder forest (roots/children) prior to time sampling
    nodes = reorder_forest(coalescence_policy, nodes)
    # Recursively sample waiting times after final ordering
    col = T[]
    sample_split_times!.((P,), nodes, T(0); collection = col)
    return nodes, col
end

"""
    tree_bridge(P::CoalescentFlow, node, Xs, target_t, current_t, collection)

Recursively traverse a `FlowNode` tree, running the underlying bridge
`bridge(P.P, Xs, node.node_data, current_t, next_t)` either up to `target_t`
or the node’s own split time. For each branch that crosses `target_t`, appends
to `collection` a named tuple with fields:
- `Xt`: the bridged state at `target_t` (or the node’s anchor if non-flowable),
- `t`: the evaluation time used for this segment,
- `X1anchor`: the anchor (node_data) ahead of the segment,
- `descendants`: the node’s descendant count (weight),
- `del`: whether this branch terminates in deletion at t=1,
- `branchable`: whether splits are permitted on this segment,
- `flowable`: whether the base process should evolve this segment,
- `group`: group id,
- `last_coalescence`: the most recent coalescence/split time behind the
  segment (open interval start),
- `id`: element id carried from the leaf; internal nodes use `0`.

Deletion handling:
- During conditional path sampling, deletions use the exact survival ratio over
  `[current_t, target_t]` induced by `P.deletion_time_dist` (i.e., survival
  S(t)/S(current_t)), removing segments that fail to survive to `target_t`.
"""
function tree_bridge(P::CoalescentFlow, node, Xs, target_t, current_t, collection)
    if !node.flowable #All state elements will have X0==X1.
        push!(collection, (;Xt = node.node_data, t = target_t, X1anchor = node.node_data, descendants = node.weight, del = node.del, branchable = false, flowable = false, group = node.group, last_coalescence = current_t, id = node.id))
        return
    end
    if node.time > target_t #<-If we're on the branch where a sample is needed
        # Deletion with general hazard: survive interval [current_t, target_t] with prob S(t)/S(current_t)
        if !(node.del && begin
            S_curr = max(1 - cdf(P.deletion_time_dist, current_t), 0.0)
            S_tgt = max(1 - cdf(P.deletion_time_dist, target_t), 0.0)
            surv_ratio = S_curr > 0 ? (S_tgt / S_curr) : 0.0
            rand() < (1 - surv_ratio)
        end)
            Xt = bridge(P.P, Xs, node.node_data, current_t, target_t)
            push!(collection, (;Xt, t = target_t, X1anchor = node.node_data, descendants = node.weight, del = node.del, branchable = node.branchable, flowable = true, group = node.group, last_coalescence = current_t, id = node.id))
        end
    else
        next_Xs = bridge(P.P, Xs, node.node_data, current_t, node.time)
        for child in node.children
            tree_bridge(P, child, next_Xs, target_t, node.time, collection)
        end
    end
end

"""
    forest_bridge(P::CoalescentFlow, X0sampler, X1, t, groups, branchable, flowable, deleted;
                  T=Float32, use_branching_time_prob=0, maxlen=Inf,
                  coalescence_factor=1.0, merger=canonical_anchor_merge,
                  coalescence_policy=P.coalescence_policy, group_mins=nothing)

Run a single conditional bridge at time `t` for each root in a forest sampled
from `X1` and `groups`. The forest is built with `sample_forest` using the
provided `branchable`, `flowable`, and `deleted` flags. Returns a flat vector
of segment tuples (see `tree_bridge`).

Arguments:
- `X0sampler(root)`: a function that, given a forest root node, returns the
  initial state to bridge from.
- `use_branching_time_prob`: with this probability, override `t` with a random
  absolute split time drawn from the forest’s `coal_times`, exposing the model
  to states exactly at split points.
- `maxlen`: if the number of segments at/after `t` would exceed `maxlen`,
  this function resamples the forest to keep the total bounded.
- `group_mins`: forwarded through to `sample_forest` (see its docstring).
"""
function forest_bridge(P::CoalescentFlow, X0sampler, X1, t, groups, branchable, flowable, deleted; T = Float32, use_branching_time_prob = 0, maxlen = Inf, coalescence_factor = 1.0, merger = canonical_anchor_merge, coalescence_policy = P.coalescence_policy, group_mins = nothing)
    elements = element.((X1,), 1:length(groups))
    forest, coal_times = sample_forest(P, elements; groupings = groups, branchable, flowable, deleted, coalescence_factor, merger, coalescence_policy, T, group_mins)
    if (rand() < use_branching_time_prob) && (length(coal_times) > 0)
        t = rand(coal_times)
    end
    if (length(forest) + sum(coal_times .<= t)) > maxlen #This resamples if you wind up greater than the max length. I should get rid of this.
        print("!")
        return forest_bridge(P, X0sampler, X1, t, groups, branchable, flowable, deleted; use_branching_time_prob, maxlen, coalescence_factor, merger, coalescence_policy, T, group_mins)
    end
    collection = []
    for root in forest
        tree_bridge(P, root, X0sampler(root), t, T(0), collection);
    end
    return collection
end

#=
if length_mins is:
1) nothing, then no min is every enforced and the coalescence amount will be governed by the coalescence_factor.
2) Int, in which case the min is set as this int for each contiguous designable segment of the sequence.
3) DiscreteUnivariateDistribution, in which case the min is set as rand(group_mins) for each contiguous designable segment of the sequence.
4) Dict{a::Int => b::Int}, in which case the min is set as b for group a.
5) Dict{a::Int => b::DiscreteUnivariateDistribution}, in which case the min is set as rand(b) for group a, drawn once before the bridge.
6) AbstractVector of any of the above, with a length that matches the number of elements in the batch.
=#

#resolve_group_mins converts different sorts of arguments into explicit group->count dictionaries.
resolve_group_mins(length_mins::Nothing, groupings) = Dict(k => 1 for k in unique(groupings)) #Min of 1 for each group if nothing.
resolve_group_mins(length_mins::Int, groupings) = Dict(k => length_mins for k in unique(groupings)) #Min of 1 for each group if nothing.
resolve_group_mins(length_mins::Dict{Int,Int}, groupings) = length_mins #User-specified fixed dictionary.
resolve_group_mins(length_mins::Dict{Int,<:DiscreteUnivariateDistribution}, groupings) = Dict(k => 1 + rand(v) for (k, v) in length_mins) #User specified per-group distribution.
resolve_group_mins(length_mins::DiscreteUnivariateDistribution, groupings) = Dict(k => 1 + rand(length_mins) for k in unique(groupings)) #Independent length draw per group.
    
"""
    branching_bridge(P::CoalescentFlow, X0sampler, X1s, times;
                     T=Float32, use_branching_time_prob=0, maxlen=Inf,
                     coalescence_factor=1.0, merger=canonical_anchor_merge,
                     coalescence_policy=P.coalescence_policy,
                     length_mins=nothing, deletion_pad=0, X1_modifier=identity)

Vectorized conditional bridging over a batch. For each `(X1, t)` pair, samples
an independent forest (respecting group minima and policy), runs `tree_bridge`
for all roots and aggregates the outputs into batched tensors.

Arguments:
- `times`: vector of times (length = batch size) or a `UnivariateDistribution`
  (a time is drawn per batch item).
- `length_mins`: per-group minima control for forest construction. May be:
  `nothing`, `Int`, `DiscreteUnivariateDistribution`, `Dict{Int,Int}`,
  `Dict{Int,<:DiscreteUnivariateDistribution}`, or a vector of any of these
  (length equal to batch size).
- `deletion_pad`: if > 0, pads each `X1` with additional to-be-deleted
  duplicates so that, in expectation, each group has
  `deletion_pad * max(len(x0_group), len(x1_group))` elements; implemented via
  per-group Poisson draws and `group_fixedcount_del_insertions`.
- `X1_modifier`: optional transform applied to each (possibly deletion-padded)
  `X1` prior to forest sampling (e.g., to enforce masking on deleted states).

Returns a named tuple:
- `t::Vector{T}`: the time used per batch item (possibly overridden by a split time).
- `Xt::BranchingState`: batched states at time `t`, with `state` as masked
  states (or tuple thereof) and bookkeeping masks/ids/groupings.
- `X1anchor`: masked anchor states batched in the same structure as `Xt.state`.
- `del::Matrix{Bool}`: deletion flags per element in `Xt`.
- `descendants::Matrix{Int}`: descendant counts (w) per element.
- `splits_target::Matrix{T}`: per-element training targets for split heads,
  computed as `max(descendants-1, 0)`.
- `prev_coalescence::Matrix{T}`: last coalescence time before each element.
"""
function branching_bridge(  P::CoalescentFlow, 
                            X0sampler, 
                            X1s, 
                            times; 
                            T = Float32, 
                            use_branching_time_prob = 0, 
                            maxlen = Inf,
                            coalescence_factor = 1.0, 
                            merger = canonical_anchor_merge, 
                            coalescence_policy = P.coalescence_policy, 
                            length_mins = nothing,
                            deletion_pad = 0,
                            X1_modifier = identity) #If you use deletion pad, it is tricky to eg. set the deleted token to mask.
    #This should be moved inside forest_bridge.
    if times isa UnivariateDistribution
        times = rand(times, length(X1s))
    end
    times = T.(times)
    #To do: make this work (or check that it works) when X1.state is not masked.

    groupings = [x.groupings for x in X1s]    
    resolved_mins = nothing
    if length_mins isa AbstractVector 
        resolved_mins = resolve_group_mins.(length_mins, groupings) #<-One per X1.
    else                
        resolved_mins = [resolve_group_mins(length_mins, groupings[i]) for i in 1:length(times)] #<-One, for all X1s globally.
    end
    #Pad deletions to match deletion_pad*max(len(x0),len(x1)) in expectation.
    #If deletion_pad >= 1, this guarantees that the del-padded X1 length will be at least len(x0).
    if deletion_pad > 0
        X1_lengths = countmap.(groupings)
        deletion_pad_counts = copy(X1_lengths)
        for i in 1:length(X1s)
            for j in 1:length(keys(X1_lengths[i]))
                total_expected = deletion_pad * max(X1_lengths[i][j], resolved_mins[i][j])
                deletion_pad_counts[i][j] = rand(Poisson(total_expected - X1_lengths[i][j]))
            end
            
        end
        X1s = [X1_modifier(group_fixedcount_del_insertions(X1, deletion_pad_counts[i])) for (i, X1) in enumerate(X1s)]
    end

    batch_bridge = [forest_bridge(P, X0sampler, X1.state, times[i], X1.groupings, X1.branchmask, X1.flowmask, X1.del; use_branching_time_prob, maxlen, coalescence_factor, merger, coalescence_policy, group_mins = resolved_mins[i]) for (i, X1) in enumerate(X1s)]
    
    maxlen = maximum(length.(batch_bridge))
    b = length(X1s)
    flowmask = ones(Bool, maxlen, b)
    branchmask = ones(Bool, maxlen, b)
    descendants = zeros(Int, maxlen, b)
    padmask = zeros(Bool, maxlen, b)
    del = zeros(Bool, maxlen, b)
    prev_coalescence = zeros(T, maxlen, b)
    groups = zeros(Int, maxlen, b)
    ids = zeros(Int, maxlen, b)
    for b in 1:length(batch_bridge)
        for i in 1:length(batch_bridge[b])
            flowmask[i,b] = batch_bridge[b][i].flowable
            branchmask[i,b] = batch_bridge[b][i].branchable
            descendants[i,b] = batch_bridge[b][i].descendants
            padmask[i,b] = true
            del[i,b] = batch_bridge[b][i].del
            prev_coalescence[i,b] = batch_bridge[b][i].last_coalescence
            groups[i,b] = batch_bridge[b][i].group
            ids[i,b] = batch_bridge[b][i].id
        end
    end

    used_times = [b[1].t for b in batch_bridge]
    splits_target = split_target.((P,), used_times', clamp.(descendants .- 1, 0, Inf)) #<-Can just be descendants .- 1 now!
    Xt_batch = MaskedState.(regroup([[b.Xt for b in bridges] for bridges in batch_bridge]), (flowmask,), (padmask .& flowmask,));
    X1anchor_batch = MaskedState.(regroup([[b.X1anchor for b in bridges] for bridges in batch_bridge]), (flowmask,), (padmask .& flowmask,));
    return (;t = used_times, Xt = BranchingState(Xt_batch, groups; ids, branchmask, flowmask, padmask), X1anchor = X1anchor_batch, del, descendants, splits_target, prev_coalescence)
end



"""
    Flowfusion.step(P::CoalescentFlow, XₜBS::BranchingState, hat::Tuple, s₁::Real, s₂::Real)

Advance the process forward in time from `s₁` to `s₂` for a single-batch
`BranchingState`, allowing split events. The underlying continuous/discrete
states are advanced via `Flowfusion.step(P.P, ...)`, and split counts are drawn
from a Poisson law parameterized by the transformed event intensities and a
factor of the truncated branch-time density at `s₁`.

Deletion handling:
- Uses a small-step hazard approximation induced by `P.deletion_time_dist`.
- Let `h(s₁) = f(s₁) / S(s₁)` where `f`/`S` are the pdf/survival. The
  base small-step probability is `1 - exp(-h(s₁) * Δt)`, scaled by
  `ρ = sigmoid(del_logits)` from the model head. Deletions are suppressed where
  `branchmask` is `false`.

Split handling:
- Per-position split counts are sampled from `Poisson(Δt * split_transform(λ) *
  pdf(Truncated(branch_time_dist, s₁, 1), s₁))`, masked by `branchmask`.
- If a discrete component changes its token at an index between `s₁` and `s₂`,
  splits at that index are suppressed for this step to avoid simultaneous
  discrete changes and splits.
- New elements are inserted adjacently after the split location (no append mode).

Returns a new single-batch `BranchingState` where:
- `state` holds the updated masked states,
- `groupings`, `flowmask`, `branchmask` are updated to reflect splits/deletions,
- default `ids`, `padmask`, and `del` are constructed consistently with shapes.
"""
function Flowfusion.step(P::CoalescentFlow, XₜBS::BranchingState, hat::Tuple, s₁::Real, s₂::Real)
    Xₜ = XₜBS.state
    time_remaining = (1-s₁)
    delta_t = s₂ - s₁
    X1targets, event_lambdas, del_logits = hat
    neXₜ = Flowfusion.step(P.P, Xₜ, X1targets, s₁, s₂)
    Xₜ = Flowfusion.mask(neXₜ, Xₜ)
    bmask = XₜBS.branchmask
    fmask = XₜBS.flowmask
    splits = bmask .* rand.(Poisson.((delta_t*P.split_transform.(event_lambdas) * pdf(Truncated(P.branch_time_dist, s₁, 1), s₁))))[:]
    # Deletions with general hazard: small-step approximation over [s₁, s₂]
    # using instantaneous hazard h(s₁) = f(s₁)/S(s₁):
    #   base_p ≈ 1 - exp(-h(s₁) * Δt), then p_del ≈ ρ * base_p
    S₁ = max(1 - cdf(P.deletion_time_dist, s₁), 0.0)
    f₁ = pdf(P.deletion_time_dist, s₁)
    h₁ = S₁ > 0 ? (f₁ / S₁) : 0.0
    base_p = 1 .- exp.(-h₁ * delta_t)
    rho = exp.(_logσ.(del_logits))  # in (0,1), elementwise
    dels = bmask .* (rand(length(splits)) .< (rho .* base_p))

    #Prevents a split if a discrete state change happens.
    splits[dels] .= 0
    for s in 1:length(XₜBS.state)
        if (XₜBS.state[s] isa DiscreteState)
            for i in 1:length(splits)
                if element(tensor(Xₜ[s]),i,1) != element(tensor(XₜBS.state[s]),i,1)
                    #println("Discrete component $s changed at $i")
                    splits[i] = 0
                end
            end
        end
    end

    current_length = size(tensor(first(Xₜ)))[end-1]
    new_length = current_length + sum(splits) - sum(dels)
    element_tuple = element.(Xₜ, 1, 1)
    newstates = Tuple([Flowfusion.zerostate(element_tuple[i],new_length,1) for i in 1:length(element_tuple)])
    newgroupings = similar(XₜBS.groupings, new_length, 1) .= 0
    newflowmask = similar(XₜBS.flowmask, new_length, 1) .= 0
    newbranchmask = similar(XₜBS.branchmask, new_length, 1) .= 0

    # Default adjacent insertion
    current_index = 1
    for i in 1:current_length
        if !dels[i]
            for s in 1:length(element_tuple)
                element(tensor(newstates[s]),current_index,1) .= tensor(element(Xₜ[s],i,1))
                newgroupings[current_index] = XₜBS.groupings[i,1]
                newflowmask[current_index] = XₜBS.flowmask[i,1]
                newbranchmask[current_index] = XₜBS.branchmask[i,1]
            end
            current_index += 1
            for j in 1:splits[i]
                for s in 1:length(element_tuple)
                    element(tensor(newstates[s]),current_index,1) .= tensor(element(Xₜ[s],i,1))
                    newgroupings[current_index] = XₜBS.groupings[i,1]
                    newflowmask[current_index] = XₜBS.flowmask[i,1]
                    newbranchmask[current_index] = XₜBS.branchmask[i,1]
                end
                current_index += 1
            end
        end
    end
    return BranchingState(MaskedState.(newstates, (newflowmask,), (newflowmask,)), newgroupings, branchmask = newbranchmask, flowmask = newflowmask)
end
