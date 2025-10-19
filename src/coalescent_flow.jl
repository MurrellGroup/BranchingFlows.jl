struct UniformDeletion end

"""
    CoalescentFlow{Proc,D,F,Pol}

Branching/coalescent flow wrapper that augments an underlying state-space
process `P` with stochastic coalescence/splitting events.

Fields:
- `P::Proc`: underlying Flowfusion process (or tuple of processes) used for
  bridging between coalescence times.
- `branch_time_dist::D`: distribution over coalescence times in (0, 1], used to
  sample event times (sorted descending, because coalescence walks backward).
- `split_transform::F`: elementwise map that converts model-predicted event
  logits into nonnegative intensities for split counts during forward-time
  simulation.
- `coalescence_policy::Pol`: policy controlling which elements coalesce at each
  event; see policies in `merging.jl`.

Constructors:
- `CoalescentFlow(P, branch_time_dist)` uses `exp(clamp(x, -100, 11))` as
  `split_transform` and `SequentialUniform()` as policy.
- `CoalescentFlow(P, branch_time_dist, policy)` uses the same default
  `split_transform`, but sets the provided policy.

Notes:
- Sequential policies assume the sequence order is meaningful. Non-sequential
  policies assume the model is invariant to sequence permutations.
"""
struct CoalescentFlow{Proc,D,F,Pol,Delp} <: Process
    P::Proc
    branch_time_dist::D
    split_transform::F
    coalescence_policy::Pol
    deletion_policy::Delp
end
CoalescentFlow(P, branch_time_dist) = CoalescentFlow(P, branch_time_dist, x -> exp.(clamp.(x, -100, 11)), SequentialUniform(), UniformDeletion())
CoalescentFlow(P, branch_time_dist, policy) = CoalescentFlow(P, branch_time_dist, x -> exp.(clamp.(x, -100, 11)), policy, UniformDeletion())
CoalescentFlow(P, branch_time_dist, policy, deletion_policy) = CoalescentFlow(P, branch_time_dist, x -> exp.(clamp.(x, -100, 11)), policy, deletion_policy)


#I think we need to add lmask and cmask to the BranchingState, and just use clean underlying states. Will need to pipe losses etc to use the outer masks.
#Otherwise the user has to do too much extra BS.
"""
    BranchingState(state, groupings)

Holds a batched state (or tuple of states) together with a matrix of group IDs
(`groupings::AbstractMatrix{<:Integer}`) with shape `(L, b)`, where `L` is
sequence length and `b` is batch size. Elements only coalesce within the same
group.
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


#ToDo: add other versions of this where we pre-specify the number of deletions, but distribute them randomly across the sequence (maybe allowing multiple deletions per element).
#This will let us pre-draw coalescence min lengths, and coordinate to allow any X0 length to pair with any X1 length.
#This will typically involve looking at the X1 length, sampling an X0 length, then computing coalescence min and deletion count to match the difference.
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



#This is pointless now - we can drop it I think:
"""
    split_target(P::CoalescentFlow, t, splits)

Training target for the split intensity head at time `t`. Returns `0` if no
descendants follow from the element, otherwise the number of splits expected.
This is the unscaled hazard target; scaling by the branch time density is
handled during sampling.
"""
split_target(P::CoalescentFlow, t, splits) = splits == 0 ? oftype(t, 0.0) : oftype(t, splits) #For splits-as-rate, drop this last (1-t) term

function possible_merges(nodes)
    merge_mask = zeros(Bool, length(nodes))
    for i in 1:length(nodes)-1
        if nodes[i].branchable && nodes[i+1].branchable && nodes[i].group == nodes[i+1].group
            merge_mask[i] = true
        end
    end
    return merge_mask
end



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
    sample_forest(P::CoalescentFlow, elements; groupings, branchable, T=Float32,
                  coalescence_factor=1.0, merger=canonical_anchor_merge,
                  coalescence_policy=P.coalescence_policy)

Sample a coalescent forest over `elements` with per-element `groupings` and
boolean `branchable` flags. Returns `(forest_nodes, coal_times)` where
`forest_nodes` is a vector of `FlowNode` roots (one per surviving group block)
and `coal_times` are the sampled coalescence times (descending).

Arguments:
- `coalescence_factor`: Binomial parameter that scales the maximum number of
  possible coalescences under the policy; `1.0` tends to collapse each group to
  one element, `0.0` produces no coalescences.
- `merger`: function used to build anchor states when two nodes coalesce.
- `coalescence_policy`: chooses which pair to coalesce at each event; defaults
  to `P.coalescence_policy`.
- `group_mins`: either nothing, or a dictionary that maps group indices to minimum sizes.
If provided, will not allow any merges when there are fewer than this many elements for each group.
Useful if you need specific size control over the X0 distribution so you know what to sample from.
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
        @assert left.group == right.group
        @assert left.branchable && right.branchable
        #Merged nodes can never be deleted. Merged nodes get an id of 0. Merged nodes are always flowable.
        merged = mergenodes(left, right, T(0), merger(left.node_data, right.node_data, left.weight, right.weight), left.weight + right.weight, left.group, true, false, 0, true)
        nodes[i] = merged
        deleteat!(nodes, j)
        update!(coalescence_policy, nodes, i, j, i)
    end
    #Recursively sample waiting times:
    col = T[]
    sample_split_times!.((P,), nodes, T(0); collection = col)
    return nodes, col
end

"""
    tree_bridge(P::CoalescentFlow, node, Xs, target_t, current_t, collection)

Recursively traverse a `FlowNode` tree, running the underlying bridge from
`current_t` to `target_t` on branches that cross the target time. Appends a
named tuple for each leaf-like segment to `collection` with fields:
`Xt`, `X1anchor`, `descendants`, `cmasked`, `group`, `last_coalescence`.
"""
function tree_bridge(P::CoalescentFlow, node, Xs, target_t, current_t, collection)
    if !node.flowable #All state elements will have X0==X1.
        push!(collection, (;Xt = node.node_data, t = target_t, X1anchor = node.node_data, descendants = node.weight, del = node.del, branchable = false, flowable = false, group = node.group, last_coalescence = current_t, id = node.id))
        return
    end
    if node.time > target_t #<-If we're on the branch where a sample is needed
        #WILL NEED MODIFICATION FOR RATE SCHEDULED DELETIONS:
        if !(node.del && (rand() < (target_t-current_t)/(node.time - current_t))) #<- Sample only included if not deleted.
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
    forest_bridge(P::CoalescentFlow, X0sampler, X1, t, groups, branchable;
                  maxlen=Inf, coalescence_factor=1.0,
                  merger=canonical_anchor_merge,
                  coalescence_policy=P.coalescence_policy)

Run a single bridge at time `t` for each tree in the sampled forest built from
`X1` and `groups`. Returns a flat vector of segment tuples (see `tree_bridge`).

Notes:
- If the sampled forest plus early-time branches would exceed `maxlen`, the
  function resamples to keep the total length bounded.
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

resolve_group_mins(length_mins) = length_mins
resolve_group_mins(length_mins::Dict{Int,<:DiscreteUnivariateDistribution}) = Dict(k => rand(v) for (k, v) in length_mins)
resolve_group_mins(length_mins::DiscreteUnivariateDistribution) = rand(length_mins)

"""
    branching_bridge(P::CoalescentFlow, X0sampler, X1s, times;
                     maxlen=Inf, coalescence_factor=1.0,
                     merger=canonical_anchor_merge,
                     coalescence_policy=P.coalescence_policy)

Vectorized bridge over a batch of inputs. For each `(X1, t)` pair, samples a
forest (per `coalescence_policy`), runs `tree_bridge` and aggregates outputs
into batched `MaskedState`s plus bookkeeping.
`use_branching_time_prob` lets you use an actual coalescence time as the bridge time, which allows the model
to see states right at the split point, which will be the case during inference.

Returns a named tuple with fields:
- `t`: the times provided
- `Xt`: `BranchingState` of the bridged states (masked)
- `X1anchor`: `BranchingState` of anchors matched to each `Xt`
- `descendants`: counts per segment
- `splits_target`: target split intensities for training heads
- `branchmask`, `padmask`, `prev_coalescence`: masks and last event time

Example:
```julia
P = CoalescentFlow(BrownianMotion(), Uniform(0.0f0, 1.0f0), last_to_nearest_coalescence(state_index=1))
out = branching_bridge(P, X0sampler, X1s, times; coalescence_factor=0.8)
```
"""
function branching_bridge(P::CoalescentFlow, X0sampler, X1s, times; T = Float32, use_branching_time_prob = 0, maxlen = Inf, coalescence_factor = 1.0, merger = canonical_anchor_merge, coalescence_policy = P.coalescence_policy, length_mins = nothing)
    #This should be moved inside forest_bridge.
    if times isa UnivariateDistribution
        times = rand(times, length(X1s))
    end
    times = T.(times)
    #To do: make this work (or check that it works) when X1.state is not masked.
    #Even better, build the mask into the BranchingState directly, so you don't need to duplicate them. Might require extra piping.

    resolved_mins = nothing
    if length_mins isa AbstractVector
        resolved_mins = resolve_group_mins.(length_mins)
    else
        resolved_mins = [resolve_group_mins(length_mins) for _ in 1:length(times)]
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
from a Poisson law parameterized by the transformed event intensities and the
truncated branch-time density at `s₁`.

Insertion behavior on split is controlled by the coalescence policy trait
`should_append_on_split(P.coalescence_policy)`:
- If `false` (default), new elements are inserted adjacently after the split
  location.
- If `true`, new elements are appended to the end of their group block (not the
  entire sequence), preserving original element order.

Returns a new `BranchingState` with updated states and groupings.
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
    
    dels = bmask .* (rand(length(splits)) .< (exp.(_logσ.(del_logits)) .* (delta_t/time_remaining)))

    #Bit hacky - should do a Gillespie-like step instead.
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
