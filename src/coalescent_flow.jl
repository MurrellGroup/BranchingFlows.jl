#Note 1: How to handle deletions:
#At some rate (which can maybe vary across states) we randomly duplicate in-place an observation.
#The duplicated copy (either left or right) gets tagged with a "deleted" flag (and maybe any discrete state gets set to "dummy"? Probably a good idea)
#Everything else proceeds the same, but we also return a deletion target. Maybe this should be DFM-style where the model outputs a deletion logit that gets softmaxed etc?
#Consequences: the model's split count should include the splits which will then get deleted!

#Note 2: We must generalize the "merge" mechanism.
#In particular, we need modes that induce correlated merges, which will cause little "edit runs" in the conditional paths.
#Also, we need a mode where the merges aren't adjacent pairs, for data types where there is no primary sequence ordering.
#These could be controlled by a pairwise "merge propensity" matrix. For Euc data this could be derived from the X1 pairwise distances.
#Note: we'll need merge proponsities to me "recursive" so a merged element's propensity to all other elements can be computed from the merge propensities of the merged elements.
#UPGMA style?



#This is the good way to get the elements.
#Need to restrict on lmask though:
#@time a = element.((compound_state,), 1:seqlength(compound_state), 1);

#Add the coalescence factor *distribution* to the process.
struct CoalescentFlow{Proc,D,F,Pol} <: Process
    P::Proc
    branch_time_dist::D
    split_transform::F
    coalescence_policy::Pol
end
CoalescentFlow(P, branch_time_dist) = CoalescentFlow(P, branch_time_dist, x -> exp.(clamp.(x, -100, 11)), SequentialUniform())
CoalescentFlow(P, branch_time_dist, policy) = CoalescentFlow(P, branch_time_dist, x -> exp.(clamp.(x, -100, 11)), policy)


#I think we need to add lmask and cmask to the BranchingState, and just use clean underlying states. Will need to pipe losses etc to use the outer masks.
#Otherwise the user has to do too much extra BS.
struct BranchingState{A,B} <: State
    state::A     #Flow state, or tuple of flow states
    groupings::B 
end

Base.copy(Xₜ::BranchingState) = deepcopy(Xₜ)

Flowfusion.resolveprediction(a, Xₜ::BranchingState) = a

#Note: Swapping to predicting the unscaled hazard - note this is a constant factor, so still linear in the generator.
#The hazard scaling is multiplied back in prior to sampling.
#split_target(P::CoalescentFlow, t, splits) = splits == 0 ? oftype(t, 0.0) : oftype(t, splits * pdf(Truncated(P.branch_time_dist, t, 1), t) * (1-t)) #For splits-as-rate, drop this last (1-t) term
split_target(P::CoalescentFlow, t, splits) = splits == 0 ? oftype(t, 0.0) : oftype(t, splits) #For splits-as-rate, drop this last (1-t) term

function possible_merges(nodes)
    merge_mask = zeros(Bool, length(nodes))
    for i in 1:length(nodes)-1
        if nodes[i].free && nodes[i+1].free && nodes[i].group == nodes[i+1].group
            merge_mask[i] = true
        end
    end
    return merge_mask
end

#This returns a forest, which is a vector of trees, annotated with anchors
function sample_forest(P::CoalescentFlow, elements::AbstractVector; 
        groupings = zeros(Int, length(elements)), 
        flowable = ones(Bool, length(elements)), 
        T = Float32, 
        coalescence_factor = 1.0,
        merger = canonical_anchor_merge,
        coalescence_policy = P.coalescence_policy,
        ) #When coalescence_factor is 1, all groups will collapse down to one node. When it is 0, no coalescences will occur (or it will error maybe).
    nodes = FlowNode[FlowNode(T(1), elements[i], 1, groupings[i], flowable[i]) for i = 1:length(elements)]
    init!(coalescence_policy, nodes)
    max_merge_count = max_coalescences(coalescence_policy, nodes)
    sampled_merges = rand(Binomial(max_merge_count, coalescence_factor))
    coal_times = T.(sort(rand(P.branch_time_dist, sampled_merges), rev = true)) #Because coal stuff walks backwards from 1
    for time in coal_times
        pair = select_coalescence(coalescence_policy, nodes; time = time) #To make correlated coalescences more likely, this will need to be non-independent.
        pair === nothing && break
        i, j = pair
        if i > j
            i, j = j, i
        end
        left, right = nodes[i], nodes[j]
        @assert left.group == right.group
        @assert left.free && right.free
        merged = mergenodes(left, right, time, merger(left.node_data, right.node_data, left.weight, right.weight), left.weight + right.weight, left.group, true)
        nodes[i] = merged
        deleteat!(nodes, j)
        update!(coalescence_policy, nodes, i, j, i)
    end
    return nodes, coal_times
end

#This takes an ordered vector of states and runs the bridge
function tree_bridge(P::CoalescentFlow, node, Xs, target_t, current_t, collection)
    if !node.free #This means it won't matter what the X0 was if the node was frozen!
        push!(collection, (;Xt = node.node_data, X1anchor = node.node_data, descendants = node.weight, cmasked = false, group = node.group, last_coalescence = current_t))
        return
    end
    if node.time > target_t #<-If we're on the branch where a sample is needed
        Xt = bridge(P.P, Xs, node.node_data, current_t, target_t)
        push!(collection, (;Xt, X1anchor = node.node_data, descendants = node.weight, cmasked = true, group = node.group, last_coalescence = current_t))
    else
        next_Xs = bridge(P.P, Xs, node.node_data, current_t, node.time)
        for child in node.children
            tree_bridge(P, child, next_Xs, target_t, node.time, collection)
        end
    end
end

#Runs a single bridge for each tree in the forest, when X1 is a (tuple of) tensor state(s), and aggregates the results across the trees in the forest
#Needs to have "flowable" allow "nothing"
function forest_bridge(P::CoalescentFlow, X0sampler, X1, t, groups, flowable; maxlen = Inf, coalescence_factor = 1.0, merger = canonical_anchor_merge, coalescence_policy = P.coalescence_policy)
    elements = element.((X1,), 1:length(groups))
    forest, coal_times = sample_forest(P, elements; groupings = groups, flowable, coalescence_factor, merger, coalescence_policy)
    if (length(forest) + sum(coal_times .< t)) > maxlen #This resamples if you wind up greater than the max length.
        print("!")
        return forest_bridge(P, X0sampler, X1, t, groups, flowable, maxlen = maxlen, coalescence_factor = coalescence_factor, merger = merger, coalescence_policy = coalescence_policy)
    end
    collection = []
    for root in forest
        tree_bridge(P, root, X0sampler(root), t, 0.0f0, collection);
    end
    return collection
end


#Takes a vector of (tuples of) tensor states, runs the bridges for each, and re-batches them and their anchors.
#Assumes masked states for now.
#function branching_bridge(P::CoalescentFlow, X0sampler, X1s::Vector{BranchingState}; t_sample = ()->rand(Float32))
"""
    branching_bridge(P::CoalescentFlow, X0sampler, X1s, times; maxlen = Inf, coalescence_factor = 1.0)

When coalescence_factor is 1.0, all groups will collapse down to one node. When it is 0, no coalescences will occur.    
"""
function branching_bridge(P::CoalescentFlow, X0sampler, X1s, times; maxlen = Inf, coalescence_factor = 1.0, merger = canonical_anchor_merge, coalescence_policy = P.coalescence_policy)
    T = eltype(times)
    #To do: make this work (or check that it works) when X1.state is not masked.
    #Even better, build the mask into the BranchingState directly, so you don't need to duplicate them. Might require extra piping.
    batch_bridge = [forest_bridge(P, X0sampler, X1.state, times[i], X1.groupings, X1.state[1].cmask; maxlen, coalescence_factor, merger, coalescence_policy) for (i, X1) in enumerate(X1s)]
    maxlen = maximum(length.(batch_bridge))
    b = length(X1s)
    cmask = ones(Bool, maxlen, b)
    for b in 1:length(batch_bridge)
        for i in 1:length(batch_bridge[b])
            cmask[i,b] = batch_bridge[b][i].cmasked
        end
    end
    descendants = zeros(Int, maxlen, b)
    for b in 1:length(batch_bridge)
        for i in 1:length(batch_bridge[b])
            descendants[i,b] = batch_bridge[b][i].descendants
        end
    end
    padmask = zeros(Bool, maxlen, b)
    for b in 1:length(batch_bridge)
        for i in 1:length(batch_bridge[b])
            padmask[i,b] = true
        end
    end
    prev_coalescence = zeros(T, maxlen, b)
    for b in 1:length(batch_bridge)
        for i in 1:length(batch_bridge[b])
            prev_coalescence[i,b] = batch_bridge[b][i].last_coalescence
        end
    end
    groups = zeros(Int, maxlen, b)
    for b in 1:length(batch_bridge)
        for i in 1:length(batch_bridge[b])
            groups[i,b] = batch_bridge[b][i].group
        end
    end
    splits_target = split_target.((P,), times', clamp.(descendants .- 1, 0, Inf))
    Xt_batch = MaskedState.(regroup([[b.Xt for b in bridges] for bridges in batch_bridge]), (cmask,), (padmask .& cmask,));
    X1anchor_batch = MaskedState.(regroup([[b.X1anchor for b in bridges] for bridges in batch_bridge]), (cmask,), (padmask .& cmask,));
    return (;t = times, Xt = BranchingState(Xt_batch, groups), X1anchor = X1anchor_batch, descendants, splits_target, freemask = cmask, padmask, prev_coalescence)
end


#Complication: we must disallow splits for any discrete state that sampled a "change"!
#Also an issue where the model will frequently see input, durign inference, with EXACTLY matching elements post-split
#which will never happen during training. We could put A) some direct sampling mass at split events during training, so the model sees this? This is easy-ish to do!
#It almost feels like the bridge should be decomposed into 1) drift, 2) split, 3) diffuse? But this doesn't necessarily make sense for some kinds of process
#like zero noise processes, or discrete processes without a "diffusion" component.

println("!")

#MUST ALWAYS HAVE A BATCH DIM OF 1.
function Flowfusion.step(P::CoalescentFlow, XₜBS::BranchingState, hat::Tuple, s₁::Real, s₂::Real)
    Xₜ = XₜBS.state
    time_remaining = (1-s₁)
    delta_t = s₂ - s₁
    X1targets, event_lambdas = hat
    Xₜ = Flowfusion.step(P.P, Xₜ, X1targets, s₁, s₂)
    #splits = rand.(Poisson.((delta_t*P.split_transform.(event_lambdas))/time_remaining))[:]
    splits = rand.(Poisson.((delta_t*P.split_transform.(event_lambdas) * pdf(Truncated(P.branch_time_dist, s₁, 1), s₁))))[:]
    
    #Zero out split events for any non-continuous states that have changed.
    for s in 1:length(XₜBS.state)
        if (XₜBS.state[s] isa DiscreteState)
            for i in 1:length(splits)
                if element(tensor(Xₜ[s]),i,1) != element(tensor(XₜBS.state[s]),i,1)
                    println("Discrete component $s changed at $i")
                    splits[i] = 0
                end
            end
        end
    end
    current_length = size(tensor(first(Xₜ)))[end-1]
    new_length = current_length + sum(splits)
    element_tuple = element.(Xₜ, 1, 1)
    newstates = Tuple([Flowfusion.zerostate(element_tuple[i],new_length,1) for i in 1:length(element_tuple)])
    newgroupings = similar(XₜBS.groupings, new_length, 1) .= 0

    if should_append_on_split(P.coalescence_policy)
        # Append-mode: copy existing sequence as-is, then append all splits at the end of their group's block
        # First pass: copy existing elements
        for i in 1:current_length
            for s in 1:length(element_tuple)
                element(tensor(newstates[s]),i,1) .= tensor(element(Xₜ[s],i,1))
            end
            newgroupings[i] = XₜBS.groupings[i,1]
        end
        # Build group order and compute append offsets per group
        groups = unique(XₜBS.groupings[1:current_length,1])
        group_to_count = Dict{eltype(XₜBS.groupings),Int}()
        for i in 1:current_length
            g = XₜBS.groupings[i,1]
            group_to_count[g] = get(group_to_count, g, 0) + 1
        end
        group_to_offset = Dict{eltype(XₜBS.groupings),Int}()
        offset = 1
        for g in sort(collect(keys(group_to_count)))
            group_to_offset[g] = offset + group_to_count[g]
            offset += 0
        end
        # Track per-group next free slot (starts at end of each group's block)
        next_slot = copy(group_to_offset)
        # Append splits for each original position i at that group's tail
        for i in 1:current_length
            g = XₜBS.groupings[i,1]
            for k in 1:splits[i]
                pos = next_slot[g]
                for s in 1:length(element_tuple)
                    element(tensor(newstates[s]),pos,1) .= tensor(element(Xₜ[s],i,1))
                end
                newgroupings[pos] = g
                next_slot[g] = pos + 1
            end
        end
    else
        # Default adjacent insertion
        current_index = 1
        for i in 1:current_length
            for s in 1:length(element_tuple)
                element(tensor(newstates[s]),current_index,1) .= tensor(element(Xₜ[s],i,1))
                newgroupings[current_index] = XₜBS.groupings[i,1]
            end
            current_index += 1
            for j in 1:splits[i]
                for s in 1:length(element_tuple)
                    element(tensor(newstates[s]),current_index,1) .= tensor(element(Xₜ[s],i,1))
                    newgroupings[current_index] = XₜBS.groupings[i,1]
                end
                current_index += 1
            end
        end
    end
    return BranchingState(newstates, newgroupings)
end
