
#This is the good way to get the elements.
#Need to restrict on lmask though:
#@time a = element.((compound_state,), 1:seqlength(compound_state), 1);

struct CoalescentFlow{Proc,D} <: Process
    P::Proc
    branch_time_dist::D
end

split_target(P::CoalescentFlow, t, splits) = splits == 0 ? oftype(t, 0.0) : oftype(t, splits * pdf(Truncated(P.branch_time_dist, t, 1), t) * (1-t)) #For splits-as-rate, drop this last (1-t) term

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
function sample_forest(P::CoalescentFlow, elements::AbstractVector; groupings = zeros(Int, length(elements)), flowable = ones(Bool, length(elements)))
    nodes = FlowNode[FlowNode(1.0, elements[i], 1, groupings[i], flowable[i]) for i = 1:length(elements)]
    coal_times = sort(rand(P.branch_time_dist, sum(possible_merges(nodes))), rev = true) #Because coal stuff walks backwards from 1
    for time in coal_times
        ind = rand(findall(possible_merges(nodes)))
        left, right = nodes[ind], nodes[ind+1]
        @assert left.group == right.group
        @assert left.free && right.free
        merged = mergenodes(left, right, time, canonical_anchor_merge(left.node_data, right.node_data, left.weight, right.weight), left.weight + right.weight, left.group, true)
        nodes[ind] = merged
        deleteat!(nodes, ind+1)
    end
    return nodes
end

#This takes an ordered vector of states and runs the bridge
function tree_bridge(P::CoalescentFlow, node, Xs, target_t, current_t, collection)
    if !node.free #This means it won't matter what the X0 was if the node was frozen!
        push!(collection, (;Xt = node.node_data, X1anchor = node.node_data, descendants = node.weight, cmasked = false, group = node.group))
        return
    end
    if node.time > target_t #<-If we're on the branch where a sample is needed
        Xt = bridge(P.P, Xs, node.node_data, current_t, target_t)
        push!(collection, (;Xt, X1anchor = node.node_data, descendants = node.weight, cmasked = true, group = node.group))
    else
        next_Xs = bridge(P.P, Xs, node.node_data, current_t, node.time)
        for child in node.children
            tree_bridge(P, child, next_Xs, target_t, node.time, collection)
        end
    end
end

#Runs a single bridge for each tree in the forest, when X1 is a (tuple of) tensor state(s), and aggregates the results across the trees in the forest
#Needs to have "flowable" allow "nothing"
function forest_bridge(P::CoalescentFlow, X0sampler, X1, t, groups, flowable)
    elements = element.((X1,), 1:length(groups))
    forest = sample_forest(P, elements, groupings = groups, flowable = flowable)
    collection = []
    for root in forest
        tree_bridge(P, root, X0sampler(root), t, 0.0f0, collection);
    end
    return collection
end


#Takes a vector of (tuples of) tensor states, runs the bridges for each, and re-batches them and their anchors.
#Assumes masked states for now.
function branching_bridge(P::CoalescentFlow, X0sampler, X1s; t_sample = ()->rand(Float32))
    times = [t_sample() for _ in 1:length(X1s)]
    batch_bridge = [forest_bridge(P, X0sampler, X1.state, times[i], X1.groupings, X1.state[1].cmask) for (i, X1) in enumerate(X1s)]
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
    groups = zeros(Int, maxlen, b)
    for b in 1:length(batch_bridge)
        for i in 1:length(batch_bridge[b])
            groups[i,b] = batch_bridge[b][i].group
        end
    end
    splits_target = split_target.((P,), times', clamp.(descendants .- 1, 0, Inf))
    Xt_batch = MaskedState.(regroup([[b.Xt for b in bridges] for bridges in batch_bridge]), (cmask,), (padmask .& cmask,));
    X1anchor_batch = MaskedState.(regroup([[b.X1anchor for b in bridges] for bridges in batch_bridge]), (cmask,), (padmask .& cmask,));
    return (;t = times, Xt = Xt_batch, X1anchor = X1anchor_batch, descendants, splits_target, freemask = cmask, padmask, groups)
end


#=
function Flowfusion.step(P::CoalescentFlow, Xₜ, hat, s₁, s₂; hook = nothing)
    time_remaining = (1-s₁)
    delta_t = s₂ - s₁
    X1targets, event_lambdas = hat
    Xₜ = bridge(P.P, Xₜ, X1targets, s₁, s₂)
    splits = rand.(Poisson.((delta_t*event_lambdas)/time_remaining))[:]
    !isnothing(splits_hook) && splits_hook(splits)
    current_length = size(tensor(Xₜ))[end-1]
    new_length = current_length + sum(splits)
    element_tuple = element.(Xₜ, 1, 1)
    newstates = [zerostate(element_tuple[i],new_length,1) for i in 1:length(element_tuple)]
    current_index = 1
    for i in 1:current_length
        for s in 1:length(element_tuple)
            element(newstates[s],current_index,1) .= element(Xₜ,i,s)
        end
        current_index += 1
        for j in 1:splits[i]
            for s in 1:length(element_tuple)
                element(newstates[s],current_index,1) .= element(Xₜ,i,s)
            end
            current_index += 1
        end
    end
    return (newstates)
end


@eval Flowfusion begin
function gen(P::Tuple{Vararg{UProcess}}, X₀::Tuple{Vararg{UState}}, model, steps::AbstractVector; tracker::Function=Returns(nothing), midpoint = false, hook = nothing)
    Xₜ = copy.(X₀)
    for (s₁, s₂) in zip(steps, steps[begin+1:end])
        t = midpoint ? (s₁ + s₂) / 2 : t = s₁
        hat = resolveprediction(model(t, Xₜ), Xₜ)
        if isnothing(hook)
            Xₜ = mask(step(P, Xₜ, hat, s₁, s₂), X₀)
        else
            Xₜ = mask(step(P, Xₜ, hat, s₁, s₂, hook = hook), X₀)
        end
        tracker(t, Xₜ, hat)
    end
    return Xₜ
end
end

#We could move the lmask and cmask to the outer state wrapper, but that would mean the individual losses wouldn't dispatch correctly.
#let's try this for now.
struct BranchingState{A,B}
    S::A     #State
    groupings::B 
end
=#