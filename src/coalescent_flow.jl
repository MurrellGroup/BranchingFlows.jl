
#This is the good way to get the elements.
#Need to restrict on lmask though:
#@time a = element.((compound_state,), 1:seqlength(compound_state), 1);

struct CoalescentFlow{Proc,D,F} <: Process
    P::Proc
    branch_time_dist::D
    split_transform::F
end
CoalescentFlow(P, branch_time_dist) = CoalescentFlow(P, branch_time_dist, x -> exp.(clamp.(x, -100, 11)))

#We could move the lmask and cmask to the outer state wrapper, but that would mean the individual losses wouldn't dispatch correctly.
#let's try this for now.

#I think we need to add lmask and cmask to the BranchingState, and just use clean underlying states. Will need to pipe losses etc to use the outer masks.
#Otherwise the user has to do too much extra BS.
struct BranchingState{A,B} <: State
    state::A     #Flow state, or tuple of flow states
    groupings::B 
end

Base.copy(Xₜ::BranchingState) = deepcopy(Xₜ)


Flowfusion.resolveprediction(a, Xₜ::BranchingState) = a
#Flowfusion.mask(a, X₀::BranchingFlows.BranchingState) = a


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
function sample_forest(P::CoalescentFlow, elements::AbstractVector; groupings = zeros(Int, length(elements)), flowable = ones(Bool, length(elements)), T = Float32)
    nodes = FlowNode[FlowNode(T(1), elements[i], 1, groupings[i], flowable[i]) for i = 1:length(elements)]
    coal_times = T.(sort(rand(P.branch_time_dist, sum(possible_merges(nodes))), rev = true)) #Because coal stuff walks backwards from 1
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
#JUST ADDED last_coalescence = current_t which needs to prop through and get returned to the user in a nice shape to use for masking recent coal events

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
#function branching_bridge(P::CoalescentFlow, X0sampler, X1s::Vector{BranchingState}; t_sample = ()->rand(Float32))
function branching_bridge(P::CoalescentFlow, X0sampler, X1s; t_sample = ()->rand(Float32), T = Float32)
    times = [T(t_sample()) for _ in 1:length(X1s)]
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


#=
#Forces the lmask and cmask to be fixed for now:
function Flowfusion.step(P::CoalescentFlow, XₜBS::BranchingState, hat, s₁, s₂)
    Xₜ = XₜBS.state
    time_remaining = (1-s₁)
    delta_t = s₂ - s₁
    X1targets, event_lambdas = hat
    
    Xₜ = bridge(P.P, Xₜ, X1targets, s₁, s₂) #This was bridge. Not sure what it should be tbh. Need to think about this in general.
    
    splits = rand.(Poisson.((delta_t*event_lambdas)/time_remaining))[:]
    
    current_length = size(tensor(Xₜ))[end-1]
    new_length = current_length + sum(splits)
    element_tuple = element.(Xₜ, 1, 1)
    newstates = [zerostate(element_tuple[i],new_length,1) for i in 1:length(element_tuple)]
    newgroupings = similar(XₜBS.groupings, new_length, 1) .= 0
    current_index = 1
    for i in 1:current_length
        for s in 1:length(element_tuple)
            element(newstates[s],current_index,1) .= element(Xₜ,i,s)
            newgroupings[current_index] = XₜBS.groupings[i,1]
        end
        current_index += 1
        for j in 1:splits[i]
            for s in 1:length(element_tuple)
                element(newstates[s],current_index,1) .= element(Xₜ,i,s)
                newgroupings[current_index] = XₜBS.groupings[i,1]
            end
            current_index += 1
        end
    end
    return BranchingState(newstates, newgroupings)
end
=#

#Complication: we must disallow splits for any discrete state that sampled a "change"!
#Also an issue where the model will frequently see input, durign inference, with EXACTLY matching elements post-split
#which will never happen during training. We could put A) some direct sampling mass at split events during training, so the model sees this? This is easy-ish to do!
#It almost feels like the bridge should be decomposed into 1) drift, 2) split, 3) diffuse? But this doesn't necessarily make sense for some kinds of process
#like zero noise processes, or discrete processes without a "diffusion" component.

function Flowfusion.step(P::CoalescentFlow, XₜBS::BranchingState, hat::Tuple, s₁::Real, s₂::Real)
    Xₜ = XₜBS.state
    time_remaining = (1-s₁)
    delta_t = s₂ - s₁
    X1targets, event_lambdas = hat
    Xₜ = Flowfusion.step(P.P, Xₜ, X1targets, s₁, s₂)
    splits = rand.(Poisson.((delta_t*P.split_transform.(event_lambdas))/time_remaining))[:]
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
    return BranchingState(newstates, newgroupings)
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


=#