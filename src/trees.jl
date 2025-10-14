mutable struct FlowNode{T, D}
    parent::Union{FlowNode,Nothing}
    children::Array{FlowNode,1}
    time::T
    node_data::D
    weight::Int
    group::Int
    free::Bool
    del::Bool #Whether this element will be deleted by t=1
    id::Int #If you need to track nodes through to t=0 for custom X0sampler tricks. Note: merged nodes get an id of 0.
end

function Base.show(io::IO, z::FlowNode)
    println(io, "FlowNode")
    println(io, "time: $(z.time)\nweight: $(z.weight)\ngroup: $(z.group)\nfree: $(z.free)\ndel: $(z.del)\nid: $(z.id)")
end

FlowNode(time::T, node_data::D) where {T,D} = FlowNode(nothing, FlowNode[], time, node_data, 1, 0, true, false, 1)
FlowNode(time::T, node_data::D, weight, group, free, del, id) where {T,D} = FlowNode(nothing, FlowNode[], time, node_data, weight, group, free, del, id)

function addchild!(parent::FlowNode, child::FlowNode)
    if child.parent === nothing
        push!(parent.children, child)
        child.parent = parent
    else
        throw(ArgumentError("You are trying to add a child to a parent when the child already has a parent."))
    end
end

function mergenodes(n1::FlowNode, n2::FlowNode, args...)
    newnode = FlowNode(args...)
    if n1.parent === nothing && n2.parent === nothing
        addchild!(newnode, n1)
        addchild!(newnode, n2)
    else
        throw(ArgumentError("You are trying to merge two nodes that already have parents."))
    end
    return newnode
end



