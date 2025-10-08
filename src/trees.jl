#Tree types and algorithms
#Must handle interoperability with the States from Flowfusion, with:
# - state -> tree
# - tree -> state
#Scaffold the anchors while constructing the tree.
#Should take a process type, and inherit the anchor behavior from that, with canonical anchoring for each process.
#Need to decide how to handle cmask. cmasked tokens do not split/die, and they're present at t=0?
#Notes:
# - I guess they block merges, because merges are adjacent? This makes sense though.
#Need to optionally specify "groupings" that only get merged withing groups but never between groups.
#Needs to take in a bifurcation event time density function



mutable struct FlowNode{T, D}
    parent::Union{FlowNode,Nothing}
    children::Array{FlowNode,1}
    time::T
    node_data::D
    weight::Int
    group::Int
    free::Bool
    del::Bool #Whether this element will be deleted by t=1
end

function Base.show(io::IO, z::FlowNode)
    println(io, "FlowNode")
    println(io, "time: $(z.time)\nweight: $(z.weight)\ngroup: $(z.group)\nfree: $(z.free)")
end

FlowNode(time::T, node_data::D) where {T,D} = FlowNode(nothing, FlowNode[], time, node_data, 1, 0, true, false)
FlowNode(time::T, node_data::D, weight, group, free, del) where {T,D} = FlowNode(nothing, FlowNode[], time, node_data, weight, group, free, del)

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



