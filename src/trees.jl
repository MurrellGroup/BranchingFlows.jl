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
end

function Base.show(io::IO, z::FlowNode)
    println(io, "FlowNode")
    println(io, "time: $(z.time)\nweight: $(z.weight)\ngroup: $(z.group)\nfree: $(z.free)")
end

FlowNode(time::T, node_data::D) where {T,D} = FlowNode(nothing, FlowNode[], time, node_data, 1, 0, true)
FlowNode(time::T, node_data::D, weight, group, free) where {T,D} = FlowNode(nothing, FlowNode[], time, node_data, weight, group, free)

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




#This needs to sample the tree, and the anchors
#States needs to be a tuple of all the states, where the elements match up.
#This isn't natively handled in Flowfusion because we usually don't care
#Either do it by assumption, or by constructing an explicit pairing.
#What do we need at a minimum? Which dims are the batch dims, maybe?
#Can a manifold state have multiple manifold dimensions? Like if we have a single element that is two points on the sphere?

#=
seqlength(S::Flowfusion.UState) = size(tensor(S))[end-1]
function seqlength(S::Tuple{Vararg{Flowfusion.UState}})
    lens = map(seqlength, S)
    if all(lens .== lens[1])
        return lens[1]
    else
        throw(ArgumentError("All states must have the same sequence length."))
    end
end
=#








#=
P = CoalescentFlow((BrownianMotion(0.1f0), ManifoldProcess(0.1f0)), Uniform(0.0f0, 1.0f0))




L = 120
b = 2
locs = 
rots = rand(Float32, 3, 3, L, b)
aas = rand(1:21, L, b)

cmask = trues(L, b)
cmask[5:7,2] .= false
padmask = trues(L, b)
padmask[5:end,2] .= false

rotM = Rotations(3)

function X1_sample(n,b) = MaskedState(ContinuousState(randn(Float32, 3, 1, L, b)), cmask, padmask)
X1rots = MaskedState(ManifoldState(rotM, reshape(Array{Float32}.(Flowfusion.rand(rotM, L*b)), L, b)), cmask, padmask)





compound_state = (X1locs, X1rots);

@time elements = element.((compound_state,), 1:seqlength(compound_state), 1);
P = CoalescentFlow(BrownianMotion(), Uniform(0.0, 1.0))

groups = zeros(Int, length(elements))
groups[500:end] .= 1
flowable = trues(length(elements))
flowable[750] = false
@time forest = sample_forest(P, elements, groupings = groups, flowable = flowable);

#0....0.2....t....0.4...



X0sampler() = (ContinuousState(randn(Float32, 3, 1)), ManifoldState(rotM, reshape(Array{Float32}.(Flowfusion.rand(rotM, 1)), 1)))


collection = []
@time tree_bridge(P, forest[1], X0sampler(), 0.4f0, 0.0f0, collection);


node = forest[1]
next_Xs = bridge(P.P, X0sampler(), node.node_data, 0.0, node.time)

tree_bridge(P.P, node, next_Xs, 0.4f0, node.time, collection)


bridge(P::Tuple{Vararg{UProcess}}, X0::Tuple{Vararg{UState}}, X1::Tuple, t0, t) = bridge.(P, X0, X1, (t0,), (t, ))



n1 = FlowNode(1.0, Dict())
n2 = FlowNode(2.0, Dict())

mergenodes(n1, n2, 3.0, Dict())

eltype(n1)((3.0, Dict())...)

=#