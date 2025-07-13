module BranchingFlows

using ForwardBackward, Flowfusion, Manifolds, Distributions, LogExpFunctions
using Flowfusion: element

#include("ff.jl")
include("states.jl")
include("trees.jl")
include("coalescent_flow.jl")
include("loss.jl")

export CoalescentFlow, branching_bridge, BranchingState

end