module BranchingFlows

using ForwardBackward, Flowfusion, Manifolds, Distributions, LogExpFunctions, StatsBase
using Flowfusion: element

#include("ff.jl")
include("merging.jl")
include("states.jl")
include("trees.jl")
include("coalescent_flow.jl")
include("loss.jl")

export CoalescentFlow, branching_bridge, BranchingState, SequentialUniform, uniform_del_insertions

end