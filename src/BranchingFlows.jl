module BranchingFlows

using ForwardBackward, Flowfusion, Manifolds, Distributions, LogExpFunctions, StatsBase, Adapt
using Flowfusion: element

#include("ff.jl")
include("merging.jl")
include("states.jl")
include("trees.jl")
include("coalescent_flow.jl")
include("flowception.jl")
include("loss.jl")

export CoalescentFlow,
       branching_bridge,
       BranchingState,
       SequentialUniform,
       uniform_del_insertions,
       FlowceptionFlow,
       FlowceptionState,
       flowception_bridge,
       linear_scheduler,
       linear_scheduler_derivative,
       linear_scheduler_inverse

end
