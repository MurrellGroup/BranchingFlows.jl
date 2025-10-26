module BranchingFlows

using ForwardBackward, Flowfusion, Manifolds, Distributions, LogExpFunctions, StatsBase
using Flowfusion: element

#include("ff.jl")
include("merging.jl")
include("states.jl")
include("trees.jl")
include("coalescent_flow.jl")
include("loss.jl")

export CoalescentFlow, branching_bridge, BranchingState
export CoalescencePolicy, SequentialCoalescencePolicy, NonSequentialCoalescencePolicy
export SequentialUniform, WeightedPairs, sequential_pairs, all_intragroup_pairs
export distance_weighted_coalescence, BalancedSequential, CorrelatedSequential
export last_to_nearest_coalescence, LastToNearest

end