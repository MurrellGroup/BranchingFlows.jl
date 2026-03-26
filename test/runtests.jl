using BranchingFlows
using Test
using Random

@testset "BranchingFlows.jl" begin
    nodes = BranchingFlows.FlowNode[
        BranchingFlows.FlowNode(1.0, 0.0, 1, 1, true, false, 1, true),
        BranchingFlows.FlowNode(1.0, 1.0, 2, 1, true, false, 2, true),
        BranchingFlows.FlowNode(1.0, 3.0, 1, 1, true, false, 3, true),
        BranchingFlows.FlowNode(1.0, 6.0, 4, 1, true, false, 4, true),
    ]

    @testset "RichGetRicherSequential" begin
        Random.seed!(7)
        pair = BranchingFlows.select_coalescence(RichGetRicherSequential(alpha = 1.0), nodes, nothing)
        @test pair in ((1, 2), (2, 3), (3, 4))
    end

    @testset "SequentialProximity" begin
        prox = SequentialProximity(extractor = identity, distance = (x, y) -> abs(x - y), tie_rtol = 0.1)
        Random.seed!(7)
        @test BranchingFlows.select_coalescence(prox, nodes, nothing) == (1, 2)
    end

    @testset "SequentialDeepLineage" begin
        deep_nodes = BranchingFlows.FlowNode[
            BranchingFlows.FlowNode(1.0, 0.0, 1, 1, true, false, 1, true),
            BranchingFlows.FlowNode(1.0, 1.0, 1, 1, true, false, 2, true),
            BranchingFlows.FlowNode(1.0, 3.0, 1, 1, true, false, 3, true),
            BranchingFlows.FlowNode(1.0, 6.0, 2, 1, true, false, 4, true),
        ]
        deep = SequentialDeepLineage(min_count = 2, trunk_target_sampler = () -> 1)
        BranchingFlows.init!(deep, deep_nodes)
        Random.seed!(7)
        @test BranchingFlows.select_coalescence(deep, deep_nodes, nothing) == (3, 4)
    end
end
