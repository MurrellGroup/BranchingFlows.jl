using BranchingFlows
using Test
using Random

function synthetic_branching_state(n::Int)
    coords = reshape(Float32.(1:(3n)), 3, n)
    continuous = BranchingFlows.Flowfusion.MaskedState(BranchingFlows.Flowfusion.ContinuousState(coords), trues(n), trues(n))
    discrete = BranchingFlows.Flowfusion.MaskedState(BranchingFlows.Flowfusion.DiscreteState(5, fill(1, n)), trues(n), trues(n))
    return BranchingState((continuous, discrete), ones(Int, n))
end

function synthetic_x0_sampler(root)
    return (
        BranchingFlows.Flowfusion.ContinuousState(zeros(Float32, 3, 1)),
        BranchingFlows.Flowfusion.DiscreteState(5, [5]),
    )
end

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

    @testset "deletion_pad semantics" begin
        P = CoalescentFlow(
            (BranchingFlows.Flowfusion.BrownianMotion(0.05f0), BranchingFlows.Flowfusion.DistNoisyInterpolatingDiscreteFlow()),
            BranchingFlows.Distributions.Beta(1, 2),
        )
        X1 = [synthetic_branching_state(3)]

        Random.seed!(11)
        no_pad = branching_bridge(
            P,
            synthetic_x0_sampler,
            X1,
            [0.0f0];
            coalescence_factor = 0.0,
            length_mins = 10,
            deletion_pad = 0.0,
        )
        @test size(no_pad.Xt.padmask, 1) == 3
        @test sum(no_pad.del) == 0

        Random.seed!(11)
        floor_pad = branching_bridge(
            P,
            synthetic_x0_sampler,
            X1,
            [0.0f0];
            coalescence_factor = 0.0,
            length_mins = 10,
            deletion_pad = 1.0,
        )
        @test size(floor_pad.Xt.padmask, 1) >= 10
        @test sum(floor_pad.del) >= 7

        Random.seed!(11)
        extra_pad = branching_bridge(
            P,
            synthetic_x0_sampler,
            X1,
            [0.0f0];
            coalescence_factor = 0.0,
            length_mins = 10,
            deletion_pad = 2.0,
        )
        @test size(extra_pad.Xt.padmask, 1) >= size(floor_pad.Xt.padmask, 1)
        @test size(extra_pad.Xt.padmask, 1) > 10
    end
end
