using BranchingFlows
using Test
using Flowfusion
using ForwardBackward
using Random

@testset "BranchingFlows.jl" begin
    @testset "Flowception bridge" begin
        target = FlowceptionState(
            MaskedState(ContinuousState(reshape(Float32.(1:4), 1, :)), trues(4), trues(4)),
            ones(Int, 4);
            branchmask = trues(4),
            flowmask = trues(4),
            padmask = trues(4),
        )
        P = FlowceptionFlow((Deterministic(),), () -> ContinuousState(zeros(Float32, 1, 1)))
        ts = flowception_bridge(P, [target], [0.25f0]; nstart = 4)

        @test size(ts.Xt.local_t, 2) == 1
        @test sum(ts.Xt.padmask[:, 1]) == 4
        @test all(ts.Xt.local_t[ts.Xt.padmask[:, 1], 1] .≈ 0.25f0)
        @test all(ts.insertions_target[ts.Xt.padmask[:, 1], 1] .== 0)
        @test all(ts.Xt.flowmask[ts.Xt.padmask[:, 1], 1])
    end

    @testset "Flowception step" begin
        base_state = FlowceptionState(
            (MaskedState(ContinuousState(zeros(Float32, 1, 2, 1)), trues(2, 1), trues(2, 1)),),
            reshape([1, 1], 2, 1);
            local_t = zeros(Float32, 2, 1),
            branchmask = trues(2, 1),
            flowmask = trues(2, 1),
            padmask = trues(2, 1),
        )

        P_noins = FlowceptionFlow((Deterministic(),), () -> ContinuousState(fill(-1f0, 1, 1)))
        hat_noins = ((ContinuousState(ones(Float32, 1, 2, 1)),), fill(-100f0, 2, 1))
        next_noins = Flowfusion.step(P_noins, base_state, hat_noins, 0f0, 0.25f0)

        @test size(ForwardBackward.tensor(next_noins.state[1]), 2) == 2
        @test all(next_noins.local_t .≈ 0.25f0)
        @test all(next_noins.flowmask)

        P_ins = FlowceptionFlow(
            (Deterministic(),),
            () -> ContinuousState(fill(-1f0, 1, 1));
            insertion_transform = identity,
        )
        hat_ins = ((ContinuousState(ones(Float32, 1, 2, 1)),), fill(25f0, 2, 1))
        next_ins = Flowfusion.step(P_ins, base_state, hat_ins, 0f0, 0.5f0)

        @test size(ForwardBackward.tensor(next_ins.state[1]), 2) > 2
        @test any(next_ins.local_t .== 0f0)
        @test any(ForwardBackward.tensor(next_ins.state[1]) .== -1f0)
    end

    @testset "Flowception total window" begin
        target = FlowceptionState(
            MaskedState(ContinuousState(reshape(Float32.(1:4), 1, :)), trues(4), trues(4)),
            ones(Int, 4);
            branchmask = trues(4),
            flowmask = trues(4),
            padmask = trues(4),
        )

        P = FlowceptionFlow((Deterministic(),), () -> ContinuousState(zeros(Float32, 1, 1)); total_time = 4f0)
        @test BranchingFlows.flowception_total_time(P, Float32) == 4f0
        @test BranchingFlows.flowception_insertion_horizon(P, Float32) == 3f0
        @test BranchingFlows.scheduler_hazard(P, 0f0) ≈ (1f0 / 3f0)
        @test BranchingFlows.scheduler_hazard(P, 2f0) ≈ 1f0
        @test BranchingFlows.scheduler_hazard(P, 3f0) == 0f0
        @test scalefloss(P, 1.5f0, 1, 0.2f0) ≈ (1f0 / 0.7f0)

        ts = flowception_bridge(P, [target], [10f0]; nstart = 4)
        @test ts.t == Float32[4f0]
        @test all(ts.Xt.local_t[ts.Xt.padmask[:, 1], 1] .== 1f0)

        late_state = FlowceptionState(
            (MaskedState(ContinuousState(zeros(Float32, 1, 1, 1)), trues(1, 1), trues(1, 1)),),
            ones(Int, 1, 1);
            local_t = zeros(Float32, 1, 1),
            branchmask = trues(1, 1),
            flowmask = trues(1, 1),
            padmask = trues(1, 1),
        )
        hat = ((ContinuousState(ones(Float32, 1, 1, 1)),), fill(50f0, 1, 1))
        late_step = Flowfusion.step(P, late_state, hat, 3.2f0, 3.8f0)
        @test size(ForwardBackward.tensor(late_step.state[1]), 2) == 1

        Pdir = DirectionalFlowceptionFlow((Deterministic(),), () -> ContinuousState(zeros(Float32, 1, 1)); total_time = 10f0)
        @test BranchingFlows.flowception_insertion_horizon(Pdir, Float32) == 9f0
        @test scalefloss(Pdir, 4.5f0, 1, 0.2f0) ≈ (1f0 / 0.7f0)
        tsdir = directional_flowception_bridge(Pdir, [target], [20f0]; nstart = 4)
        @test tsdir.t == Float32[10f0]
        @test all(tsdir.Xt.local_t[tsdir.Xt.padmask[:, 1], 1] .== 1f0)
    end

    @testset "Directional Flowception targets and pooling" begin
        groups = [1, 1, 1, 2, 2, 2]
        visible = Bool[true, false, true, true, false, true]
        padmask = trues(6)
        visible_inds, left_counts, right_counts = BranchingFlows.directional_slot_targets(groups, visible, padmask)

        @test visible_inds == [1, 3, 4, 6]
        @test left_counts == [0, 1, 0, 1]
        @test right_counts == [1, 0, 1, 0]

        Xt = FlowceptionState(
            (MaskedState(ContinuousState(zeros(Float32, 1, 3, 1)), trues(3, 1), trues(3, 1)),),
            reshape([1, 1, 2], 3, 1);
            local_t = zeros(Float32, 3, 1),
            branchmask = trues(3, 1),
            flowmask = trues(3, 1),
            padmask = trues(3, 1),
        )

        P = DirectionalFlowceptionFlow(
            (Deterministic(),),
            () -> ContinuousState(fill(-1f0, 1, 1));
            insertion_transform = identity,
        )

        hat_insertions = cat(
            reshape(Float32[10, 14, 20], 1, 3, 1),
            reshape(Float32[6, 8, 9], 1, 3, 1);
            dims = 1,
        )
        target_insertions = cat(
            reshape(Float32[2, 7, 5], 1, 3, 1),
            reshape(Float32[7, 11, 13], 1, 3, 1);
            dims = 1,
        )

        before_rates, after_rates, _ = BranchingFlows.directional_physical_rates(P, hat_insertions, Xt)
        @test vec(before_rates) == Float32[10, 0, 20]
        @test vec(after_rates) == Float32[10, 8, 9]

        expected = sum(Float32[
            BranchingFlows.sbpl(10f0, 2f0),
            BranchingFlows.sbpl(10f0, 7f0),
            BranchingFlows.sbpl(8f0, 11f0),
            BranchingFlows.sbpl(20f0, 5f0),
            BranchingFlows.sbpl(9f0, 13f0),
        ]) / 5f0
        @test floss(P, hat_insertions, target_insertions, Xt, 1f0) ≈ expected
    end

    @testset "Directional Flowception bridge" begin
        target = FlowceptionState(
            MaskedState(ContinuousState(reshape(Float32.(1:4), 1, :)), trues(4), trues(4)),
            ones(Int, 4);
            branchmask = trues(4),
            flowmask = trues(4),
            padmask = trues(4),
        )
        P = DirectionalFlowceptionFlow((Deterministic(),), () -> ContinuousState(zeros(Float32, 1, 1)))
        ts = directional_flowception_bridge(P, [target], [0.25f0]; nstart = 4)

        @test size(ts.insertions_target) == (2, 4, 1)
        @test all(ts.insertions_target[:, ts.Xt.padmask[:, 1], 1] .== 0)
        @test all(ts.Xt.local_t[ts.Xt.padmask[:, 1], 1] .≈ 0.25f0)
    end

    @testset "Flowception discrete local-time convergence" begin
        Random.seed!(1)

        P = FlowceptionFlow(
            Flowfusion.DistNoisyInterpolatingDiscreteFlow(),
            () -> DiscreteState(3, fill(3, 1, 1)),
        )

        function make_disc_state(local_ts)
            n = length(local_ts)
            mask = trues(n, 1)
            disc = DiscreteState(3, fill(3, n, 1))
            return FlowceptionState(MaskedState(disc, mask, mask), ones(Int, n, 1);
                local_t = reshape(Float32.(local_ts), n, 1),
                branchmask = falses(n, 1),
                flowmask = trues(n, 1),
                padmask = mask,
            )
        end

        function predictor(t, Xt)
            n = size(Xt.local_t, 1)
            logits = fill(-80f0, 3, n, 1)
            logits[1, :, :] .= 80f0
            insertion_logits = fill(-80f0, n, 1)
            return logits, insertion_logits
        end

        local_ts = Float32[0.0, 0.2, 0.5, 0.8, 0.95, 0.99]
        Xf = gen(P, make_disc_state(local_ts), predictor, 0f0:0.01f0:2f0)

        @test all(vec(Xf.state[1].S.state) .== 1)
        @test all(Xf.local_t .≈ 1f0)
    end
end
