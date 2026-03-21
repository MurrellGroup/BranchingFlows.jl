using BranchingFlows
using Test
using Flowfusion
using ForwardBackward
using Random

@testset "BranchingFlows.jl" begin
    @testset "Component cmask invariants" begin
        branchmask = Bool[true, false]
        mixed_component = (
            MaskedState(ContinuousState(reshape(Float32[1, 2], 1, :)), trues(2), trues(2)),
            MaskedState(ContinuousState(reshape(Float32[3, 4], 1, :)), Bool[true, false], trues(2)),
        )

        @test_throws ErrorException BranchingState(
            mixed_component,
            ones(Int, 2);
            branchmask = trues(2),
            flowmask = trues(2),
            padmask = trues(2),
        )

        @test_throws ErrorException FlowceptionState(
            mixed_component,
            ones(Int, 2);
            branchmask = trues(2),
            flowmask = trues(2),
            padmask = trues(2),
        )

        plain_component = (ContinuousState(reshape(Float32[5, 6], 1, :)),)
        @test_throws ErrorException BranchingState(
            plain_component,
            ones(Int, 2);
            branchmask = trues(2),
            flowmask = Bool[false, true],
            padmask = trues(2),
        )

        @test_throws ErrorException FlowceptionState(
            plain_component,
            ones(Int, 2);
            branchmask = trues(2),
            flowmask = Bool[false, true],
            padmask = trues(2),
        )

        valid_branching = BranchingState(
            mixed_component,
            ones(Int, 2);
            branchmask,
            flowmask = trues(2),
            padmask = trues(2),
        )
        @test valid_branching.branchmask == branchmask
        @test vec(valid_branching.state[2].cmask) == Bool[true, false]
    end

    @testset "Branching bridge and step preserve component cmasks" begin
        target = BranchingState(
            (
                MaskedState(ContinuousState(reshape(Float32[1, 2], 1, :)), trues(2), trues(2)),
                MaskedState(ContinuousState(reshape(Float32[10, 20], 1, :)), Bool[true, false], Bool[true, false]),
            ),
            ones(Int, 2);
            branchmask = Bool[true, false],
            flowmask = trues(2),
            padmask = trues(2),
        )
        P = CoalescentFlow((Deterministic(), Deterministic()), BranchingFlows.Uniform(0f0, 1f0))
        X0sampler(_) = (
            ContinuousState(fill(-1f0, 1, 1)),
            ContinuousState(fill(-2f0, 1, 1)),
        )

        ts = branching_bridge(P, X0sampler, [target], [0.5f0]; coalescence_factor = 0.0)
        @test vec(ts.Xt.state[1].cmask[:, 1]) == Bool[true, true]
        @test vec(ts.Xt.state[2].cmask[:, 1]) == Bool[true, false]
        @test vec(ts.X1anchor[2].cmask[:, 1]) == Bool[true, false]
        @test vec(ts.Xt.branchmask[:, 1]) == Bool[true, false]

        plain_target = BranchingState(
            (ContinuousState(reshape(Float32[1, 2], 1, :)),),
            ones(Int, 2);
            branchmask = falses(2),
            flowmask = Bool[false, true],
            padmask = trues(2),
        )
        plain_ts = branching_bridge(
            CoalescentFlow((Deterministic(),), BranchingFlows.Uniform(0f0, 1f0)),
            _ -> (ContinuousState(fill(-1f0, 1, 1)),),
            [plain_target],
            [0.5f0];
            coalescence_factor = 0.0,
        )
        @test vec(plain_ts.Xt.state[1].cmask[:, 1]) == Bool[false, true]
        @test vec(plain_ts.X1anchor[1].lmask[:, 1]) == Bool[false, true]

        padded = fixedcount_del_insertions(target, 3)
        @test count(.!vec(padded.state[2].cmask)) == 1

        base_state = BranchingState(
            (
                MaskedState(ContinuousState(zeros(Float32, 1, 2, 1)), trues(2, 1), trues(2, 1)),
                MaskedState(ContinuousState(reshape(Float32[5, 6], 1, 2, 1)), reshape(Bool[true, false], 2, 1), trues(2, 1)),
            ),
            ones(Int, 2, 1);
            branchmask = reshape(Bool[true, false], 2, 1),
            flowmask = trues(2, 1),
            padmask = trues(2, 1),
        )
        hat = (
            (
                ContinuousState(fill(1f0, 1, 2, 1)),
                ContinuousState(fill(9f0, 1, 2, 1)),
            ),
            fill(-100f0, 2, 1),
            fill(-100f0, 2, 1),
        )
        next_state = Flowfusion.step(P, base_state, hat, 0f0, 0.25f0)
        @test next_state.state[2].cmask == base_state.state[2].cmask
        @test tensor(next_state.state[2].S)[1, 2, 1] == tensor(base_state.state[2].S)[1, 2, 1]
        @test tensor(next_state.state[2].S)[1, 1, 1] != tensor(base_state.state[2].S)[1, 1, 1]
        @test size(tensor(next_state.state[1]), 2) == 2

        plain_base_state = BranchingState(
            (ContinuousState(reshape(Float32[0, 10], 1, 2, 1)),),
            ones(Int, 2, 1);
            branchmask = falses(2, 1),
            flowmask = reshape(Bool[false, true], 2, 1),
            padmask = trues(2, 1),
        )
        plain_hat = (
            (ContinuousState(fill(5f0, 1, 2, 1)),),
            fill(-100f0, 2, 1),
            fill(-100f0, 2, 1),
        )
        plain_next_state = Flowfusion.step(CoalescentFlow((Deterministic(),), BranchingFlows.Uniform(0f0, 1f0)), plain_base_state, plain_hat, 0f0, 0.25f0)
        @test tensor(plain_next_state.state[1])[1, 1, 1] == 0f0
        @test tensor(plain_next_state.state[1])[1, 2, 1] != 10f0
    end

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

        plain_target = FlowceptionState(
            (ContinuousState(reshape(Float32[1, 2], 1, :)),),
            ones(Int, 2);
            branchmask = falses(2),
            flowmask = Bool[false, true],
            padmask = trues(2),
        )
        plain_ts = flowception_bridge(P, [plain_target], [0.25f0]; nstart = 2)
        @test vec(plain_ts.Xt.state[1].cmask[:, 1]) == Bool[false, true]
        @test vec(plain_ts.X1anchor[1].lmask[:, 1]) == Bool[false, true]
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

        plain_base_state = FlowceptionState(
            (ContinuousState(reshape(Float32[0, 10], 1, 2, 1)),),
            reshape([1, 1], 2, 1);
            local_t = zeros(Float32, 2, 1),
            branchmask = falses(2, 1),
            flowmask = reshape(Bool[false, true], 2, 1),
            padmask = trues(2, 1),
        )
        plain_hat = ((ContinuousState(fill(5f0, 1, 2, 1)),), fill(-100f0, 2, 1))
        plain_next = Flowfusion.step(P_noins, plain_base_state, plain_hat, 0f0, 0.25f0)
        @test ForwardBackward.tensor(plain_next.state[1])[1, 1, 1] == 0f0
        @test ForwardBackward.tensor(plain_next.state[1])[1, 2, 1] != 10f0
        @test vec(plain_next.local_t[:, 1]) == Float32[0f0, 0.25f0]
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

        plain_target = FlowceptionState(
            (ContinuousState(reshape(Float32[1, 2], 1, :)),),
            ones(Int, 2);
            branchmask = falses(2),
            flowmask = Bool[false, true],
            padmask = trues(2),
        )
        plain_ts = directional_flowception_bridge(P, [plain_target], [0.25f0]; nstart = 2)
        @test vec(plain_ts.Xt.state[1].cmask[:, 1]) == Bool[false, true]
        @test vec(plain_ts.X1anchor[1].lmask[:, 1]) == Bool[false, true]
    end

    @testset "Flowception component cmasks" begin
        birth_sampler(_) = (
            ContinuousState(fill(-1f0, 1, 1)),
            ContinuousState(fill(-2f0, 1, 1)),
        )
        target = FlowceptionState(
            (
                MaskedState(ContinuousState(reshape(Float32[1, 2], 1, :)), trues(2), trues(2)),
                MaskedState(ContinuousState(reshape(Float32[10, 20], 1, :)), Bool[true, false], Bool[true, false]),
            ),
            ones(Int, 2);
            branchmask = Bool[true, false],
            flowmask = trues(2),
            padmask = trues(2),
        )

        P = FlowceptionFlow((Deterministic(), Deterministic()), birth_sampler)
        ts = flowception_bridge(P, [target], [0.25f0]; nstart = 2)
        @test vec(ts.Xt.state[2].cmask[:, 1]) == Bool[true, false]
        @test vec(ts.X1anchor[2].cmask[:, 1]) == Bool[true, false]

        base_state = FlowceptionState(
            (
                MaskedState(ContinuousState(zeros(Float32, 1, 2, 1)), trues(2, 1), trues(2, 1)),
                MaskedState(ContinuousState(reshape(Float32[5, 6], 1, 2, 1)), reshape(Bool[true, false], 2, 1), trues(2, 1)),
            ),
            ones(Int, 2, 1);
            local_t = zeros(Float32, 2, 1),
            branchmask = reshape(Bool[true, false], 2, 1),
            flowmask = trues(2, 1),
            padmask = trues(2, 1),
        )
        hat = (
            (
                ContinuousState(fill(1f0, 1, 2, 1)),
                ContinuousState(fill(9f0, 1, 2, 1)),
            ),
            fill(-100f0, 2, 1),
        )
        next_state = Flowfusion.step(P, base_state, hat, 0f0, 0.25f0)
        @test next_state.state[2].cmask == base_state.state[2].cmask
        @test tensor(next_state.state[2].S)[1, 2, 1] == tensor(base_state.state[2].S)[1, 2, 1]
        @test tensor(next_state.state[2].S)[1, 1, 1] != tensor(base_state.state[2].S)[1, 1, 1]

        Pdir = DirectionalFlowceptionFlow((Deterministic(), Deterministic()), birth_sampler)
        tsdir = directional_flowception_bridge(Pdir, [target], [0.25f0]; nstart = 2)
        @test vec(tsdir.Xt.state[2].cmask[:, 1]) == Bool[true, false]
        dir_hat = (
            (
                ContinuousState(fill(1f0, 1, 2, 1)),
                ContinuousState(fill(9f0, 1, 2, 1)),
            ),
            fill(-100f0, 2, 2, 1),
        )
        next_dir_state = Flowfusion.step(Pdir, base_state, dir_hat, 0f0, 0.25f0)
        @test next_dir_state.state[2].cmask == base_state.state[2].cmask
        @test tensor(next_dir_state.state[2].S)[1, 2, 1] == tensor(base_state.state[2].S)[1, 2, 1]
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
