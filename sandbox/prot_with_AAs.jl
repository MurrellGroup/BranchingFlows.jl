using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path = "../")

using BranchingFlows, Flowfusion, ForwardBackward, Distributions
using Flux, RandomFeatureMaps, Onion, InvariantPointAttention, BatchedTransformations, ProteinChains, DLProteinFormats, LearningSchedules, CannotWaitForTheseOptimisers, JLD2
using DLProteinFormats: load, PDBSimpleFlat, batch_flatrecs, sample_batched_inds, length2batch

#Mixing lengths in batches will interact badly with using the mean for the loss.
#Each token counts the same towards the loss, so if a short seq batches with a long one, the short seq contributed very little.
#This could be counteracted, in expectation, by weighting t higher for lower t (when the bridges will result in short sequences)


#=
@eval Flowfusion begin
function velo_step(P, Xₜ::DiscreteState, delta_t, log_velocity, scale)
    ohXₜ = onehot(Xₜ)
    velocity = rate_constraint(tensor(ohXₜ), log_velocity, P.transform) .* scale
    newXₜ = CategoricalLikelihood(eltype(delta_t).(tensor(ohXₜ) .+ (delta_t .* velocity)))
    clamp!(tensor(newXₜ), 0, Inf) #Because one velo will be < 0 and a large step might push Xₜ < 0
    return rand(newXₜ)
end
step(P::DoobMatchingFlow, Xₜ::DiscreteState, veloX̂₁::Flowfusion.Guide, s₁, s₂) = velo_step(P, Xₜ, s₂ .- s₁, veloX̂₁.H, expand(1 ./ onescale(P, s₁), ndims(veloX̂₁.H)))
step(P::DoobMatchingFlow, Xₜ::DiscreteState, veloX̂₁, s₁, s₂) = velo_step(P, Xₜ, s₂ .- s₁, veloX̂₁, expand(1 ./ onescale(P, s₁), ndims(veloX̂₁)))
bridge(p::DoobMatchingFlow, x0::DiscreteState, x1::DiscreteState, t) = bridge(p.P, x0, x1, t)
bridge(p::DoobMatchingFlow, x0::DiscreteState, x1::DiscreteState, t0, t) = bridge(p.P, x0, x1, t0, t)
#ForwardBackward.forward!(a::CategoricalLikelihood, b::CategoricalLikelihood, P::DoobMatchingFlow, t::Real) = ForwardBackward.forward!(a, b, P.P, t)
#ForwardBackward.backward!(a::CategoricalLikelihood, b::CategoricalLikelihood, P::DoobMatchingFlow, t::Real) = ForwardBackward.backward!(a, b, P.P, t)
end
=#



#During inference, to track indices, we can spoof an extra discrete process with zero rates,
#and each step we populate it with 1:N, which will let us build an X1hat that has the right shape.
#Self-cond construction: during training, we pick a pair or contiguous residues from the same chain,
#and we mask their element-wise inputs, and their attention matrices.

#=
ipa(l, f, x, pf, c, m) = l(f, x, pair_feats = pf, cond = c, mask = m)
crossipa(l, f1, f2, x, pf, c, m) = l(f1, f2, x, pair_feats = pf, cond = c, mask = m)

struct ChainStormV1{L}
    layers::L
end
Flux.@layer ChainStormV1
function ChainStormV1(dim::Int = 384, depth::Int = 6, f_depth::Int = 6)
    layers = (;
        dim = dim,
        depth = depth,
        f_depth = f_depth,
        t_rff = RandomFourierFeatures(1 => dim, 1f0),
        cond_t_encoding = Dense(dim => dim, bias=false),
        pair_rff = RandomFourierFeatures(2 => 64, 1f0),
        pair_project = Dense(64 => 32, bias=false),
        AAencoder = Dense(21 => dim, bias=false),
        ipa_blocks = [IPAblock(dim, IPA(IPA_settings(dim, c_z = 32)), ln1 = AdaLN(dim, dim), ln2 = AdaLN(dim, dim)) for _ in 1:depth],
        framemovers = [Framemover(dim) for _ in 1:f_depth],
        count_decoder = Chain(StarGLU(dim, 3dim), Dense(dim => 1, bias=false)),
    )
    return ChainStormV1(layers)
end
function (fc::ChainStormV1)(t, Xt, chainids, resinds)
    l = fc.layers
    pmask = Flux.Zygote.@ignore self_att_padding_mask(Xt[1].lmask)
    pre_z = Flux.Zygote.@ignore l.pair_rff(pair_encode(resinds, chainids))
    pair_feats = l.pair_project(pre_z)
    t_rff = Flux.Zygote.@ignore l.t_rff(t)
    cond = reshape(l.cond_t_encoding(t_rff), :, 1, size(t,2))
    frames = Translation(tensor(Xt[1])) ∘ Rotation(tensor(Xt[2]))
    x = Flux.Zygote.@ignore similar(tensor(Xt[1]), l.dim, size(tensor(Xt[1]))[3:end]...) .= 0
    for i in 1:l.depth
        x = Flux.Zygote.checkpointed(ipa, l.ipa_blocks[i], frames, x, pair_feats, cond, pmask)
        if i > l.depth - l.f_depth
            frames = l.framemovers[i - l.depth + l.f_depth](frames, x, t = t)
        end
    end
    expected_splits = reshape(exp.(clamp.(l.count_decoder(x), -100, 11)), :, length(t))
    return frames, expected_splits
end
=#

ipa(l, f, x, pf, c, m) = l(f, x, pair_feats = pf, cond = c, mask = m)
crossipa(l, f1, f2, x, pf, c, m) = l(f1, f2, x, pair_feats = pf, cond = c, mask = m)

struct BranchChainV1{L}
    layers::L
end
Flux.@layer BranchChainV1
function BranchChainV1(dim::Int = 384, depth::Int = 6, f_depth::Int = 6)
    layers = (;
        depth = depth,
        f_depth = f_depth,
        t_rff = RandomFourierFeatures(1 => dim, 1f0),
        cond_t_encoding = Dense(dim => dim, bias=false),
        AApre_t_encoding = Dense(dim => dim, bias=false),
        pair_rff = RandomFourierFeatures(2 => 64, 1f0),
        pair_project = Dense(64 => 32, bias=false),
        AAencoder = Dense(21 => dim, bias=false),
        ipa_blocks = [IPAblock(dim, IPA(IPA_settings(dim, c_z = 32)), ln1 = AdaLN(dim, dim), ln2 = AdaLN(dim, dim)) for _ in 1:depth],
        framemovers = [Framemover(dim) for _ in 1:f_depth],
        AAdecoder = Chain(StarGLU(dim, 3dim), Dense(dim => 21, bias=false)),
        count_decoder = Chain(StarGLU(dim, 3dim), Dense(dim => 1, bias=false)),
        #SC:
        selfcond_AAencoder = StarGLU(Dense(21 => 2dim, bias=false), Dense(2dim => dim, bias=false), Dense(21 => 2dim, bias=false), swish), 
        selfcond_splits_encoder = StarGLU(Dense(1 => 2dim, bias=false), Dense(2dim => dim, bias=false), Dense(1 => 2dim, bias=false), swish), 
        selfcond_crossipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = AdaLN(dim, dim)) for _ in 1:depth],
        selfcond_selfipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = AdaLN(dim, dim)) for _ in 1:depth],

    )
    return BranchChainV1(layers)
end
function (fc::BranchChainV1)(t, Xt, chainids, resinds; sc_frames = nothing, sc_AAs = nothing, sc_splits = nothing, sc_condmask = nothing)
    l = fc.layers
    pmask = Flux.Zygote.@ignore self_att_padding_mask(Xt[1].lmask)
    if sc_condmask !== nothing
        sc_pmask = Flux.Zygote.@ignore self_att_padding_mask(sc_condmask)
    end
    pre_z = Flux.Zygote.@ignore l.pair_rff(pair_encode(resinds, chainids))
    pair_feats = l.pair_project(pre_z)
    t_rff = Flux.Zygote.@ignore l.t_rff(t)
    cond = reshape(l.cond_t_encoding(t_rff), :, 1, size(t,2))
    frames = Translation(tensor(Xt[1])) ∘ Rotation(tensor(Xt[2]))
    AA_one_hots = tensor(Xt[3])
    x = l.AAencoder(AA_one_hots .+ 0)
    if sc_frames !== nothing
        x += (l.selfcond_AAencoder(sc_AAs) + l.selfcond_splits_encoder(reshape(sc_splits, 1, size(sc_splits)...))) .* Onion.glut(sc_condmask, ndims(x), 0)
    end
    for i in 1:l.depth
        if sc_frames !== nothing
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_selfipa[i], sc_frames, sc_frames, x, pair_feats, cond, sc_pmask)
            f1, f2 = mod(i, 2) == 0 ? (frames, sc_frames) : (sc_frames, frames)
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_crossipa[i], f1, f2, x, pair_feats, cond, sc_pmask)
        end
        x = Flux.Zygote.checkpointed(ipa, l.ipa_blocks[i], frames, x, pair_feats, cond, pmask)
        if i > l.depth - l.f_depth
            frames = l.framemovers[i - l.depth + l.f_depth](frames, x, t = t)
        end
    end
    aa_logits = l.AAdecoder(x .+ reshape(l.AApre_t_encoding(t_rff), :, 1, size(t,2)))
    expected_splits = reshape(l.count_decoder(x), :, length(t))
    return frames, aa_logits, expected_splits
end


#dat = load(PDBSimpleFlat500);


function compoundstate(rec)
    L = length(rec.AAs)
    cmask = rec.AAs .< 100
    X1locs = MaskedState(ContinuousState(rec.locs), cmask, cmask)
    X1rots = MaskedState(ManifoldState(rotM,eachslice(rec.rots, dims=3)), cmask, cmask)
    X1aas = MaskedState((DiscreteState(21, rec.AAs)), cmask, cmask)
    return BranchingState((X1locs, X1rots, X1aas), rec.chainids) #<- .state, .groupings
end

const rotM = Flowfusion.Rotations(3)

#P = CoalescentFlow((BrownianMotion(0.1f0), ManifoldProcess(0.1f0), DoobMatchingFlow(UniformDiscrete(1f0), true, x -> Flowfusion.NNlib.softplus(x) .+ 1f-8)), Uniform(0.0f0, 1.0f0))


P = CoalescentFlow((BrownianMotion(0.1f0), ManifoldProcess(0.1f0), DoobMatchingFlow(UniformDiscrete(1f0), true, x -> Flowfusion.NNlib.softplus(x) .+ 1f-8)), Beta(3,5))


X0sampler(root) = (ContinuousState(randn(Float32, 3, 1, 1)), 
                    ManifoldState(rotM, reshape(Array{Float32}.(Flowfusion.rand(rotM, 1)), 1)),
                    (DiscreteState(21, [21]))
)

function losses(P, X1hat, loss_bundle)
    hatframes, hat_aas, hat_splits = X1hat
    rotangent = Flowfusion.so3_tangent_coordinates_stack(values(linear(hatframes)), tensor(loss_bundle.Xt_rots))
    hatloc, hatrot = (values(translation(hatframes)), rotangent)
    l_loc = floss(P.P[1], hatloc, loss_bundle.X1_locs_target, scalefloss(P.P[1], loss_bundle.t, 2, 0.2f0)) / 2
    l_rot = floss(P.P[2], hatrot, loss_bundle.rotξ_target, scalefloss(P.P[2], loss_bundle.t, 2, 0.2f0)) / 10
    l_aas = floss(P.P[3], loss_bundle.X1_aas, hat_aas, loss_bundle.doob_target, scalefloss(P.P[3],loss_bundle.t,1, 0.2f0)) / 20
    #Note: second term here doesn't affect the gradients but just makes it so the min count loss is zero
    #l_splits = BranchingFlows.poisson_loss(hat_splits, loss_bundle.splits_target, loss_bundle.padmask) - BranchingFlows.poisson_loss(loss_bundle.splits_target, loss_bundle.splits_target, loss_bundle.padmask)
    l_splits = floss(P, hat_splits, loss_bundle.splits_target, loss_bundle.padmask, scalefloss(P,loss_bundle.t,1, 0.2f0)) / 5
    return l_loc, l_rot, l_aas, l_splits
end

device = identity

#dat = load(PDBSimpleFlat);

model = BranchChainV1(128, 3,3) |> device
sched = burnin_learning_schedule(0.000005f0, 0.001f0, 1.05f0, 0.999975f0)
#       burnin_learning_schedule(0.000005f0, 0.001f0, 1.05f0, 0.999975f0)
opt_state = Flux.setup(Muon(eta = sched.lr), model)
#sched = linear_decay_schedule(0.001f0, 0.000000001f0, 540) 


#model = ChainStormV1(128, 3,3)
#Flux.loadmodel!(model, JLD2.load("model_epoch_46.jld", "model_state"))
#opt_state = JLD2.load("model_epoch_46.jld", "opt_state")
#sched = linear_decay_schedule(0.000375f0, 0.000000001f0, 540) 

#sched = burnin_learning_schedule(0.00005f0, 0.0005f0, 1.05f0, 0.99995f0)

Ls = []
for epoch in 2:100
    if epoch == 2 #So that the model doesn't get jolted by the self-cond
        burnin_learning_schedule(0.000005f0, 0.001f0, 1.05f0, 0.999975f0)
        Flux.adjust!(opt_state, next_rate(sched))
    end
    batchinds = sample_batched_inds(dat,l2b = length2batch(100, 1.9))
    avg_l = 0f0
    for (i, b) in enumerate(batchinds)
        X1s = compoundstate.(dat[b]);
        bat = branching_bridge(P, X0sampler, X1s);
        Xtstate, groupings = bat.Xt.state, bat.Xt.groupings
        resinds = similar(groupings) .= 1:size(groupings, 1)
        rotξ = Guide(Xtstate[2], bat.X1anchor[2])
        doobG = Guide(P.P[3], bat.t, Xtstate[3].S, bat.X1anchor[3].S) #This sets up the "training target rate" via a Doob h-transform
        input_bundle = (bat.t', (Xtstate[1], Xtstate[2], onehot(Xtstate[3])), groupings, resinds) |> device
        loss_bundle = (;t = bat.t, Xt_rots = Xtstate[2], rotξ_target = rotξ, X1_locs_target = bat.X1anchor[1], doob_target = doobG, X1_aas = onehot(bat.X1anchor[3]), splits_target = bat.splits_target, padmask = bat.padmask) |> device
        sc_frames, sc_aas, sc_splits, sc_condmask = nothing, nothing, nothing, nothing
        if epoch > 1 && rand() < 0.5
            sc_frames, sc_aas, sc_splits = model(input_bundle...)
            sc_condmask = (bat.prev_coalescence .< (reshape(bat.t, 1, :) .* (rand() * 0.1 + 0.9))) .& bat.padmask
        end
        l, grad = Flux.withgradient(model) do m
            X1hat = m(input_bundle..., sc_frames = sc_frames, sc_AAs = sc_aas, sc_splits = sc_splits, sc_condmask = sc_condmask)
            l_loc, l_rot, l_aas, l_splits = losses(P, X1hat, loss_bundle)
            mod(i, 50) == 0 && println("l_loc: $l_loc, l_rot: $l_rot, l_aas: $l_aas, l_splits: $l_splits")
            l_loc + l_rot + l_aas + l_splits
        end
        avg_l += l
        Flux.update!(opt_state, model, grad[1])
        if mod(i, 10) == 0
            println("Loss: $(avg_l/10), Epoch: $epoch, Iter: $i, Rate: $(sched.lr)")
            push!(Ls, avg_l / 10)
            avg_l = 0f0
            Flux.adjust!(opt_state, next_rate(sched))
        end
    end
    jldsave("model_epoch_$epoch.jld", model_state = Flux.state(cpu(model)), opt_state=cpu(opt_state))
end



#floss(P::Flowfusion.fbu(DoobMatchingFlow), Xt::Flowfusion.msu(DiscreteState), X̂₁, X₁::Guide, c) = Flowfusion.scaledmaskedmean(cgm_dloss(P, tensor(Xt), tensor(X̂₁), X₁.H), c, Flowfusion.getlmask(X₁))


#bat = branching_bridge(P, X0sampler, X1s) <- X1s is a vector of state-tuples and their groupings.
#we're going to replace that with an actual type!


#model = ChainStormV1(128, 3,3)
#Flux.loadmodel!(model, JLD2.load("model_epoch_46.jld", "model_state"))




function m_wrapper(t, Xₜ)
    Xtstate = MaskedState.(Xₜ.state, (Xₜ.groupings .< Inf,), (Xₜ.groupings .< Inf,))
    resinds = similar(Xₜ.groupings) .= 1:size(Xₜ.groupings, 1)
    #input_bundle = ([t]', Xtstate, Xₜ.groupings, resinds)
    input_bundle = ([t]', (Xtstate[1], Xtstate[2], onehot(Xtstate[3])), Xₜ.groupings, resinds) 
    pred = model(input_bundle...)
    #if t < 0.1f0
    #    pred[3] .= max.(5f0, pred[3])
    #end
    println("t: $t, length(pred[3]): $(length(pred[3]))")
    state_pred = ContinuousState(values(translation(pred[1]))), ManifoldState(rotM, eachslice(cpu(values(linear(pred[1]))), dims=(3,4))), pred[2]
    return state_pred, pred[3]
end



function bbtensor(samp)
    prot = DLProteinFormats.unflatten(tensor(samp.state[1])[:,:,:,1], tensor(samp.state[2])[:,:,:,1], ones(Int,length(samp.groupings[:])), samp.groupings[:], 1:length(samp.groupings[:]))
    bb = cat(get_backbone.(prot)..., dims = 3)
    return bb
end

X0n = 3
X0 = BranchingFlows.BranchingState(BranchingFlows.regroup([[X0sampler(BranchingFlows.FlowNode(1f0, nothing)) for _ in 1:X0n]]), [1:X0n ;;])
X0 = BranchingFlows.BranchingState((X0.state[1], X0.state[2], onehot(X0.state[3])), X0.groupings)
paths = Tracker()
samp = gen(P, X0, m_wrapper, 0f0:0.005f0:1f0, tracker = paths)


x0_t = bbtensor(X0)
samp_t = bbtensor(samp)
path_t = [bbtensor(p[1]) for p in paths.xt]
aug_path = vcat([x0_t for _ in 1:10], path_t, [samp_t for _ in 1:10])
grouping_path = vcat([X0.groupings for _ in 1:10], [p[1].groupings for p in paths.xt], [samp.groupings for _ in 1:10])

anim = @animate for (i,bb) in enumerate(aug_path)
    pl = scatter3d(bb[1,1,:], bb[2,1,:], bb[3,1,:], msw = 0, markersize = 1.5, label = :none)
    scatter3d!(bb[1,2,:], bb[2,2,:], bb[3,2,:], msw = 0, markersize  = 1.5, label = :none )
    scatter3d!(bb[1,3,:], bb[2,3,:], bb[3,3,:], msw = 0, markersize = 1.5, label = :none)
    for g in union(grouping_path[i][:])
        inds = (g .== grouping_path[i][:])
        bbr = reshape(bb[:,:,inds], 3, :)
        plot3d!(bbr[1,:], bbr[2,:], bbr[3,:], line_z = g, label = :none, cmap = :rainbow)
    end
   plot3d!(samp_t[1,2,:], samp_t[2,2,:], samp_t[3,2,:], color = "black", alpha = 0, label = :none)
    pl
end
gif(anim, "anim_fps_$(rand(10000:99999)).mp4", fps = 15)



plot((1:length(path_t)) ./ length(path_t), [size(p,3)-3 for p in path_t]/(size(path_t[end],3)-3))
plot!(0:0.0001:1, x -> cdf(P.branch_time_dist,x))




anim = @animate for bb in aug_path
    pl = scatter3d(bb[1,1,:], bb[2,1,:], bb[3,1,:], msw = 0, markersize = 3, label = :none)
    scatter3d!(bb[1,2,:], bb[2,2,:], bb[3,2,:], msw = 0, markersize  = 3, label = :none )
    scatter3d!(bb[1,3,:], bb[2,3,:], bb[3,3,:], msw = 0, markersize = 3, label = :none)
    bbr = reshape(bb, 3, :)
    plot3d!(bbr[1,:], bbr[2,:], bbr[3,:], color = "black", label = :none)
    plot3d!(samp_t[1,2,:], samp_t[2,2,:], samp_t[3,2,:], color = "black", alpha = 0, label = :none)
    pl
end
gif(anim, "anim_fps_$(rand(10000:99999)).mp4", fps = 15)


bb = samp_t
pl = scatter3d(bb[1,1,:], bb[2,1,:], bb[3,1,:], msw = 0, markersize = 1.5, label = :none)
scatter3d!(bb[1,2,:], bb[2,2,:], bb[3,2,:], msw = 0, markersize  = 1.5, label = :none )
scatter3d!(bb[1,3,:], bb[2,3,:], bb[3,3,:], msw = 0, markersize = 1.5, label = :none)
for g in union(samp.groupings[:])
    inds = (g .== samp.groupings[:])
    bbr = reshape(bb[:,:,inds], 3, :)
    plot3d!(bbr[1,:], bbr[2,:], bbr[3,:], line_z = g, label = :none, cmap = :rainbow)
end
plot3d!(samp_t[1,2,:], samp_t[2,2,:], samp_t[3,2,:], color = "black", alpha = 0, label = :none)
pl





s = tensor(samp.state[1])[:,1,:,1]
plot3d(s[1,:], s[2,:], s[3,:])




#Static plot - can't see shit:
pl = plot()
for bb in path_t
    #scatter3d!(bb[1,1,:], bb[2,1,:], bb[3,1,:], msw = 0, markersize = 1, label = :none)
    scatter3d!(bb[1,2,:], bb[2,2,:], bb[3,2,:], msw = 0, markersize  = 1, label = :none, marker_z = 1:size(bb,3), cmap = :rainbow, colorbar = false)
    #scatter3d!(bb[1,3,:], bb[2,3,:], bb[3,3,:], msw = 0, markersize = 1, label = :none)
end
scatter3d!(x0_t[1,2,:], x0_t[2,2,:], x0_t[3,2,:], msw = 0, markersize = 3, label = :none, color = "green")
scatter3d!(samp_t[1,2,:], samp_t[2,2,:], samp_t[3,2,:], msw = 0, markersize = 3, label = :none, color = "blue")
pl



anim = @animate for bb in aug_path
    pl = scatter3d(bb[1,1,:], bb[2,1,:], bb[3,1,:], msw = 0, markersize = 3, label = :none)
    scatter3d!(bb[1,2,:], bb[2,2,:], bb[3,2,:], msw = 0, markersize  = 3, label = :none )
    scatter3d!(bb[1,3,:], bb[2,3,:], bb[3,3,:], msw = 0, markersize = 3, label = :none)
    bbr = reshape(bb, 3, :)
    plot3d!(bbr[1,:], bbr[2,:], bbr[3,:], color = "black", label = :none)
    plot3d!(samp_t[1,2,:], samp_t[2,2,:], samp_t[3,2,:], color = "black", alpha = 0, label = :none)
    pl
end
gif(anim, "anim_fps15.mp4", fps = 15)


#prot = DLProteinFormats.unflatten(tensor(samp.state[1])[:,:,:,1], tensor(samp.state[2])[:,:,:,1], ones(Int,length(samp.groupings[:])), samp.groupings[:], 1:length(samp.groupings[:]))
#bb = get_backbone(prot[1])


function 


tensor(BranchingFlows.element(samp.state[2],1,1))[1,1,1] = 2
tensor(BranchingFlows.element(samp.state[2],1,1))[1,1,1]



X₀ = BranchingFlows.BranchingState(BranchingFlows.zerostate.(X0sampler(BranchingFlows.FlowNode(1f0, nothing)), 1, 1), ones(Int,1,1))
Xₜ = deepcopy(X₀)


s₁, s₂ = 0.01f0, 0.05f0
t = s₁
hat = Flowfusion.resolveprediction(m_wrapper(t, Xₜ), Xₜ)
hat[1][1] |> tensor |> size
hat[1][2] |> tensor |> size
Xₜ = Flowfusion.mask(BranchingFlows.Flowfusion.step(P, Xₜ, hat, s₁, s₂), X₀)


s₁, s₂ = 0.01f0, 0.05f0
t = s₁
hat = Flowfusion.resolveprediction(m_wrapper(t, Xₜ), Xₜ)
Xₜ = Flowfusion.mask(BranchingFlows.Flowfusion.step(P, Xₜ, hat, s₁, s₂), X₀)





X1targets, event_lambdas = hat
    Xₜ = bridge(P.P, Xₜ, X1targets, s₁, s₂)


BranchingFlows.Flowfusion.step(P, Xₜ, hat, s₁, s₂)


X0 = BranchingFlows.BranchingState(BranchingFlows.zerostate.(X0sampler(BranchingFlows.FlowNode(1f0, nothing)), 1, 1), ones(Int,1,1))
samp = gen(P, X0, m_wrapper, 0f0:0.01f0:1f0)



function gen(P::BranchingProcess, X₀, model, args...; kwargs...)
    gen((P,), (X₀,), (t, Xₜ) -> (model(t[1], Xₜ[1]),), args...; kwargs...)[1]
end


@eval Flowfusion begin
    function gen(P, X₀, model, args...; kwargs...)
        println(typeof(P))
        println(typeof(X₀))
        println("##################################")
        println("##################################")
        println("##################################")
        gen((P,), (X₀,), (t, Xₜ) -> (model(t[1], Xₜ[1]),), args...; kwargs...)[1]
    end
end



@eval Flowfusion begin
function gen(P::Tuple{Vararg{UProcess}}, X₀::Tuple{Vararg{UState}}, model, steps::AbstractVector; tracker::Function=Returns(nothing), midpoint = false)
    @show "1"
    Xₜ = deepcopy.(X₀)
    for (s₁, s₂) in zip(steps, steps[begin+1:end])
        t = midpoint ? (s₁ + s₂) / 2 : t = s₁
        hat = resolveprediction(model(t, Xₜ), Xₜ)
        Xₜ = mask(step(P, Xₜ, hat, s₁, s₂), X₀)
        tracker(t, Xₜ, hat)
    end
    return Xₜ
end
end


genX0sampler(root) = BranchingState((ContinuousState(randn(Float32, 3, 1, 1)), ManifoldState(rotM, reshape(Array{Float32}.(Flowfusion.rand(rotM, 1, 1)), 1, 1)),), ones(Int,1,1))
                    #DiscreteState(21, [21])



X0 = BranchingState(BranchingFlows.zerostate.(X0sampler(BranchingFlows.FlowNode(1f0, nothing)), 1, 1), ones(Int,1,1))
#X0 = genX0sampler(BranchingFlows.FlowNode(1f0, nothing))
m_wrapper(0.000001f0, X0)

Xₜ = X0
t = 0.001f0
Xtstate = MaskedState.(Xₜ.state, (Xₜ.groupings .< Inf,), (Xₜ.groupings .< Inf,))
    resinds = similar(Xₜ.groupings) .= 1:size(Xₜ.groupings, 1)
    input_bundle = ([t]', Xtstate, Xₜ.groupings, resinds)


samp = gen(P, X0, m_wrapper, 0f0:0.01f0:1f0)


(t, Xt, chainids, resinds)


