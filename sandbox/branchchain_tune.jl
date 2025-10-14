#concern: the fast mixing schedule might not be a good target for learning.
#try: mixed-segment schedule, where we switch from eg. slow BM to fast mixing at t=0.5.
#This will get Xt to a point where the end structure is very predictable, but still allow fluctuations near the end.


using Pkg
Pkg.activate(".")
Pkg.add("Revise")

using Revise

Pkg.develop(path = "../")

#Pkg.add(["JLD2", "Flux", "CannotWaitForTheseOptimisers", "LearningSchedules", "DLProteinFormats"])
#Pkg.add(["CUDA", "cuDNN"])

#Pkg.develop(path="../../ForwardBackward.jl/")
#Pkg.develop(path="../../Flowfusion.jl/")
#Pkg.develop(path="../../ChainStorm.jl/")

using DLProteinFormats, Flux, CannotWaitForTheseOptimisers, LearningSchedules, JLD2, Dates, BatchedTransformations, ProteinChains
using BranchingFlows, Flowfusion, Distributions, ForwardBackward, RandomFeatureMaps, InvariantPointAttention, Onion
using DLProteinFormats: load, PDBSimpleFlat, batch_flatrecs, sample_batched_inds, length2batch
using CUDA

device!(0) #Because we have set CUDA_VISIBLE_DEVICES = GPUnum
device = gpu

rundir = "runs/jittery_moresplitty_$(Date(now()))_$(rand(100000:999999))"
mkpath("$(rundir)/samples")

function textlog(filepath::String, l; also_print = true)
    f = open(filepath,"a")
        write(f, join(string.(l),", "))
        write(f, "\n")
    close(f)
    if also_print
        println(join(string.(l),", "))
    end
end

oldAdaLN(dim,cond_dim) = AdaLN(Flux.LayerNorm(dim), Dense(cond_dim, dim), Dense(cond_dim, dim))

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
        selfcond_crossipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = oldAdaLN(dim, dim)) for _ in 1:depth],
        selfcond_selfipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = oldAdaLN(dim, dim)) for _ in 1:depth],
        ipa_blocks = [IPAblock(dim, IPA(IPA_settings(dim, c_z = 32)), ln1 = oldAdaLN(dim, dim), ln2 = oldAdaLN(dim, dim)) for _ in 1:depth],
        framemovers = [Framemover(dim) for _ in 1:f_depth],
        AAdecoder = Chain(StarGLU(dim, 3dim), Dense(dim => 21, bias=false)),
        #New layesr to tune-in
        indelpre_t_encoding = Dense(dim => 3dim),
        count_decoder = StarGLU(Dense(3dim => 2dim, bias=false), Dense(2dim => 1, bias=false), Dense(3dim => 2dim, bias=false), Flux.swish),
        del_decoder   = StarGLU(Dense(3dim => 2dim, bias=false), Dense(2dim => 1, bias=false), Dense(3dim => 2dim, bias=false), Flux.swish)
    )
    return BranchChainV1(layers)
end
function (fc::BranchChainV1)(t, Xt, chainids, resinds; sc_frames = nothing)
    l = fc.layers
    pmask = Flux.Zygote.@ignore self_att_padding_mask(Xt[1].lmask)
    pre_z = Flux.Zygote.@ignore l.pair_rff(pair_encode(resinds, chainids))
    pair_feats = l.pair_project(pre_z)
    t_rff = Flux.Zygote.@ignore l.t_rff(t)
    cond = reshape(l.cond_t_encoding(t_rff), :, 1, size(t,2))
    frames = Translation(tensor(Xt[1])) ∘ Rotation(tensor(Xt[2]))
    AA_one_hots = tensor(Xt[3])
    x = l.AAencoder(AA_one_hots .+ 0)
    x_a,x_b,x_c = nothing, nothing, nothing
    for i in 1:l.depth
        if sc_frames !== nothing
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_selfipa[i], sc_frames, sc_frames, x, pair_feats, cond, pmask)
            f1, f2 = mod(i, 2) == 0 ? (frames, sc_frames) : (sc_frames, frames)
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_crossipa[i], f1, f2, x, pair_feats, cond, pmask)
        end
        x = Flux.Zygote.checkpointed(ipa, l.ipa_blocks[i], frames, x, pair_feats, cond, pmask)
        if i > l.depth - l.f_depth
            frames = l.framemovers[i - l.depth + l.f_depth](frames, x, t = t)
        end
        if i==4 (x_a = x) end
        if i==5 (x_b = x) end
        if i==6 (x_c = x) end
    end
    aa_logits = l.AAdecoder(x .+ reshape(l.AApre_t_encoding(t_rff), :, 1, size(t,2)))
    #New:
    catted = vcat(x_a,x_b,x_c)
    indel_pre_t = reshape(l.indelpre_t_encoding(t_rff), :, 1, size(t,2))
    count_log = reshape(l.count_decoder(catted .+ indel_pre_t),  :, length(t))
    del_logits = reshape(l.del_decoder(catted .+ indel_pre_t),  :, length(t))
    return frames, aa_logits, count_log, del_logits
end

P = CoalescentFlow(((OUBridgeExpVar(100f0, 150f0, 0.000000001f0, dec = -3f0), 
                     ManifoldProcess(OUBridgeExpVar(100f0, 150f0, 0.000000001f0, dec = -3f0)), 
                     DistNoisyInterpolatingDiscreteFlow(D1=Beta(3.0,1.5)))), 
                    Beta(1,2))

const rotM = Flowfusion.Rotations(3)

X0sampler(root) = (ContinuousState(randn(Float32, 3, 1, 1)), 
                    ManifoldState(rotM, reshape(Array{Float32}.(Flowfusion.rand(rotM, 1)), 1)),
                    (DiscreteState(21, [21]))
)

function compoundstate(rec; del_prob_dist = Uniform(0.0, 0.1))
    L = length(rec.AAs)
    cmask = rec.AAs .< 100
    X1locs = MaskedState(ContinuousState(rec.locs), cmask, cmask)
    X1rots = MaskedState(ManifoldState(rotM,eachslice(rec.rots, dims=3)), cmask, cmask)
    X1aas = MaskedState((DiscreteState(21, rec.AAs)), cmask, cmask)
    X1 = BranchingState((X1locs, X1rots, X1aas), rec.chainids) #<- .state, .groupings
    X1 = uniform_del_insertions(X1, rand(del_prob_dist))
    X1.state[3].S.state[X1.del] .= 21 #Set deleted discrete components to the dummy!
    return X1
end

function training_prep(b)
    X1s = compoundstate.(dat[b], del_prob_dist = Uniform(0.2, 0.20000001))
    t = Uniform(0f0,1f0)
    bat = branching_bridge(P, X0sampler, X1s, t, 
                            coalescence_factor = Uniform(0.25, 0.250000001), 
                            use_branching_time_prob = 0.5,
                            merger = BranchingFlows.canonical_anchor_merge,
                            maxlen = maximum(dat.len[b]) #Because this OOM'd
                        )
    rotξ = Guide(bat.Xt.state[2], bat.X1anchor[2])
    resinds = similar(bat.Xt.groupings) .= 1:size(bat.Xt.groupings, 1)
    return (;t = bat.t, chainids = bat.Xt.groupings, resinds,
                    Xt = (bat.Xt.state[1], bat.Xt.state[2], onehot(bat.Xt.state[3])),
                    rotξ_target = rotξ, X1_locs_target = bat.X1anchor[1], X1aas_target = bat.X1anchor[3],
                    splits_target = bat.splits_target, del = bat.del, padmask = bat.padmask)
end


function losses(P, X1hat, ts)
    hat_frames, hat_aas, hat_splits, hat_del = X1hat
    rotangent = Flowfusion.so3_tangent_coordinates_stack(values(linear(hat_frames)), tensor(ts.Xt[2]))
    hat_loc, hat_rot, hat_aas = (values(translation(hat_frames)), rotangent, hat_aas)
    l_loc = floss(P.P[1], hat_loc, ts.X1_locs_target,                scalefloss(P.P[1], ts.t, 1, 0.2f0)) * 20
    l_rot = floss(P.P[2], hat_rot, ts.rotξ_target,                   scalefloss(P.P[2], ts.t, 1, 0.2f0)) * 2
    l_aas = floss(P.P[3], hat_aas, onehot(ts.X1aas_target),          scalefloss(P.P[3], ts.t, 1, 0.2f0)) / 10
    splits_loss = floss(P, hat_splits, ts.splits_target, ts.padmask, scalefloss(P, ts.t, 1, 0.2f0))
    del_loss = floss(P.deletion_policy, hat_del, ts.del, ts.padmask, scalefloss(P, ts.t, 1, 0.2f0))
    return l_loc, l_rot, l_aas, splits_loss, del_loss
end

function mod_wrapper(t, Xₜ)
    Xtstate = MaskedState.(Xₜ.state, (Xₜ.groupings .< Inf,), (Xₜ.groupings .< Inf,))
    resinds = similar(Xₜ.groupings) .= 1:size(Xₜ.groupings, 1)
    input_bundle = ([t]', (Xtstate[1], Xtstate[2], onehot(Xtstate[3])), Xₜ.groupings, resinds) |> device
    sc_frames, _ = model(input_bundle...)
    sc_frames, _ = model(input_bundle..., sc_frames = sc_frames)
    sc_frames, _ = model(input_bundle..., sc_frames = sc_frames)
    pred = model(input_bundle..., sc_frames = sc_frames) |> cpu
    println("t: $t, $(length(pred[3]))")
    state_pred = ContinuousState(values(translation(pred[1]))), ManifoldState(rotM, eachslice(cpu(values(linear(pred[1]))), dims=(3,4))), pred[2]
    return state_pred, pred[3], pred[4]
end



function gen2prot(samp, chainids, resnums; name = "Gen", )
    d = Dict(zip(0:25,'A':'Z'))
    chain_letters = get.((d,), chainids, 'Z')
    ProteinStructure(name, Atom{eltype(tensor(samp[1]))}[], DLProteinFormats.unflatten(tensor(samp[1]), tensor(samp[2]), tensor(samp[3]), chain_letters, resnums)[1])
 end
export_pdb(path, samp, chainids, resnums) = ProteinChains.writepdb(path, gen2prot(samp, chainids, resnums))

function test_sample(path; numchains = 1, chainlength_dist = Poisson(30))
    X0n = numchains
    chainlengths = rand(chainlength_dist, X0n)
    groupings = reshape(vcat([i for i in 1:X0n for _ in 1:chainlengths[i]]), :, 1)
    X0 = BranchingFlows.BranchingState(BranchingFlows.regroup([[X0sampler(BranchingFlows.FlowNode(1f0, nothing)) for _ in 1:length(groupings)]]), [1:length(groupings) ;;])
    X0 = BranchingFlows.BranchingState((X0.state[1], X0.state[2], X0.state[3]), groupings) #Converts to onehot
    @show size(X0.groupings)
    #paths = Tracker()
    samp = gen(P, X0, mod_wrapper, 0f0:0.005f0:1f0)#, tracker = paths)
    @show sum(tensor(samp.state[3]) .== 21)
    export_pdb(path, samp.state, samp.groupings, collect(1:length(samp.groupings)))
end

dat = load(PDBSimpleFlat);

#CSmodel = Flux.loadmodel!(ChainStormV1(), JLD2.load("/home/murrellb/BranchingFlows.jl/sandbox/model_epoch_3_branchchain1_2025-10-09_933211.jld", "model_state"))
#CSmodel = Flux.loadmodel!(ChainStormV1(), JLD2.load("ChainStormV1_lessjittery_epoch_1.jld", "model_state"))
model = Flux.loadmodel!(BranchChainV1(), JLD2.load("model_epoch_3_branchchain1_2025-10-09_933211.jld", "model_state")) |> device

sched = burnin_learning_schedule(0.0005f0, 0.0010f0, 1.05f0, 0.9995f0)
opt_state = Flux.setup(Muon(eta = sched.lr, fallback = x -> any(size(x) .== 21)), model)
Flux.MLDataDevices.Internal.unsafe_free!(x) = (Flux.fmapstructure(Flux.MLDataDevices.Internal.unsafe_free_internal!, x); return nothing)

struct BatchDataset{T}
    batchinds::T
end
Base.length(x::BatchDataset) = length(x.batchinds)
Base.getindex(x::BatchDataset, i) = training_prep(x.batchinds[i])
function batchloader(; device=identity, parallel=true)
    uncapped_l2b = length2batch(1500, 1.9)
    batchinds = sample_batched_inds(dat,l2b = x -> min(uncapped_l2b(x), 100))
    @show length(batchinds)
    x = BatchDataset(batchinds)
    dataloader = Flux.DataLoader(x; batchsize=-1, parallel)
    return device(dataloader)
end

#sched = linear_decay_schedule(sched.lr, 0.000000001f0, 1700) 
textlog("$(rundir)/log.csv", ["epoch", "batch", "learning rate", "loss"])
for epoch in 1:3
    if epoch == 3
        sched = linear_decay_schedule(sched.lr, 0.000000001f0, 1700) 
    end
    for (i, ts) in enumerate(batchloader(; device = device))
        sc_frames = nothing
        if rand() < 0.5
            sc_frames, _ = model(ts.t', ts.Xt, ts.chainids, ts.resinds)
        end
        l, grad = Flux.withgradient(model) do m
            frames, aa_logits, count_log, del_logit = m(ts.t', ts.Xt, ts.chainids, ts.resinds, sc_frames = sc_frames)
            l_loc, l_rot, l_aas, l_splits, l_del = losses(P, (frames, aa_logits, count_log, del_logit), ts)
            @show l_loc, l_rot, l_aas, l_splits, l_del
            l_loc + l_rot + l_aas + l_splits + l_del
        end
        Flux.update!(opt_state, model, grad[1])
        (mod(i, 10) == 0) && Flux.adjust!(opt_state, next_rate(sched))
        textlog("$(rundir)/log.csv", [epoch, i, sched.lr, l])
        if mod(i, 1000) == 1
            for samp in 1:10
                test_sample("$(rundir)/samples/test_sample_$(epoch)_$(i)_$(samp).pdb", numchains = rand(1:3), chainlength_dist = Poisson(rand(10:150)))
            end
        end
    end
    jldsave("$(rundir)/model_epoch_$(epoch).jld", model_state = Flux.state(cpu(model)), opt_state=cpu(opt_state))
end


#lessjittery wound up as:
#CoalescentFlow{Tuple{OUBridgeExpVar{Float32, Float32, Vector{Float32}, Vector{Float32}}, ManifoldProcess{OUBridgeExpVar{Float32, Float32, Vector{Float32}, Vector{Float32}}}, DistNoisyInterpolatingDiscreteFlow{Beta{Float64}, Beta{Float64}, Nothing}}, Beta{Float64}, BranchingFlows.var"#26#27", SequentialUniform, BranchingFlows.UniformDeletion}((OUBridgeExpVar{Float32, Float32, Vector{Float32}, Vector{Float32}}(10.0f0, -0.7859354f0, Float32[15.785935], Float32[-3.0]), ManifoldProcess{OUBridgeExpVar{Float32, Float32, Vector{Float32}, Vector{Float32}}}(OUBridgeExpVar{Float32, Float32, Vector{Float32}, Vector{Float32}}(10.0f0, -0.7859354f0, Float32[15.785935], Float32[-3.0])), DistNoisyInterpolatingDiscreteFlow{Beta{Float64}, Beta{Float64}, Nothing}(Beta{Float64}(α=3.0, β=1.5), Beta{Float64}(α=2.0, β=2.0), 0.2, nothing)), Beta{Float64}(α=1.0, β=2.0), BranchingFlows.var"#26#27"(), SequentialUniform(), BranchingFlows.UniformDeletion())

