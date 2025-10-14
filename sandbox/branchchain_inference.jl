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
using BranchingFlows, Flowfusion, Distributions, ForwardBackward, RandomFeatureMaps, InvariantPointAttention, Onion, RandomFeatureMaps
using DLProteinFormats: load, PDBSimpleFlat, batch_flatrecs, sample_batched_inds, length2batch
using CUDA

device!(0) #Because we have set CUDA_VISIBLE_DEVICES = GPUnum
device = gpu

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

function test_sample(path; numchains = 1, chainlength_dist = Poisson(30), chainlengths = nothing)
    if isnothing(chainlengths)
        X0n = numchains
        chainlens = rand(chainlength_dist, X0n)
    else
        X0n = length(chainlengths)
    end
    groupings = reshape(vcat([i for i in 1:X0n for _ in 1:chainlengths[i]]), :, 1)
    X0 = BranchingFlows.BranchingState(BranchingFlows.regroup([[X0sampler(BranchingFlows.FlowNode(1f0, nothing)) for _ in 1:length(groupings)]]), [1:length(groupings) ;;])
    X0 = BranchingFlows.BranchingState((X0.state[1], X0.state[2], X0.state[3]), groupings) #Converts to onehot
    @show size(X0.groupings)
    #paths = Tracker()
    samp = gen(P, X0, mod_wrapper, 0f0:0.005f0:0.999f0)#, tracker = paths)
    @show sum(tensor(samp.state[3]) .== 21)
    export_pdb(path, samp.state, samp.groupings, collect(1:length(samp.groupings)))
    println("Saved to $path")
    return samp
end

model = Flux.loadmodel!(BranchChainV1(), JLD2.load("model_epoch_3_branchchain1_2025-10-09_933211.jld", "model_state")) |> device

#samp = test_sample("samples/test_sample_$(rand(100000:999999)).pdb", numchains = rand(1:1), chainlength_dist = 1200:1200)
#samp = test_sample("samples/test_sample_$(rand(100000:999999)).pdb", numchains = rand(4:4), chainlength_dist = 96:96)
samp = test_sample("samples/test_sample_$(rand(100000:999999)).pdb", chainlengths = [96,96])
prot = gen2prot(samp.state, samp.groupings, collect(1:length(samp.groupings)); name = "Gen");
println(">seq\n$(prot.chains[1].sequence)")

chainlengths = [30]
X0n = length(chainlengths)
groupings = reshape(vcat([i for i in 1:X0n for _ in 1:chainlengths[i]]), :, 1)
X0 = BranchingFlows.BranchingState(BranchingFlows.regroup([[X0sampler(BranchingFlows.FlowNode(1f0, nothing)) for _ in 1:length(groupings)]]), [1:length(groupings) ;;])
X0 = BranchingFlows.BranchingState((X0.state[1], X0.state[2], X0.state[3]), groupings) #Converts to onehot
samp = gen(P, X0, mod_wrapper, 0f0:0.005f0:1f0)#, tracker = paths)
tensor(samp.state[3])


#=
@eval Flowfusion begin
    function step(P::DistNoisyInterpolatingDiscreteFlow,
                  Xₜ::DiscreteState{<:AbstractArray{<:Signed}},
                  X̂₁logits, s₁, s₂)
        X̂₁ = LogExpFunctions.softmax(X̂₁logits)
        T    = eltype(s₁)
        Δt   = s₂ .- s₁
        ohXₜ = onehot(Xₜ)
        pu   = T(1 / Xₜ.K)
        ϵ    = T(1e-10)
        κ1_ = κ1(P, s₁)
        κ2_ = κ2(P, s₁)
        κ3_ = 1 .- κ1_ .- κ2_
        dκ1_ = dκ1(P, s₁)
        dκ2_ = dκ2(P, s₁)
        dκ3_ = .- (dκ1_ .+ dκ2_)
        βt = dκ3_ ./ max.(κ3_, ϵ)
        # v = (dκ1 - κ1*β) * X̂₁ + (dκ2 - κ2*β) * pu + β * oh(X_t)
        velo = (dκ1_ .- κ1_ .* βt) .* tensor(X̂₁) .+
               (dκ2_ .- κ2_ .* βt) .* pu .+
               βt .* tensor(ohXₜ)
        newXₜ = CategoricalLikelihood(eltype(s₁).(tensor(ohXₜ) .+ (Δt .* velo)))
        clamp!(tensor(newXₜ), T(0), T(Inf))
        return rand(newXₜ)
    end
    end
    =#

