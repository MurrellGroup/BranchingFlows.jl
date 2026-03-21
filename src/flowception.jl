"""
    linear_scheduler(t)

Default Flowception reveal scheduler `κ(t) = t`.
"""
linear_scheduler(t) = t

"""
    linear_scheduler_derivative(t)

Derivative of `linear_scheduler`.
"""
linear_scheduler_derivative(t) = one(t)

"""
    linear_scheduler_inverse(t)

Inverse of `linear_scheduler`.
"""
linear_scheduler_inverse(t) = t

state_tuple(x) = x isa Tuple ? x : (x,)
smalltime(x::Real) = oftype(x, 1e-6)
clip01(x::Real) = clamp(x, zero(x), one(x))
clip01(x::AbstractArray) = clamp.(x, zero(eltype(x)), one(eltype(x)))
expand_like(x, ref) = reshape(x, ntuple(_ -> 1, ndims(ref) - ndims(x))..., size(x)...)

"""
    FlowceptionState(state, groupings;
                     local_t = ones(Float32, size(groupings)),
                     branchmask = ones(Bool, size(groupings)),
                     flowmask = ones(Bool, size(groupings)),
                     padmask = ones(Bool, size(groupings)))

Variable-length sequence state used by `FlowceptionFlow`.

`state` is stored as a tuple of masked or unmasked Flowfusion states. The
sequence-level bookkeeping matches `BranchingState` closely, with one extra
field:

- `local_t`: per-element denoising times in `[0, 1]`.

Semantics:

- `branchmask`: whether insertions are permitted to the right of an element.
- `flowmask`: whether the element should keep denoising.
- `padmask`: whether the sequence position is valid.

Unbatched targets are typically vectors, while batched bridge outputs use
matrices of shape `(L, B)`.
"""
struct FlowceptionState{A,B,C,D,E,F} <: State
    state::A
    groupings::B
    local_t::C
    branchmask::D
    flowmask::E
    padmask::F
end

function FlowceptionState(state, groupings;
        local_t = ones(Float32, size(groupings)),
        branchmask = ones(Bool, size(groupings)),
        flowmask = ones(Bool, size(groupings)),
        padmask = ones(Bool, size(groupings)))
    return FlowceptionState(state_tuple(state), groupings, local_t, branchmask, flowmask, padmask)
end

Base.copy(Xₜ::FlowceptionState) = deepcopy(Xₜ)
Adapt.adapt_structure(to, Xₜ::FlowceptionState) = FlowceptionState(
    Adapt.adapt(to, Xₜ.state),
    Adapt.adapt(to, Xₜ.groupings);
    local_t = Adapt.adapt(to, Xₜ.local_t),
    branchmask = Adapt.adapt(to, Xₜ.branchmask),
    flowmask = Adapt.adapt(to, Xₜ.flowmask),
    padmask = Adapt.adapt(to, Xₜ.padmask),
)
Flowfusion.resolveprediction(a, Xₜ::FlowceptionState) = a

"""
    FlowceptionFlow(P, birth_sampler;
                    scheduler = linear_scheduler,
                    scheduler_derivative = linear_scheduler_derivative,
                    scheduler_inverse = linear_scheduler_inverse,
                    insertion_transform = x -> exp.(clamp.(x, -100, 11)),
                    split_transform = insertion_transform)

Flowception-style variable-length wrapper over a Flowfusion process or process
tuple `P`.

At sampling time, existing elements denoise according to their per-element
`local_t`, while the insertion head emits Poisson intensities for adjacent
insertions to the right. Inserted elements are initialized from
`birth_sampler`, typically the same source prior used to define `X₀`.

The scheduler fields implement the reveal schedule `κ`, its derivative, and its
inverse. The default is the linear scheduler from the Flowception paper.
"""
struct FlowceptionFlow{Proc,Birth,S,SD,SI,F} <: Process
    P::Proc
    birth_sampler::Birth
    scheduler::S
    scheduler_derivative::SD
    scheduler_inverse::SI
    insertion_transform::F
end

function FlowceptionFlow(P, birth_sampler;
        scheduler = linear_scheduler,
        scheduler_derivative = linear_scheduler_derivative,
        scheduler_inverse = linear_scheduler_inverse,
        insertion_transform = x -> exp.(clamp.(x, -100, 11)),
        split_transform = insertion_transform)
    return FlowceptionFlow(P, birth_sampler, scheduler, scheduler_derivative, scheduler_inverse, split_transform)
end

function scheduler_hazard(P::FlowceptionFlow, t)
    tc = clip01(t)
    tc >= one(tc) && return zero(tc)
    κ = P.scheduler(tc)
    return P.scheduler_derivative(tc) / max(one(tc) - κ, smalltime(tc))
end

sample_birth(P::FlowceptionFlow, ref) = applicable(P.birth_sampler, ref) ? P.birth_sampler(ref) : P.birth_sampler()

function normalized_birth(P::FlowceptionFlow, ref::Tuple)
    sample = sample_birth(P, ref)
    return sample isa Tuple ? sample : (sample,)
end

function make_birth_batch(P::FlowceptionFlow, refs)
    isempty(refs) && error("Cannot sample an empty birth batch.")
    births = normalized_birth.(Ref(P), refs)
    mask = ones(Bool, length(births), 1)
    return MaskedState.(Flowfusion.regroup([births]), (mask,), (mask,))
end

function original_visible_mask(P::FlowceptionFlow, X1::FlowceptionState, τg::T, nstart::Int) where T
    groups = vec(X1.groupings)
    target_flow = vec(X1.flowmask)
    target_pad = vec(X1.padmask)
    visible = falses(length(groups))
    local_t = zeros(T, length(groups))
    seen = Dict{Int,Int}()

    for i in eachindex(groups)
        target_pad[i] || continue
        if !target_flow[i]
            visible[i] = true
            local_t[i] = one(T)
            continue
        end
        g = groups[i]
        count = get(seen, g, 0)
        delay = count < nstart ? zero(T) : T(P.scheduler_inverse(rand(T)))
        seen[g] = count + 1
        τi = τg - delay
        if τi >= zero(T)
            visible[i] = true
            local_t[i] = clip01(τi)
        end
    end

    for g in unique(groups[target_pad])
        any(visible[(groups .== g) .& target_pad]) && continue
        error("Group $g has no visible frames at τg=$τg. Increase `nstart` or provide a visible context frame.")
    end

    return visible, local_t
end

function slot_targets(groups, visible, padmask)
    visible_inds = findall(visible .& padmask)
    counts = zeros(Int, length(visible_inds))
    for (k, i) in enumerate(visible_inds)
        g = groups[i]
        c = 0
        j = i + 1
        while j <= length(groups) && padmask[j] && groups[j] == g
            if visible[j]
                break
            end
            c += 1
            j += 1
        end
        counts[k] = c
    end
    return visible_inds, counts
end

function single_flowception_bridge(P::FlowceptionFlow, X1::FlowceptionState, τg::T, nstart::Int) where T
    visible, local_t_full = original_visible_mask(P, X1, τg, nstart)
    visible_inds, counts = slot_targets(vec(X1.groupings), visible, vec(X1.padmask))

    X1_elements = Flowfusion.element.((X1.state,), visible_inds)
    X0_elements = normalized_birth.(Ref(P), X1_elements)
    X0_batch = Flowfusion.regroup(X0_elements)
    X1_batch = Flowfusion.regroup(X1_elements)
    local_t = local_t_full[visible_inds]
    Xt_batch = bridge(P.P, X0_batch, X1_batch, local_t)

    groups = vec(X1.groupings)[visible_inds]
    branchmask = vec(X1.branchmask)[visible_inds]
    base_flowmask = vec(X1.flowmask)[visible_inds] .& (vec(local_t) .< one(T))

    return [
        (;
            Xt = Flowfusion.element(Xt_batch, i),
            X1anchor = X1_elements[i],
            t = τg,
            local_t = local_t[i],
            branchable = branchmask[i],
            flowable = base_flowmask[i],
            group = groups[i],
            insertions = counts[i],
        ) for i in eachindex(X1_elements)
    ]
end

"""
    flowception_bridge(P::FlowceptionFlow, X1s, times;
                       T = Float32,
                       nstart = 1,
                       maxlen = Inf)

Batch Flowception training bridge.

For each target `X1`, samples a global extended time `τg`, samples per-element
reveal delays according to the scheduler, removes not-yet-visible elements, and
bridges visible elements from the source prior to the target using their
per-element `local_t`.

The return value intentionally mirrors `branching_bridge`:

- `t`: the sampled global extended times.
- `Xt`: a batched `FlowceptionState`.
- `X1anchor`: batched visible targets aligned with `Xt.state`.
- `insertions_target`: number of missing elements to the right of each visible slot.
"""
function flowception_bridge(P::FlowceptionFlow, X1s, times; T = Float32, nstart = 1, maxlen = Inf)
    if times isa UnivariateDistribution
        times = rand(times, length(X1s))
    end
    times = clamp.(T.(times), zero(T), T(2))

    bridges = [single_flowception_bridge(P, X1s[i], times[i], nstart) for i in eachindex(X1s)]
    batch_maxlen = maximum(length.(bridges))
    batch_maxlen <= maxlen || error("Visible sequence length $batch_maxlen exceeds `maxlen=$maxlen`.")

    b = length(bridges)
    flowmask = falses(batch_maxlen, b)
    branchmask = falses(batch_maxlen, b)
    padmask = falses(batch_maxlen, b)
    groups = zeros(Int, batch_maxlen, b)
    local_t = zeros(T, batch_maxlen, b)
    insertion_counts = zeros(Int, batch_maxlen, b)

    for j in eachindex(bridges)
        for i in eachindex(bridges[j])
            flowmask[i, j] = bridges[j][i].flowable
            branchmask[i, j] = bridges[j][i].branchable
            padmask[i, j] = true
            groups[i, j] = bridges[j][i].group
            local_t[i, j] = bridges[j][i].local_t
            insertion_counts[i, j] = bridges[j][i].insertions
        end
    end

    Xt_batch = MaskedState.(Flowfusion.regroup([[x.Xt for x in bridge] for bridge in bridges]), (flowmask,), (padmask .& flowmask,))
    X1_batch = MaskedState.(Flowfusion.regroup([[x.X1anchor for x in bridge] for bridge in bridges]), (flowmask,), (padmask .& flowmask,))
    insertions_target = oftype.(local_t, insertion_counts)

    return (;
        t = times,
        Xt = FlowceptionState(Xt_batch, groups;
            local_t,
            branchmask,
            flowmask,
            padmask,
        ),
        X1anchor = X1_batch,
        insertions_target,
        splits_target = insertions_target,
    )
end

function flowception_component_step(P, Xₜ, X̂₁, s₁, s₂)
    return Flowfusion.step(P, Xₜ, X̂₁, s₁, s₂)
end

function flowception_component_step(P::Flowfusion.DistNoisyInterpolatingDiscreteFlow,
        Xₜ::MaskedState{<:DiscreteState{<:AbstractArray{<:Signed}}}, X̂₁logits, s₁, s₂)
    return flowception_component_step(P, Xₜ.S, X̂₁logits, s₁, s₂)
end

function flowception_component_step(P::Flowfusion.DistNoisyInterpolatingDiscreteFlow,
        Xₜ::DiscreteState{<:AbstractArray{<:Signed}}, X̂₁logits, s₁, s₂)
    X̂₁ = LogExpFunctions.softmax(X̂₁logits, dims = 1)
    probs = tensor(X̂₁)
    T = eltype(s₁)
    Δt = expand_like(s₂ .- s₁, probs)
    pu = T(1 / Xₜ.K)
    ϵ = T(1e-10)

    κ1_ = expand_like(Flowfusion.κ1(P, s₁), probs)
    κ2_ = expand_like(Flowfusion.κ2(P, s₁), probs)
    κ3_ = one(T) .- κ1_ .- κ2_

    dκ1_ = expand_like(Flowfusion.dκ1(P, s₁), probs)
    dκ2_ = expand_like(Flowfusion.dκ2(P, s₁), probs)
    dκ3_ = .- (dκ1_ .+ dκ2_)

    βt = dκ3_ ./ max.(κ3_, ϵ)
    ohXₜ = T.(tensor(onehot(Xₜ)))
    velo = (dκ1_ .- κ1_ .* βt) .* probs .+
        (dκ2_ .- κ2_ .* βt) .* pu .+
        βt .* ohXₜ

    raw_probs = ifelse.(isfinite.(ohXₜ .+ (Δt .* velo)), ohXₜ .+ (Δt .* velo), zero(T))
    raw_probs = max.(raw_probs, zero(T))
    totals = sum(raw_probs, dims = 1)
    bad = .!isfinite.(totals) .| (totals .<= ϵ)
    raw_probs = ifelse.(bad, ohXₜ, raw_probs)
    totals = ifelse.(bad, one(T), totals)
    newXₜ = CategoricalLikelihood(raw_probs ./ totals)
    return rand(newXₜ)
end

function Flowfusion.step(P::FlowceptionFlow, XₜFS::FlowceptionState, hat::Tuple, s₁::Real, s₂::Real)
    Xₜ = XₜFS.state
    local_t₁ = XₜFS.local_t
    T = eltype(local_t₁)
    Δg = T(s₂ - s₁)
    local_t₂ = clip01(local_t₁ .+ T.(XₜFS.flowmask) .* Δg)

    X1targets, insertion_logits = hat
    next_Xₜ = map(state_tuple(P.P), Xₜ, state_tuple(X1targets)) do Pi, Xi, X̂i
        flowception_component_step(Pi, Xi, X̂i, local_t₁, local_t₂)
    end
    Xₜ = Flowfusion.mask(next_Xₜ, Xₜ)

    insert_dt = max(min(T(s₂), one(T)) - min(T(s₁), one(T)), zero(T))
    safe_logits = ifelse.(isfinite.(insertion_logits), insertion_logits, zero(T))
    λ = max.(P.insertion_transform.(safe_logits), zero(T))
    ρ = scheduler_hazard(P, T(s₁))
    counts = vec(XₜFS.branchmask .* XₜFS.padmask .* rand.(Poisson.(insert_dt .* ρ .* λ)))

    current_flowmask = XₜFS.padmask .& (local_t₂ .< one(T))
    if sum(counts) == 0
        state = map(Xₜ) do Xi
            MaskedState(Flowfusion.unmask(Xi), current_flowmask, current_flowmask)
        end
        return FlowceptionState(state, XₜFS.groupings;
            local_t = local_t₂,
            branchmask = XₜFS.branchmask,
            flowmask = current_flowmask,
            padmask = XₜFS.padmask,
        )
    end

    refs = Tuple[]
    for i in 1:size(XₜFS.groupings, 1)
        n = counts[i]
        n <= 0 && continue
        ref = Flowfusion.element(Xₜ, i, 1)
        append!(refs, ntuple(_ -> ref, n))
    end

    births = normalized_birth.(Ref(P), refs)
    current_length = size(XₜFS.groupings, 1)
    new_length = current_length + sum(counts)
    example = Flowfusion.element(Xₜ, 1, 1)
    newstates = Tuple(Flowfusion.zerostate(example[i], new_length, 1) for i in eachindex(example))
    groupings = similar(XₜFS.groupings, new_length, 1)
    branchmask = similar(XₜFS.branchmask, new_length, 1)
    local_t = similar(local_t₂, new_length, 1)
    flowmask = similar(current_flowmask, new_length, 1)
    padmask = similar(XₜFS.padmask, new_length, 1)

    birth_index = 1
    dst = 1
    for i in 1:current_length
        current = Flowfusion.element(Xₜ, i, 1)
        for s in eachindex(current)
            Flowfusion.element(ForwardBackward.tensor(newstates[s]), dst, 1) .= ForwardBackward.tensor(current[s])
        end
        groupings[dst, 1] = XₜFS.groupings[i, 1]
        branchmask[dst, 1] = XₜFS.branchmask[i, 1]
        local_t[dst, 1] = local_t₂[i, 1]
        flowmask[dst, 1] = current_flowmask[i, 1]
        padmask[dst, 1] = true
        dst += 1
        for _ in 1:counts[i]
            birth = births[birth_index]
            for s in eachindex(birth)
                Flowfusion.element(ForwardBackward.tensor(newstates[s]), dst, 1) .= ForwardBackward.tensor(birth[s])
            end
            groupings[dst, 1] = XₜFS.groupings[i, 1]
            branchmask[dst, 1] = XₜFS.branchmask[i, 1]
            local_t[dst, 1] = zero(T)
            flowmask[dst, 1] = true
            padmask[dst, 1] = true
            dst += 1
            birth_index += 1
        end
    end

    state = MaskedState.(newstates, (flowmask,), (flowmask,))
    return FlowceptionState(state, groupings;
        local_t,
        branchmask,
        flowmask,
        padmask,
    )
end

Flowfusion.scalefloss(P::FlowceptionFlow, t::Real, pow = 2, eps = typeof(t)(0.05)) = one(t) / (((one(t) + eps) - clip01(t)) ^ pow)
Flowfusion.scalefloss(P::FlowceptionFlow, t::AbstractArray, pow = 2, eps = eltype(t)(0.05)) = 1 ./ ((1 + eps) .- clip01(t)) .^ pow
Flowfusion.floss(P::FlowceptionFlow, X̂₁, X₁, mask, c) = Flowfusion.scaledmaskedmean(sbpl(P.insertion_transform(X̂₁), X₁), c, mask)
