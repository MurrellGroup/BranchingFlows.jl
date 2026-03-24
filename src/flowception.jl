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

smalltime(x::Real) = oftype(x, 1e-6)
clip01(x::Real) = clamp(x, zero(x), one(x))
clip01(x::AbstractArray) = clamp.(x, zero(eltype(x)), one(eltype(x)))
expand_like(x, ref) = reshape(x, ntuple(_ -> 1, ndims(ref) - ndims(x))..., size(x)...)
prefix_token_axis(x::Real, n) = x
prefix_token_axis(x::AbstractVector, n) = length(x) == 1 ? x : view(x, 1:n)
prefix_token_axis(x::AbstractArray, n) = size(x, 1) == 1 ? x : view(x, 1:n, ntuple(_ -> Colon(), ndims(x) - 1)...)
masked_slot_sum(loss, c, mask) = sum(loss .* prefix_token_axis(c, size(loss, 1)) .* mask)

abstract type AbstractFlowceptionProcess <: Process end
abstract type AbstractFlowceptionRevealOrder end

"""
    AbstractSeededRevealTarget

Target family for `SeededRevealOrder` insertion supervision.

These targets apply only to the structured reveal-order bridge used by
`SeededRevealOrder`. They change how hidden residues are converted into
directional slot targets for `DirectionalFlowceptionFlow`.
"""
abstract type AbstractSeededRevealTarget end
struct IndependentRevealOrder <: AbstractFlowceptionRevealOrder end

"""
    CountRevealTarget()

Legacy count target for structured reveal order.

This reuses the independent-reveal slot-count target even when the bridge
samples an ordered reveal policy. It is kept for backward compatibility and
benchmarking. For `SeededRevealOrder`, this target does not match the
structured conditional generator.
"""
struct CountRevealTarget <: AbstractSeededRevealTarget end

"""
    SparseRevealTarget()

Pathwise structured target for `SeededRevealOrder`.

For each group, all insertion mass is placed on the current physical slot that
contains the next unrevealed residue in the sampled latent reveal order, scaled
by the number of unrevealed residues still remaining in that group.
"""
struct SparseRevealTarget <: AbstractSeededRevealTarget end

"""
    RaoBlackwellizedRevealTarget()

Rao-Blackwellized structured target for `SeededRevealOrder`.

For each group, the insertion target is averaged over the conditional
distribution of the next unrevealed residue under the current Gumbel-perturbed
reveal policy, then mapped onto current physical slots and scaled by the
number of unrevealed residues still remaining in the group.
"""
struct RaoBlackwellizedRevealTarget <: AbstractSeededRevealTarget end
struct DefaultSeedPriority end
struct DefaultRevealPriority end

struct SeededRevealOrder{T,F,G,H} <: AbstractFlowceptionRevealOrder
    temperature::T
    seed_priority::F
    reveal_priority::G
    target::H
end

"""
    SeededRevealOrder(; temperature = 0.25f0,
                        seed_priority = DefaultSeedPriority(),
                        reveal_priority = DefaultRevealPriority(),
                        target = RaoBlackwellizedRevealTarget())

Structured reveal-order policy for Flowception bridges.

`SeededRevealOrder` first chooses visible seed residues in each group and then
samples the remaining reveal order by repeatedly selecting a hidden residue
using a deterministic score plus optional Gumbel noise. By default it uses
`RaoBlackwellizedRevealTarget()`; the `target` argument controls how that
structured reveal policy is converted into insertion supervision:

- `CountRevealTarget()`: legacy count target
- `SparseRevealTarget()`: pathwise next-slot target
- `RaoBlackwellizedRevealTarget()`: conditional expectation of the next-slot target

Non-count targets are currently implemented only for
`DirectionalFlowceptionFlow`.
"""
SeededRevealOrder(; temperature = 0.25f0, seed_priority = DefaultSeedPriority(), reveal_priority = DefaultRevealPriority(), target = RaoBlackwellizedRevealTarget()) =
    SeededRevealOrder(temperature, seed_priority, reveal_priority, target)

flowception_total_time(P::AbstractFlowceptionProcess, ::Type{T}) where T = T(P.total_time)
flowception_total_time(P::AbstractFlowceptionProcess) = P.total_time
flowception_insertion_horizon(P::AbstractFlowceptionProcess, ::Type{T}) where T = max(flowception_total_time(P, T) - one(T), zero(T))
flowception_insertion_horizon(P::AbstractFlowceptionProcess) = max(P.total_time - 1, 0)

function flowception_insertion_phase(P::AbstractFlowceptionProcess, t::Real)
    T = typeof(t)
    horizon = flowception_insertion_horizon(P, T)
    horizon <= zero(T) && return one(T)
    return clip01(t / horizon)
end

function flowception_insertion_phase(P::AbstractFlowceptionProcess, t::AbstractArray)
    T = eltype(t)
    horizon = flowception_insertion_horizon(P, T)
    horizon <= zero(T) && return one.(t)
    return clip01(t ./ horizon)
end

function flowception_insert_dt(P::AbstractFlowceptionProcess, s₁::Real, s₂::Real, ::Type{T}) where T
    horizon = flowception_insertion_horizon(P, T)
    return max(min(T(s₂), horizon) - min(T(s₁), horizon), zero(T))
end

"""
    FlowceptionState(state, groupings;
                     local_t = ones(Float32, size(groupings)),
                     branchmask = ones(Bool, size(groupings)),
                     flowmask = ones(Bool, size(groupings)),
                     padmask = ones(Bool, size(groupings)))

Variable-length sequence state used by `FlowceptionFlow`.

`state` is stored as a tuple of masked or unmasked Flowfusion states. Explicit
`MaskedState` components keep their own `cmask/lmask`; plain components use
`flowmask` as their effective component `cmask` internally. The
sequence-level bookkeeping matches `BranchingState` closely, with one extra
field:

- `local_t`: per-element denoising times in `[0, 1]`.

Semantics:

- `branchmask`: whether insertions are permitted to the right of an element.
- `flowmask`: whether the element should keep denoising. For plain
  non-`MaskedState` components, this is also the effective component `cmask`.
- `padmask`: whether the sequence position is valid.

Unbatched targets are typically vectors, while batched bridge outputs use
matrices of shape `(L, B)`.

Invariant:
- For every live element (`padmask=1`) with `branchmask=1`, every component
  must be designable. Explicit `MaskedState` components must therefore have
  `cmask=1` there, and plain components must have `flowmask=1`.
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
    validate_branchmask_cmask(state, branchmask, flowmask, padmask; context = "FlowceptionState")
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
                    total_time = 2,
                    split_transform = insertion_transform)

Flowception-style variable-length wrapper over a Flowfusion process or process
tuple `P`.

At sampling time, existing elements denoise according to their per-element
`local_t`, while the insertion head emits Poisson intensities for adjacent
insertions to the right. Inserted elements are initialized from
`birth_sampler`, typically the same source prior used to define `X₀`.

The scheduler fields implement the reveal schedule `κ`, its derivative, and its
inverse over the insertion/reveal phase `[0, total_time - 1]`. The final unit
of global time is reserved for already-visible elements to complete one full
unit of local denoising.
"""
struct FlowceptionFlow{Proc,Birth,S,SD,SI,F,TW,RO} <: AbstractFlowceptionProcess
    P::Proc
    birth_sampler::Birth
    scheduler::S
    scheduler_derivative::SD
    scheduler_inverse::SI
    insertion_transform::F
    total_time::TW
    reveal_order::RO
end

function FlowceptionFlow(P, birth_sampler;
        scheduler = linear_scheduler,
        scheduler_derivative = linear_scheduler_derivative,
        scheduler_inverse = linear_scheduler_inverse,
        insertion_transform = x -> exp.(clamp.(x, -100, 11)),
        total_time = 2,
        reveal_order = IndependentRevealOrder(),
        split_transform = insertion_transform)
    total_time >= 1 || error("Flowception total_time must be at least 1 so each element has one unit of local denoising time.")
    return FlowceptionFlow(P, birth_sampler, scheduler, scheduler_derivative, scheduler_inverse, split_transform, total_time, reveal_order)
end

"""
    DirectionalFlowceptionFlow(P, birth_sampler;
                               scheduler = linear_scheduler,
                               scheduler_derivative = linear_scheduler_derivative,
                               scheduler_inverse = linear_scheduler_inverse,
                               insertion_transform = x -> exp.(clamp.(x, -100, 11)),
                               total_time = 2,
                               split_transform = insertion_transform)

Flowception-style wrapper that keeps the same `FlowceptionState`/local-time
machinery as `FlowceptionFlow`, but expects a directional insertion head with
shape `(2, L, B)` or a `(left, right)` tuple. The first channel predicts
insertions to the left of each visible element, the second to the right.

Interior within-group gaps are pooled directly from adjacent token predictions
using `groupings`, with no explicit slot tensor:

- same-group interior gap `(i, i+1)`: pooled from `right[i]` and `left[i+1]`
- group start: uses `left[first]`
- group end: uses `right[last]`

This provides a separate reference implementation for bidirectional insertions
without changing the original `FlowceptionFlow` behavior. As for
`FlowceptionFlow`, `total_time` sets the global horizon while each individual
element still receives exactly one unit of local denoising time.
"""
struct DirectionalFlowceptionFlow{Proc,Birth,S,SD,SI,F,TW,RO} <: AbstractFlowceptionProcess
    P::Proc
    birth_sampler::Birth
    scheduler::S
    scheduler_derivative::SD
    scheduler_inverse::SI
    insertion_transform::F
    total_time::TW
    reveal_order::RO
end

function DirectionalFlowceptionFlow(P, birth_sampler;
        scheduler = linear_scheduler,
        scheduler_derivative = linear_scheduler_derivative,
        scheduler_inverse = linear_scheduler_inverse,
        insertion_transform = x -> exp.(clamp.(x, -100, 11)),
        total_time = 2,
        reveal_order = IndependentRevealOrder(),
        split_transform = insertion_transform)
    total_time >= 1 || error("Flowception total_time must be at least 1 so each element has one unit of local denoising time.")
    return DirectionalFlowceptionFlow(P, birth_sampler, scheduler, scheduler_derivative, scheduler_inverse, split_transform, total_time, reveal_order)
end

function scheduler_hazard(P::AbstractFlowceptionProcess, t)
    T = typeof(t)
    horizon = flowception_insertion_horizon(P, T)
    horizon <= zero(T) && return zero(T)
    tc = flowception_insertion_phase(P, t)
    tc >= one(tc) && return zero(tc)
    κ = P.scheduler(tc)
    return P.scheduler_derivative(tc) / (horizon * max(one(tc) - κ, smalltime(tc)))
end

sample_birth(P::AbstractFlowceptionProcess, ref) = applicable(P.birth_sampler, ref) ? P.birth_sampler(ref) : P.birth_sampler()

function normalized_birth(P::AbstractFlowceptionProcess, ref::Tuple)
    sample = sample_birth(P, ref)
    return sample isa Tuple ? sample : (sample,)
end

function make_birth_batch(P::AbstractFlowceptionProcess, refs)
    isempty(refs) && error("Cannot sample an empty birth batch.")
    births = normalized_birth.(Ref(P), refs)
    mask = trues(length(births))
    return batch_masked(births, mask, mask)
end

gumbel_noise(::Type{T}) where T = -log(-log(rand(T)))

function nearest_seed_distances(seedmask::AbstractVector{Bool})
    n = length(seedmask)
    infdist = max(n + 1, 1)
    d = fill(infdist, n)
    last_seed = -infdist
    for i in eachindex(seedmask)
        if seedmask[i]
            last_seed = i
            d[i] = 0
        else
            d[i] = i - last_seed
        end
    end
    last_seed = 2 * infdist
    for i in length(seedmask):-1:1
        if seedmask[i]
            last_seed = i
        else
            d[i] = min(d[i], last_seed - i)
        end
    end
    return d
end

function group_segment_stop(groups, padmask, start)
    stop = start
    while stop <= length(groups) && padmask[stop] && groups[stop] == groups[start]
        stop += 1
    end
    return stop - 1
end

function independent_visible_mask(P::AbstractFlowceptionProcess, X1::FlowceptionState, τg::T, nstart::Int) where T
    groups = vec(X1.groupings)
    target_flow = vec(X1.flowmask)
    target_pad = vec(X1.padmask)
    visible = falses(length(groups))
    local_t = zeros(T, length(groups))
    seen = Dict{Int,Int}()
    insertion_horizon = flowception_insertion_horizon(P, T)

    for i in eachindex(groups)
        target_pad[i] || continue
        if !target_flow[i]
            visible[i] = true
            local_t[i] = one(T)
            continue
        end
        g = groups[i]
        count = get(seen, g, 0)
        delay = count < nstart ? zero(T) : insertion_horizon * T(P.scheduler_inverse(rand(T)))
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

function default_seed_scores(design_local::AbstractVector{Int}, fixed_local::AbstractVector{Int}, ::Type{T}) where T
    isempty(design_local) && return T[]
    center = T(first(design_local) + last(design_local)) / T(2)
    if isempty(fixed_local)
        centrality = abs.(T.(design_local) .- center)
        return centrality .+ T(1e-4) .* T.(design_local)
    end

    fixed_seedmask = falses(maximum(vcat(design_local, fixed_local)))
    fixed_seedmask[fixed_local] .= true
    fixed_distance = nearest_seed_distances(fixed_seedmask)
    centrality = abs.(T.(design_local) .- center)
    return T.(fixed_distance[design_local]) .+ T(1e-3) .* centrality .+ T(1e-4) .* T.(design_local)
end

seed_priority_scores(::DefaultSeedPriority, X1::FlowceptionState, seg, design_local::AbstractVector{Int}, fixed_local::AbstractVector{Int}, ::Type{T}) where T =
    default_seed_scores(design_local, fixed_local, T)

function seed_priority_scores(seed_priority, X1::FlowceptionState, seg, design_local::AbstractVector{Int}, fixed_local::AbstractVector{Int}, ::Type{T}) where T
    scores = seed_priority(X1, seg, design_local, fixed_local, T)
    length(scores) == length(design_local) || error("Seeded seed priority returned $(length(scores)) scores for $(length(design_local)) designable positions.")
    return T.(scores)
end

function default_reveal_cache(design_local::AbstractVector{Int}, fixed_local::AbstractVector{Int}, ::Type{T}) where T
    isempty(design_local) && return (; design_design = zeros(T, 0, 0), design_fixed = zeros(T, 0, 0), bias = T[])
    design_positions = T.(design_local)
    design_design = abs.(reshape(design_positions, :, 1) .- reshape(design_positions, 1, :))
    design_fixed = if isempty(fixed_local)
        zeros(T, length(design_local), 0)
    else
        fixed_positions = T.(fixed_local)
        abs.(reshape(design_positions, :, 1) .- reshape(fixed_positions, 1, :))
    end
    center = T(first(design_local) + last(design_local)) / T(2)
    bias = abs.(design_positions .- center) .+ T(1e-4) .* design_positions
    return (; design_design, design_fixed, bias)
end

reveal_priority_cache(::DefaultRevealPriority, X1::FlowceptionState, seg, design_local::AbstractVector{Int}, fixed_local::AbstractVector{Int}, ::Type{T}) where T =
    default_reveal_cache(design_local, fixed_local, T)

function reveal_priority_cache(reveal_priority, X1::FlowceptionState, seg, design_local::AbstractVector{Int}, fixed_local::AbstractVector{Int}, ::Type{T}) where T
    cache = reveal_priority(X1, seg, design_local, fixed_local, T)
    hasproperty(cache, :design_design) || error("Reveal priority must return a cache with a `design_design` field.")
    hasproperty(cache, :design_fixed) || error("Reveal priority must return a cache with a `design_fixed` field.")
    hasproperty(cache, :bias) || error("Reveal priority must return a cache with a `bias` field.")
    design_design = T.(Array(cache.design_design))
    design_fixed = T.(Array(cache.design_fixed))
    bias = vec(T.(Array(cache.bias)))
    ndims(design_fixed) == 1 && (design_fixed = reshape(design_fixed, :, 1))
    ndesign = length(design_local)
    size(design_design) == (ndesign, ndesign) || error("Reveal priority returned a design-design matrix of size $(size(design_design)) for $ndesign designable positions.")
    size(design_fixed, 1) == ndesign || error("Reveal priority returned a design-fixed matrix with $(size(design_fixed, 1)) rows for $ndesign designable positions.")
    length(bias) == ndesign || error("Reveal priority returned $(length(bias)) bias entries for $ndesign designable positions.")
    return (; design_design, design_fixed, bias)
end

function seeded_reveal_score_state(cache, design_local::AbstractVector{Int}, fixed_local::AbstractVector{Int}, visible_design_local::AbstractVector{Int}, ::Type{T}) where T
    isempty(design_local) && return (; local_to_design = Int[], min_scores = T[])
    local_to_design = zeros(Int, maximum(design_local))
    local_to_design[design_local] .= eachindex(design_local)
    visible_design = isempty(visible_design_local) ? Int[] : filter(!iszero, local_to_design[visible_design_local])
    min_scores = if size(cache.design_fixed, 2) == 0
        fill(T(Inf), length(design_local))
    else
        vec(minimum(cache.design_fixed, dims = 2))
    end
    if !isempty(visible_design)
        min_scores = min.(min_scores, vec(minimum(cache.design_design[:, visible_design], dims = 2)))
    end
    return (; local_to_design, min_scores)
end

function reveal_choice_probabilities(scores::AbstractVector{T}, temperature::T) where T
    isempty(scores) && return T[]
    if !any(isfinite, scores)
        return fill(inv(T(length(scores))), length(scores))
    end
    if temperature <= zero(T)
        min_score = minimum(scores)
        winners = scores .== min_score
        probs = zeros(T, length(scores))
        probs[winners] .= inv(T(count(winners)))
        return probs
    end
    shifted = scores .- minimum(scores)
    weights = exp.(-shifted ./ temperature)
    total = sum(weights)
    total > zero(T) || return fill(inv(T(length(scores))), length(scores))
    return weights ./ total
end

function choose_seed_positions(X1::FlowceptionState, seg, design_local::AbstractVector{Int}, fixed_local::AbstractVector{Int}, nstart::Int, policy::SeededRevealOrder, ::Type{T}) where T
    nseed = min(nstart, length(design_local))
    nseed == 0 && return Int[]
    scores = seed_priority_scores(policy.seed_priority, X1, seg, design_local, fixed_local, T)
    order = sortperm(scores)
    return collect(design_local[order[1:nseed]])
end

function choose_reveal_positions(X1::FlowceptionState, seg, design_local::AbstractVector{Int}, fixed_local::AbstractVector{Int}, seed_local::AbstractVector{Int}, policy::SeededRevealOrder, ::Type{T}) where T
    isempty(design_local) && return Int[]
    cache = reveal_priority_cache(policy.reveal_priority, X1, seg, design_local, fixed_local, T)
    score_state = seeded_reveal_score_state(cache, design_local, fixed_local, seed_local, T)
    local_to_design = score_state.local_to_design
    seed_design = isempty(seed_local) ? Int[] : filter(!iszero, local_to_design[seed_local])
    min_scores = score_state.min_scores

    remaining = trues(length(design_local))
    remaining[seed_design] .= false
    reveal_order = Int[]
    while any(remaining)
        scores = cache.bias .+ min_scores
        scores[.!remaining] .= T(Inf)
        if policy.temperature > zero(policy.temperature)
            active = findall(remaining)
            scores[active] .+= T(policy.temperature) .* T[gumbel_noise(T) for _ in active]
        end
        chosen = argmin(scores)
        push!(reveal_order, design_local[chosen])
        remaining[chosen] = false
        min_scores = min.(min_scores, cache.design_design[:, chosen])
    end
    return reveal_order
end

function seeded_group_plan(P::AbstractFlowceptionProcess, X1::FlowceptionState, seg, τg::T, nstart::Int, policy::SeededRevealOrder) where T
    local_flow = vec(X1.flowmask[seg])
    fixed_local = findall(.!local_flow)
    design_local = findall(local_flow)
    segment_visible = falses(length(seg))
    segment_local_t = zeros(T, length(seg))
    horizon = flowception_insertion_horizon(P, T)

    if !isempty(fixed_local)
        segment_visible[fixed_local] .= true
        segment_local_t[fixed_local] .= one(T)
    end

    seed_local = Int[]
    reveal_local = Int[]
    if !isempty(design_local)
        if isempty(fixed_local) && nstart <= 0
            error("Group $(X1.groupings[first(seg)]) has no fixed context and `nstart=0` under `SeededRevealOrder`. Increase `nstart` or provide fixed visible context.")
        end

        seed_local = choose_seed_positions(X1, seg, design_local, fixed_local, nstart, policy, T)
        if !isempty(seed_local)
            segment_visible[seed_local] .= true
            segment_local_t[seed_local] .= clip01(τg)
        end

        if length(seed_local) < length(design_local)
            reveal_local = choose_reveal_positions(X1, seg, design_local, fixed_local, seed_local, policy, T)
            reveal_levels = sort!(rand(T, length(reveal_local)))
            for (rank, local_idx) in enumerate(reveal_local)
                delay = horizon * T(P.scheduler_inverse(reveal_levels[rank]))
                if τg >= delay
                    segment_visible[local_idx] = true
                    segment_local_t[local_idx] = clip01(τg - delay)
                end
            end
        end
    end

    visible_design_local = [loc for loc in design_local if segment_visible[loc]]
    hidden_local = [loc for loc in design_local if !segment_visible[loc]]
    return (;
        seg,
        visible = segment_visible,
        local_t = segment_local_t,
        fixed_local,
        design_local,
        seed_local,
        reveal_local,
        visible_design_local,
        hidden_local,
    )
end

function seeded_visible_plans(P::AbstractFlowceptionProcess, X1::FlowceptionState, τg::T, nstart::Int, policy::SeededRevealOrder) where T
    groups = vec(X1.groupings)
    target_pad = vec(X1.padmask)
    visible = falses(length(groups))
    local_t = zeros(T, length(groups))
    plans = NamedTuple[]

    start = firstindex(groups)
    while start <= length(groups)
        if !target_pad[start]
            start += 1
            continue
        end
        stop = group_segment_stop(groups, target_pad, start)
        seg = start:stop
        plan = seeded_group_plan(P, X1, seg, τg, nstart, policy)
        visible[seg] .= plan.visible
        local_t[seg] .= plan.local_t
        push!(plans, plan)
        start = stop + 1
    end

    return plans, visible, local_t
end

function directional_slot_targets_from_hidden_weights(visible::AbstractVector{Bool}, hidden_weights::AbstractVector{T}) where T
    visible_local = findall(visible)
    isempty(visible_local) && return T[], T[]
    left_target = zeros(T, length(visible_local))
    right_target = zeros(T, length(visible_local))
    for local_idx in eachindex(hidden_weights)
        weight = hidden_weights[local_idx]
        iszero(weight) && continue
        slot_ix = searchsortedlast(visible_local, local_idx)
        if slot_ix == 0
            left_target[1] += weight
        else
            right_target[slot_ix] += weight
        end
    end
    return left_target, right_target
end

function seeded_hidden_weights(::CountRevealTarget, plan, X1::FlowceptionState, policy::SeededRevealOrder, ::Type{T}) where T
    weights = zeros(T, length(plan.visible))
    weights[plan.hidden_local] .= one(T)
    return weights
end

function seeded_hidden_weights(::SparseRevealTarget, plan, X1::FlowceptionState, policy::SeededRevealOrder, ::Type{T}) where T
    weights = zeros(T, length(plan.visible))
    isempty(plan.hidden_local) && return weights
    next_hidden = nothing
    for loc in plan.reveal_local
        if !plan.visible[loc]
            next_hidden = loc
            break
        end
    end
    isnothing(next_hidden) && return weights
    weights[next_hidden] = T(length(plan.hidden_local))
    return weights
end

function seeded_hidden_weights(::RaoBlackwellizedRevealTarget, plan, X1::FlowceptionState, policy::SeededRevealOrder, ::Type{T}) where T
    weights = zeros(T, length(plan.visible))
    isempty(plan.hidden_local) && return weights
    cache = reveal_priority_cache(policy.reveal_priority, X1, plan.seg, plan.design_local, plan.fixed_local, T)
    score_state = seeded_reveal_score_state(cache, plan.design_local, plan.fixed_local, plan.visible_design_local, T)
    remaining_design = score_state.local_to_design[plan.hidden_local]
    scores = cache.bias[remaining_design] .+ score_state.min_scores[remaining_design]
    probs = reveal_choice_probabilities(scores, T(policy.temperature))
    weights[plan.hidden_local] .= T(length(plan.hidden_local)) .* probs
    return weights
end

function seeded_visible_mask(P::AbstractFlowceptionProcess, X1::FlowceptionState, τg::T, nstart::Int, policy::SeededRevealOrder) where T
    groups = vec(X1.groupings)
    target_pad = vec(X1.padmask)
    _, visible, local_t = seeded_visible_plans(P, X1, τg, nstart, policy)

    for g in unique(groups[target_pad])
        any(visible[(groups .== g) .& target_pad]) && continue
        error("Group $g has no visible frames at τg=$τg. Increase `nstart` or provide a visible context frame.")
    end

    return visible, local_t
end

original_visible_mask(P::AbstractFlowceptionProcess, X1::FlowceptionState, τg::T, nstart::Int) where T =
    original_visible_mask(P, X1, τg, nstart, P.reveal_order)

original_visible_mask(P::AbstractFlowceptionProcess, X1::FlowceptionState, τg::T, nstart::Int, ::IndependentRevealOrder) where T =
    independent_visible_mask(P, X1, τg, nstart)

original_visible_mask(P::AbstractFlowceptionProcess, X1::FlowceptionState, τg::T, nstart::Int, policy::SeededRevealOrder) where T =
    seeded_visible_mask(P, X1, τg, nstart, policy)

function directional_heads(x::Tuple{<:AbstractArray,<:AbstractArray})
    return x
end

function directional_heads(x::NamedTuple{(:left, :right)})
    return x.left, x.right
end

function directional_heads(x::AbstractArray)
    size(x, 1) == 2 || error("Directional insertions must have size 2 along the first dimension.")
    return selectdim(x, 1, 1), selectdim(x, 1, 2)
end

function directional_slot_masks(groupings, padmask)
    tails = ntuple(_ -> Colon(), ndims(groupings) - 1)
    same_prev = similar(padmask, Bool)
    same_next = similar(padmask, Bool)
    fill!(same_prev, false)
    fill!(same_next, false)

    if size(groupings, 1) > 1
        same_prev[2:end, tails...] .=
            padmask[2:end, tails...] .&
            padmask[1:end-1, tails...] .&
            (groupings[2:end, tails...] .== groupings[1:end-1, tails...])
        same_next[1:end-1, tails...] .= same_prev[2:end, tails...]
    end

    group_start = padmask .& .!same_prev
    group_end = padmask .& .!same_next
    return (; same_prev, same_next, group_start, group_end)
end

function directional_insertion_masks(groupings, branchmask, padmask)
    slot_masks = directional_slot_masks(groupings, padmask)
    axes_tail = ntuple(_ -> Colon(), ndims(branchmask) - 1)
    interior_mask = similar(branchmask, Bool, max(size(groupings, 1) - 1, 0), size(branchmask)[2:end]...)
    fill!(interior_mask, false)
    if size(groupings, 1) > 1
        interior_mask .=
            slot_masks.same_next[1:end-1, axes_tail...] .&
            branchmask[1:end-1, axes_tail...] .&
            branchmask[2:end, axes_tail...]
    end
    return (;
        slot_masks...,
        left_mask = slot_masks.group_start .& branchmask,
        right_mask = slot_masks.group_end .& branchmask,
        interior_mask,
    )
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

function directional_slot_targets(groups, visible, padmask)
    visible_inds = findall(visible .& padmask)
    isempty(visible_inds) && return visible_inds, Int[], Int[]

    positions = collect(eachindex(groups))
    slot_masks = directional_slot_masks(groups, padmask)
    group_starts = accumulate(max, ifelse.(slot_masks.group_start, positions, zero(Int)))
    group_ends = reverse(accumulate(min, reverse(ifelse.(slot_masks.group_end, positions, length(groups) + 1))))

    visible_groups = groups[visible_inds]
    same_prev_visible = falses(length(visible_inds))
    same_next_visible = falses(length(visible_inds))
    if length(visible_inds) > 1
        same_prev_visible[2:end] .= visible_groups[2:end] .== visible_groups[1:end-1]
        same_next_visible[1:end-1] .= same_prev_visible[2:end]
    end

    prev_visible = similar(visible_inds)
    next_visible = similar(visible_inds)
    prev_visible[1] = visible_inds[1]
    next_visible[end] = visible_inds[end]
    if length(visible_inds) > 1
        prev_visible[2:end] .= visible_inds[1:end-1]
        next_visible[1:end-1] .= visible_inds[2:end]
    end

    left_anchor = ifelse.(same_prev_visible, prev_visible, group_starts[visible_inds])
    right_anchor = ifelse.(same_next_visible, next_visible, group_ends[visible_inds])
    left_counts = visible_inds .- left_anchor .- Int.(same_prev_visible)
    right_counts = right_anchor .- visible_inds .- Int.(same_next_visible)
    return visible_inds, left_counts, right_counts
end

"""
    flowception_visible_payload(P, X1, visible_inds, local_t_full)

Materialize the visible subset of one `FlowceptionState` for bridge training.

Explicit component masks are preserved on extraction. Plain components are
wrapped using `X1.flowmask` as `cmask` and `X1.padmask & X1.flowmask` as
`lmask` before batching and bridging.
"""
function flowception_visible_payload(P::AbstractFlowceptionProcess, X1::FlowceptionState, visible_inds, local_t_full)
    X1_elements = [masked_element(X1.state, X1.flowmask, X1.padmask .& X1.flowmask, i) for i in visible_inds]
    X0_elements = normalized_birth.(Ref(P), X1_elements)
    X0_mask = trues(length(X0_elements))
    X0_batch = batch_masked(X0_elements, X0_mask, X0_mask)
    X1_batch = batch_masked(X1_elements)
    local_t = local_t_full[visible_inds]
    Xt_batch = bridge(P.P, X0_batch, X1_batch, local_t)

    T = eltype(local_t)
    groups = vec(X1.groupings)[visible_inds]
    branchmask = vec(X1.branchmask)[visible_inds]
    base_flowmask = vec(X1.flowmask)[visible_inds] .& (vec(local_t) .< one(T))
    return (; Xt_batch, X1_elements, local_t, groups, branchmask, base_flowmask)
end

function seeded_directional_slot_targets(P::DirectionalFlowceptionFlow, X1::FlowceptionState, τg::T, nstart::Int, policy::SeededRevealOrder) where T
    plans, visible, local_t_full = seeded_visible_plans(P, X1, τg, nstart, policy)
    visible_inds = findall(visible .& vec(X1.padmask))
    left_counts = zeros(T, length(visible_inds))
    right_counts = zeros(T, length(visible_inds))
    offset = 0
    for plan in plans
        group_nvisible = count(identity, plan.visible)
        group_nvisible == 0 && continue
        hidden_weights = seeded_hidden_weights(policy.target, plan, X1, policy, T)
        group_left, group_right = directional_slot_targets_from_hidden_weights(plan.visible, hidden_weights)
        left_counts[offset+1:offset+group_nvisible] .= group_left
        right_counts[offset+1:offset+group_nvisible] .= group_right
        offset += group_nvisible
    end
    return visible, local_t_full, visible_inds, left_counts, right_counts
end

function single_flowception_bridge(P::FlowceptionFlow, X1::FlowceptionState, τg::T, nstart::Int) where T
    if P.reveal_order isa SeededRevealOrder && !(P.reveal_order.target isa CountRevealTarget)
        error("Non-count structured reveal targets are currently implemented only for `DirectionalFlowceptionFlow`.")
    end
    visible, local_t_full = original_visible_mask(P, X1, τg, nstart)
    visible_inds, counts = slot_targets(vec(X1.groupings), visible, vec(X1.padmask))
    payload = flowception_visible_payload(P, X1, visible_inds, local_t_full)

    return [
        (;
            Xt = masked_element(payload.Xt_batch, i),
            X1anchor = payload.X1_elements[i],
            t = τg,
            local_t = payload.local_t[i],
            branchable = payload.branchmask[i],
            flowable = payload.base_flowmask[i],
            group = payload.groups[i],
            insertions = counts[i],
        ) for i in eachindex(payload.X1_elements)
    ]
end

function single_directional_flowception_bridge(P::DirectionalFlowceptionFlow, X1::FlowceptionState, τg::T, nstart::Int) where T
    if P.reveal_order isa SeededRevealOrder
        visible, local_t_full, visible_inds, left_counts, right_counts = seeded_directional_slot_targets(P, X1, τg, nstart, P.reveal_order)
    else
        visible, local_t_full = original_visible_mask(P, X1, τg, nstart)
        visible_inds, left_counts, right_counts = directional_slot_targets(vec(X1.groupings), visible, vec(X1.padmask))
    end
    payload = flowception_visible_payload(P, X1, visible_inds, local_t_full)

    return [
        (;
            Xt = masked_element(payload.Xt_batch, i),
            X1anchor = payload.X1_elements[i],
            t = τg,
            local_t = payload.local_t[i],
            branchable = payload.branchmask[i],
            flowable = payload.base_flowmask[i],
            group = payload.groups[i],
            left_insertions = left_counts[i],
            right_insertions = right_counts[i],
        ) for i in eachindex(payload.X1_elements)
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

`times` is clamped to `[0, P.total_time]`. Reveal / insertion events are
distributed over `[0, P.total_time - 1]`.

The return value intentionally mirrors `branching_bridge`:

- `t`: the sampled global extended times.
- `Xt`: a batched `FlowceptionState`.
- `X1anchor`: batched visible targets aligned with `Xt.state`.
- `insertions_target`: number of missing elements to the right of each visible slot.

Mask semantics:
- Explicit `MaskedState` component masks are preserved through extraction and
  batching.
- Plain components are wrapped internally using `flowmask` as `cmask`
  and `padmask & flowmask` as `lmask`, matching the historical
  sequence-level masking behavior.
"""
function flowception_bridge(P::FlowceptionFlow, X1s, times; T = Float32, nstart = 1, maxlen = Inf)
    if times isa UnivariateDistribution
        times = rand(times, length(X1s))
    end
    times = clamp.(T.(times), zero(T), flowception_total_time(P, T))

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

    Xt_batch = pad_batch_masked([[x.Xt for x in bridge] for bridge in bridges])
    X1_batch = pad_batch_masked([[x.X1anchor for x in bridge] for bridge in bridges])
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

"""
    directional_flowception_bridge(P::DirectionalFlowceptionFlow, X1s, times;
                                   T = Float32,
                                   nstart = 1,
                                   maxlen = Inf)

Bidirectional Flowception bridge mirroring `flowception_bridge`, but the
insertion targets live on the token axis with shape `(2, L, B)`:

- `insertions_target[1, i, b]`: missing elements to the left of visible token `i`
- `insertions_target[2, i, b]`: missing elements to the right of visible token `i`

The loss for `DirectionalFlowceptionFlow` then pools adjacent same-group
left/right predictions directly from `Xt.groupings`.

`times` is clamped to `[0, P.total_time]`.

Mask semantics match `flowception_bridge`: explicit component masks are
preserved, while plain components are wrapped using sequence-level masks from
`flowmask` and `padmask`.
"""
function directional_flowception_bridge(P::DirectionalFlowceptionFlow, X1s, times; T = Float32, nstart = 1, maxlen = Inf)
    if times isa UnivariateDistribution
        times = rand(times, length(X1s))
    end
    times = clamp.(T.(times), zero(T), flowception_total_time(P, T))

    bridges = [single_directional_flowception_bridge(P, X1s[i], times[i], nstart) for i in eachindex(X1s)]
    batch_maxlen = maximum(length.(bridges))
    batch_maxlen <= maxlen || error("Visible sequence length $batch_maxlen exceeds `maxlen=$maxlen`.")

    b = length(bridges)
    flowmask = falses(batch_maxlen, b)
    branchmask = falses(batch_maxlen, b)
    padmask = falses(batch_maxlen, b)
    groups = zeros(Int, batch_maxlen, b)
    local_t = zeros(T, batch_maxlen, b)
    insertion_counts = zeros(T, 2, batch_maxlen, b)

    for j in eachindex(bridges)
        for i in eachindex(bridges[j])
            flowmask[i, j] = bridges[j][i].flowable
            branchmask[i, j] = bridges[j][i].branchable
            padmask[i, j] = true
            groups[i, j] = bridges[j][i].group
            local_t[i, j] = bridges[j][i].local_t
            insertion_counts[1, i, j] = bridges[j][i].left_insertions
            insertion_counts[2, i, j] = bridges[j][i].right_insertions
        end
    end

    Xt_batch = pad_batch_masked([[x.Xt for x in bridge] for bridge in bridges])
    X1_batch = pad_batch_masked([[x.X1anchor for x in bridge] for bridge in bridges])

    return (;
        t = times,
        Xt = FlowceptionState(Xt_batch, groups;
            local_t,
            branchmask,
            flowmask,
            padmask,
        ),
        X1anchor = X1_batch,
        insertions_target = insertion_counts,
        left_insertions_target = selectdim(insertion_counts, 1, 1),
        right_insertions_target = selectdim(insertion_counts, 1, 2),
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

"""
    flowception_base_step(P, XₜFS, X1targets, s₁, s₂)

Advance the base process on the currently visible Flowception state.

Before stepping, plain components of `XₜFS.state` are wrapped with explicit
`MaskedState`s using `XₜFS.flowmask` as both `cmask` and `lmask`, so
non-flowing elements remain fixed while explicit component masks are preserved.
"""
function flowception_base_step(P::AbstractFlowceptionProcess, XₜFS::FlowceptionState, X1targets, s₁::Real, s₂::Real)
    Xₜ = effective_masked_tuple(XₜFS.state, XₜFS.flowmask, XₜFS.flowmask)
    local_t₁ = XₜFS.local_t
    T = eltype(local_t₁)
    Δg = T(s₂ - s₁)
    local_t₂ = clip01(local_t₁ .+ T.(XₜFS.flowmask) .* Δg)

    next_Xₜ = map(state_tuple(P.P), Xₜ, state_tuple(X1targets)) do Pi, Xi, X̂i
        flowception_component_step(Pi, Xi, X̂i, local_t₁, local_t₂)
    end
    Xₜ = Flowfusion.mask(next_Xₜ, Xₜ)
    current_flowmask = XₜFS.padmask .& (local_t₂ .< one(T))
    return Xₜ, local_t₂, current_flowmask
end

"""
    flowception_static_state(Xₜ, XₜFS, local_t₂, current_flowmask)

Rebuild a `FlowceptionState` after a base step with no insertions, preserving
explicit component masks and applying `current_flowmask` as the explicit mask
for any plain components.
"""
function flowception_static_state(Xₜ, XₜFS::FlowceptionState, local_t₂, current_flowmask)
    return FlowceptionState(remask_like_tuple(XₜFS.state, Xₜ, current_flowmask, current_flowmask), XₜFS.groupings;
        local_t = local_t₂,
        branchmask = XₜFS.branchmask,
        flowmask = current_flowmask,
        padmask = XₜFS.padmask,
    )
end

"""
    rebuild_flowception_state(P, XₜFS, Xₜ, local_t₂, current_flowmask, before_counts, after_counts)

Insert newly born elements around the current visible sequence and return the
updated `FlowceptionState`.

Existing elements preserve their explicit component masks; newly rebuilt plain
components inherit `current_flowmask` as explicit `cmask/lmask`.
"""
function rebuild_flowception_state(P::AbstractFlowceptionProcess,
        XₜFS::FlowceptionState,
        Xₜ,
        local_t₂,
        current_flowmask,
        before_counts::AbstractVector{<:Integer},
        after_counts::AbstractVector{<:Integer})
    total_insertions = sum(before_counts) + sum(after_counts)
    total_insertions == 0 && return flowception_static_state(Xₜ, XₜFS, local_t₂, current_flowmask)

    refs = Tuple[]
    newelements = Any[]
    for i in 1:size(XₜFS.groupings, 1)
        ref = remask_element_like(XₜFS.state, Xₜ, current_flowmask, current_flowmask, i, 1)
        before_counts[i] > 0 && append!(refs, ntuple(_ -> ref, before_counts[i]))
        after_counts[i] > 0 && append!(refs, ntuple(_ -> ref, after_counts[i]))
    end

    births = normalized_birth.(Ref(P), refs)
    current_length = size(XₜFS.groupings, 1)
    new_length = current_length + total_insertions
    groupings = similar(XₜFS.groupings, new_length, 1)
    branchmask = similar(XₜFS.branchmask, new_length, 1)
    local_t = similar(local_t₂, new_length, 1)
    flowmask = similar(current_flowmask, new_length, 1)
    padmask = similar(XₜFS.padmask, new_length, 1)

    birth_index = 1
    dst = 1
    for i in 1:current_length
        for _ in 1:before_counts[i]
            push!(newelements, births[birth_index])
            groupings[dst, 1] = XₜFS.groupings[i, 1]
            branchmask[dst, 1] = XₜFS.branchmask[i, 1]
            local_t[dst, 1] = zero(eltype(local_t₂))
            flowmask[dst, 1] = true
            padmask[dst, 1] = true
            dst += 1
            birth_index += 1
        end

        push!(newelements, remask_element_like(XₜFS.state, Xₜ, current_flowmask, current_flowmask, i, 1))
        groupings[dst, 1] = XₜFS.groupings[i, 1]
        branchmask[dst, 1] = XₜFS.branchmask[i, 1]
        local_t[dst, 1] = local_t₂[i, 1]
        flowmask[dst, 1] = current_flowmask[i, 1]
        padmask[dst, 1] = true
        dst += 1

        for _ in 1:after_counts[i]
            push!(newelements, births[birth_index])
            groupings[dst, 1] = XₜFS.groupings[i, 1]
            branchmask[dst, 1] = XₜFS.branchmask[i, 1]
            local_t[dst, 1] = zero(eltype(local_t₂))
            flowmask[dst, 1] = true
            padmask[dst, 1] = true
            dst += 1
            birth_index += 1
        end
    end

    return FlowceptionState(pad_batch_masked([newelements], [vec(flowmask)], [vec(flowmask)]), groupings;
        local_t,
        branchmask,
        flowmask,
        padmask,
    )
end

function Flowfusion.step(P::FlowceptionFlow, XₜFS::FlowceptionState, hat::Tuple, s₁::Real, s₂::Real)
    X1targets, insertion_logits = hat
    Xₜ, local_t₂, current_flowmask = flowception_base_step(P, XₜFS, X1targets, s₁, s₂)
    T = eltype(local_t₂)
    insert_dt = flowception_insert_dt(P, s₁, s₂, T)
    safe_logits = ifelse.(isfinite.(insertion_logits), insertion_logits, zero(T))
    λ = max.(P.insertion_transform.(safe_logits), zero(T))
    ρ = scheduler_hazard(P, T(s₁))
    after_counts = vec(XₜFS.branchmask .* XₜFS.padmask .* rand.(Poisson.(insert_dt .* ρ .* λ)))
    before_counts = zeros(Int, length(after_counts))
    return rebuild_flowception_state(P, XₜFS, Xₜ, local_t₂, current_flowmask, before_counts, after_counts)
end

function directional_physical_rates(P::DirectionalFlowceptionFlow, insertion_logits, XₜFS::FlowceptionState)
    left_logits, right_logits = directional_heads(insertion_logits)
    T = eltype(XₜFS.local_t)
    safe_left = ifelse.(isfinite.(left_logits), left_logits, zero(T))
    safe_right = ifelse.(isfinite.(right_logits), right_logits, zero(T))
    λ_left = max.(P.insertion_transform(safe_left), zero(T))
    λ_right = max.(P.insertion_transform(safe_right), zero(T))
    masks = directional_insertion_masks(XₜFS.groupings, XₜFS.branchmask, XₜFS.padmask)

    after_rates = zero.(λ_right)
    if size(λ_right, 1) > 1
        after_rates[1:end-1, :] .= ifelse.(
            masks.interior_mask,
            (λ_right[1:end-1, :] .+ λ_left[2:end, :]) ./ T(2),
            zero(T),
        )
    end
    after_rates .= ifelse.(masks.right_mask, λ_right, after_rates)
    before_rates = ifelse.(masks.left_mask, λ_left, zero(T))
    return before_rates, after_rates, masks
end

function Flowfusion.step(P::DirectionalFlowceptionFlow, XₜFS::FlowceptionState, hat::Tuple, s₁::Real, s₂::Real)
    X1targets, insertion_logits = hat
    Xₜ, local_t₂, current_flowmask = flowception_base_step(P, XₜFS, X1targets, s₁, s₂)
    T = eltype(local_t₂)
    insert_dt = flowception_insert_dt(P, s₁, s₂, T)
    before_rates, after_rates, _ = directional_physical_rates(P, insertion_logits, XₜFS)
    ρ = scheduler_hazard(P, T(s₁))
    before_counts = vec(rand.(Poisson.(insert_dt .* ρ .* before_rates)))
    after_counts = vec(rand.(Poisson.(insert_dt .* ρ .* after_rates)))
    return rebuild_flowception_state(P, XₜFS, Xₜ, local_t₂, current_flowmask, before_counts, after_counts)
end

Flowfusion.scalefloss(P::FlowceptionFlow, t::Real, pow = 2, eps = typeof(t)(0.05)) = one(t) / (((one(t) + eps) - flowception_insertion_phase(P, t)) ^ pow)
Flowfusion.scalefloss(P::FlowceptionFlow, t::AbstractArray, pow = 2, eps = eltype(t)(0.05)) = 1 ./ ((1 + eps) .- flowception_insertion_phase(P, t)) .^ pow
Flowfusion.floss(P::FlowceptionFlow, X̂₁, X₁, mask, c) = Flowfusion.scaledmaskedmean(sbpl(P.insertion_transform(X̂₁), X₁), c, mask)
Flowfusion.scalefloss(P::DirectionalFlowceptionFlow, t::Real, pow = 2, eps = typeof(t)(0.05)) = one(t) / (((one(t) + eps) - flowception_insertion_phase(P, t)) ^ pow)
Flowfusion.scalefloss(P::DirectionalFlowceptionFlow, t::AbstractArray, pow = 2, eps = eltype(t)(0.05)) = 1 ./ ((1 + eps) .- flowception_insertion_phase(P, t)) .^ pow

function Flowfusion.floss(P::DirectionalFlowceptionFlow, X̂₁, X₁, Xt::FlowceptionState, c)
    left_hat, right_hat = directional_heads(X̂₁)
    left_target, right_target = directional_heads(X₁)
    T = promote_type(eltype(left_hat), eltype(right_hat), eltype(left_target), eltype(right_target))
    λ_left = max.(P.insertion_transform(left_hat), zero(T))
    λ_right = max.(P.insertion_transform(right_hat), zero(T))
    masks = ChainRulesCore.ignore_derivatives() do
        directional_insertion_masks(Xt.groupings, Xt.branchmask, Xt.padmask)
    end

    left_loss = sbpl(λ_left, left_target)
    right_loss = sbpl(λ_right, right_target)
    interior_loss = sbpl((λ_right[1:end-1, :] .+ λ_left[2:end, :]) ./ T(2), right_target[1:end-1, :])

    numerator = masked_slot_sum(left_loss, c, masks.left_mask) +
        masked_slot_sum(right_loss, c, masks.right_mask) +
        masked_slot_sum(interior_loss, c, masks.interior_mask)
    denominator = sum(masks.left_mask) + sum(masks.right_mask) + sum(masks.interior_mask)
    return numerator / (oftype(numerator, denominator) + oftype(numerator, 1e-6))
end

Flowfusion.floss(P::DirectionalFlowceptionFlow, X̂₁, X₁, mask::AbstractArray, c) = error("DirectionalFlowceptionFlow loss needs the full `FlowceptionState` so it can pool insertions using `groupings`; call `floss(P, logits, target, Xt, c)`.")
