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
struct IndependentRevealOrder <: AbstractFlowceptionRevealOrder end
struct DefaultSeedPriority end

struct SeededRevealOrder{T,F} <: AbstractFlowceptionRevealOrder
    temperature::T
    seed_priority::F
end

SeededRevealOrder(; temperature = 0.25f0, seed_priority = DefaultSeedPriority()) = SeededRevealOrder(temperature, seed_priority)

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

function choose_seed_positions(X1::FlowceptionState, seg, design_local::AbstractVector{Int}, fixed_local::AbstractVector{Int}, nstart::Int, policy::SeededRevealOrder, ::Type{T}) where T
    nseed = min(nstart, length(design_local))
    nseed == 0 && return Int[]
    scores = seed_priority_scores(policy.seed_priority, X1, seg, design_local, fixed_local, T)
    order = sortperm(scores)
    return collect(design_local[order[1:nseed]])
end

function seeded_visible_mask(P::AbstractFlowceptionProcess, X1::FlowceptionState, τg::T, nstart::Int, policy::SeededRevealOrder) where T
    groups = vec(X1.groupings)
    target_flow = vec(X1.flowmask)
    target_pad = vec(X1.padmask)
    visible = falses(length(groups))
    local_t = zeros(T, length(groups))
    horizon = flowception_insertion_horizon(P, T)

    start = firstindex(groups)
    while start <= length(groups)
        if !target_pad[start]
            start += 1
            continue
        end
        stop = group_segment_stop(groups, target_pad, start)
        seg = start:stop
        segment_visible = view(visible, seg)
        segment_local_t = view(local_t, seg)
        local_flow = target_flow[seg]
        fixed_local = findall(.!local_flow)
        design_local = findall(local_flow)

        if !isempty(fixed_local)
            segment_visible[fixed_local] .= true
            segment_local_t[fixed_local] .= one(T)
        end

        if !isempty(design_local)
            if isempty(fixed_local) && nstart <= 0
                error("Group $(groups[start]) has no fixed context and `nstart=0` under `SeededRevealOrder`. Increase `nstart` or provide fixed visible context.")
            end

            seed_local = choose_seed_positions(X1, seg, design_local, fixed_local, nstart, policy, T)
            if !isempty(seed_local)
                segment_visible[seed_local] .= true
                segment_local_t[seed_local] .= clip01(τg)
            end

            remaining_local = setdiff(design_local, seed_local)
            if !isempty(remaining_local)
                seedmask = falses(length(seg))
                seedmask[fixed_local] .= true
                seedmask[seed_local] .= true
                layer_distance = nearest_seed_distances(seedmask)
                scores = T.(layer_distance[remaining_local]) .+ T(1e-4) .* T.(remaining_local)
                if policy.temperature > zero(policy.temperature)
                    scores .+= T(policy.temperature) .* T[gumbel_noise(T) for _ in eachindex(remaining_local)]
                end
                order = sortperm(scores)
                reveal_levels = sort!(rand(T, length(remaining_local)))
                for (rank, idx) in enumerate(order)
                    local_idx = remaining_local[idx]
                    delay = horizon * T(P.scheduler_inverse(reveal_levels[rank]))
                    if τg >= delay
                        segment_visible[local_idx] = true
                        segment_local_t[local_idx] = clip01(τg - delay)
                    end
                end
            end
        end

        start = stop + 1
    end

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

function single_flowception_bridge(P::FlowceptionFlow, X1::FlowceptionState, τg::T, nstart::Int) where T
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
    visible, local_t_full = original_visible_mask(P, X1, τg, nstart)
    visible_inds, left_counts, right_counts = directional_slot_targets(vec(X1.groupings), visible, vec(X1.padmask))
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
