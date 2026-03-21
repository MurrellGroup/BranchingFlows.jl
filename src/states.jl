"""
    canonical_anchor_merge(S1, S2, w1, w2)

Merge two anchor states `S1` and `S2` (each possibly a tuple of component states)
into a single anchor representing their parent during tree coalescence.

Behavior by component:
- Tuple of states: applies `canonical_anchor_merge` elementwise with the same weights.
- `MaskedState`: unwraps and merges the underlying state.
- `ContinuousState`: weighted Euclidean average, `(w1*S1 + w2*S2)/(w1+w2)`.
- `ManifoldState`: geodesic interpolation via `ForwardBackward.interpolate(S1, S2, w2, w1)`.
- `DiscreteState`: sets the anchor to the mask/dummy token (assumes the dummy token
  is the last category `K`); both dense-int and onehot encodings are supported.

`w1` and `w2` are nonnegative weights, typically child descendant counts, used
only by the continuous/manifold components; discrete components always resolve
to the dummy token to avoid leaking label information through anchors.
"""
canonical_anchor_merge(S1::Tuple{Vararg{Flowfusion.UState}}, S2::Tuple{Vararg{Flowfusion.UState}}, w1, w2) = canonical_anchor_merge.(S1, S2, w1, w2)
canonical_anchor_merge(S1::MaskedState, S2::MaskedState, w1, w2) = canonical_anchor_merge(S1.S, S2.S, w1, w2)
canonical_anchor_merge(S1::ContinuousState, S2::ContinuousState, w1, w2) = ContinuousState((tensor(S1) * w1 + tensor(S2) * w2) / (w1 + w2))
canonical_anchor_merge(S1::ManifoldState, S2::ManifoldState, w1, w2) = ForwardBackward.interpolate(S1, S2, w2, w1) #<-NOTE: for protein model, check argument order here

#NOTE: ASSUMES THAT THE DUMMY/MASKED TOKEN IS THE LAST ONE! OVERLOAD THIS IF YOU WANT OTHER BEHAVIOR.
function canonical_anchor_merge(S1::DiscreteState{<:AbstractArray{<:Signed}}, S2::DiscreteState{<:AbstractArray{<:Signed}}, w1, w2) #Will dispatch on non-onehot
    dest = copy(S1)
    dest.state .= dest.K
    return dest
end
function canonical_anchor_merge(S1::DiscreteState, S2::DiscreteState, w1, w2)
    dest = copy(S1)
    dest.state .= Flowfusion.onehotbatch(dest.K, 1:dest.K)
    return dest
end

#Note: this will dispatch on the Tuple first, and will assign the weights the same for all components of the state, which is the desired behavior.
#We don't want to sample separately for eg. rots and locs.
#This will just pick one or the other with no interpolation.
"""
    select_anchor_merge(S1, S2, w1, w2)

Stochastic alternative to `canonical_anchor_merge` that selects one of the two
children as the parent anchor without interpolation. Chooses `S1` with probability
`w1/(w1+w2)` and `S2` otherwise, then calls `canonical_anchor_merge` with weights
`(1,0)` or `(0,1)` to preserve type-specific behavior (e.g., discrete masking).

Intended for scenarios where copying a child is preferable to averaging (e.g.,
to keep anchors on-manifold for complex discrete/structured components).
"""
function select_anchor_merge(S1, S2, w1, w2)
    if rand() < w1 / (w1 + w2)
        return canonical_anchor_merge(S1, S2, 1, 0)
    else
        return canonical_anchor_merge(S2, S1, 0, 1)
    end
end

#This needs to be moved into Flowfusion
"""
    Flowfusion.regroup(elarray::AbstractVector{<:Tuple})

Utility to batch a vector of element-wise tuples of states into a tuple of
batched states. Given `elarray::Vector{Tuple{S₁, S₂, …, Sₖ}}` of length `L`,
returns `Tuple{Ŝ₁, Ŝ₂, …, Ŝₖ}` where each `Ŝᵢ` has length `L` (stacking the
elements along the sequence dimension).

This specialization enables:
```julia
MaskedState.(regroup(elements), (flowmask,), (padmask,))
```
when building `BranchingState` batches from per-segment bridges.
"""
function Flowfusion.regroup(elarray::AbstractVector{<:Tuple})
    example_tuple = elarray[1]
    len = length(elarray)
    newstates = [Flowfusion.zerostate(example_tuple[i],len) for i in 1:length(example_tuple)]
    for j in 1:len
        for k in 1:length(example_tuple)
            Flowfusion.element(tensor(newstates[k]),j) .= tensor(elarray[j][k])
        end
    end
    return Tuple(newstates)
end

state_tuple(x) = x isa Tuple ? x : (x,)

"""
    mask_preserving_element(S, inds...)

Element extraction variant of `Flowfusion.element` that preserves
`MaskedState.cmask` and `MaskedState.lmask`.

This is only defined for already-masked components (or tuples thereof). If a
plain component reaches this path, that indicates a missing explicit mask
wrapping step and an informative error is thrown.
"""
function mask_preserving_element(S::MaskedState, inds...)
    return MaskedState(
        Flowfusion.element(S.S, inds...),
        Flowfusion.element(S.cmask, inds...),
        Flowfusion.element(S.lmask, inds...),
    )
end
mask_preserving_element(S::Tuple{Vararg{Flowfusion.UState}}, inds...) = map(component -> mask_preserving_element(component, inds...), S)
mask_preserving_element(S, inds...) = error("Expected a `MaskedState` component in `mask_preserving_element`, got $(typeof(S)). Wrap plain components with explicit masks before calling this path.")

"""
    effective_masked_element(S, cmask, lmask, inds...)

Extract one element from `S`, preserving explicit `MaskedState` masks when they
exist and otherwise wrapping plain components in a `MaskedState` using the
provided sequence-level `cmask` and `lmask`.
"""
effective_masked_element(S::MaskedState, cmask, lmask, inds...) = mask_preserving_element(S, inds...)
effective_masked_element(S::Tuple{Vararg{Flowfusion.UState}}, cmask, lmask, inds...) = map(component -> effective_masked_element(component, cmask, lmask, inds...), S)
function effective_masked_element(S, cmask, lmask, inds...)
    return MaskedState(
        Flowfusion.element(S, inds...),
        Flowfusion.element(cmask, inds...),
        Flowfusion.element(lmask, inds...),
    )
end

"""
    explicit_masked_tuple(x, cmask, lmask)

Return `x` as a tuple of `MaskedState`s using the explicitly provided masks for
any plain components. Existing `MaskedState` components are preserved.
"""
explicit_masked_component(S::MaskedState, cmask, lmask) = S
explicit_masked_component(S, cmask, lmask) = MaskedState(S, copy(cmask), copy(lmask))
explicit_masked_tuple(x, cmask, lmask) = map(component -> explicit_masked_component(component, cmask, lmask), state_tuple(x))

strict_masked_component(S::MaskedState) = S
strict_masked_component(S) = error("Expected a `MaskedState` component, got $(typeof(S)). Pass explicit `cmask/lmask` when batching or stepping plain components.")
strict_masked_tuple(x) = map(strict_masked_component, state_tuple(x))

"""
    batch_mask_preserving_elements(Xs)
    batch_mask_preserving_elements(Xs, cmask, lmask)

Batch a vector of element states into a tuple of batched `MaskedState`s.

The no-mask method requires every component to already be a `MaskedState`.
The explicit-mask method wraps any plain components using the supplied
`cmask/lmask` while preserving explicit component masks.
"""
batch_mask_preserving_elements(Xs::AbstractVector) = Flowfusion.batch(strict_masked_tuple.(Xs))
batch_mask_preserving_elements(Xs::AbstractVector, cmask, lmask) = Flowfusion.batch([
    explicit_masked_tuple(Xs[i], Flowfusion.element(cmask, i), Flowfusion.element(lmask, i))
    for i in eachindex(Xs)
])

"""
    single_batch_mask_preserving_elements(Xs)
    single_batch_mask_preserving_elements(Xs, cmask, lmask)

Single-batch wrapper around `padded_batch_mask_preserving_elements`, preserving
explicit component masks. The explicit-mask method wraps plain components using
the supplied `cmask/lmask`.
"""
single_batch_mask_preserving_elements(Xs::AbstractVector) = padded_batch_mask_preserving_elements([Xs])
single_batch_mask_preserving_elements(Xs::AbstractVector, cmask, lmask) = padded_batch_mask_preserving_elements([Xs], [cmask], [lmask])

"""
    assign_batched_slice!(dest, src, batchindex)

Copy one batch slice from `src` into the `batchindex`th slice of `dest`,
supporting both state tensors and mask tensors with arbitrary leading feature
dimensions.
"""
function assign_batched_slice!(dest, src, batchindex)
    target = selectdim(dest, ndims(dest), batchindex)
    seq_len = size(src, ndims(src))
    prefix = ntuple(_ -> Colon(), max(ndims(target) - 1, 0))
    if isempty(prefix)
        target[1:seq_len] .= src
    else
        target[prefix..., 1:seq_len] .= src
    end
    return dest
end

function combine_masked_batches(per_batch, maxlen, b)
    example = per_batch[findfirst(!isnothing, per_batch)]
    components = map(eachindex(example)) do k
        example_component = example[k]
        example_element = mask_preserving_element(example_component, 1)
        state = Flowfusion.zerostate(Flowfusion.unmask(example_element), maxlen, b)
        cmask = falses(size(example_component.cmask)[1:end-1]..., maxlen, b)
        lmask = falses(size(example_component.lmask)[1:end-1]..., maxlen, b)
        for j in eachindex(per_batch)
            batch_component = per_batch[j]
            batch_component === nothing && continue
            assign_batched_slice!(tensor(state), tensor(Flowfusion.unmask(batch_component[k])), j)
            assign_batched_slice!(cmask, batch_component[k].cmask, j)
            assign_batched_slice!(lmask, batch_component[k].lmask, j)
        end
        MaskedState(state, cmask, lmask)
    end
    return Tuple(components)
end

"""
    padded_batch_mask_preserving_elements(Xss)
    padded_batch_mask_preserving_elements(Xss, cmasks, lmasks)

Pad and batch a vector of variable-length element collections into a tuple of
batched `MaskedState`s with a common `(maxlen, batch)` tail.

The no-mask method requires every component to already be a `MaskedState`.
The explicit-mask method wraps plain components using the supplied per-batch
`cmasks/lmasks` while preserving explicit component masks.
"""
function padded_batch_mask_preserving_elements(Xss::AbstractVector{<:AbstractVector})
    lengths = length.(Xss)
    maxlen = maximum(lengths)
    b = length(Xss)
    first_nonempty = findfirst(>(0), lengths)
    first_nonempty === nothing && error("Cannot batch an empty collection of states.")
    per_batch = Union{Nothing,Tuple}[isempty(Xs) ? nothing : batch_mask_preserving_elements(Xs) for Xs in Xss]
    return combine_masked_batches(per_batch, maxlen, b)
end

function padded_batch_mask_preserving_elements(Xss::AbstractVector{<:AbstractVector}, cmasks::AbstractVector, lmasks::AbstractVector)
    lengths = length.(Xss)
    maxlen = maximum(lengths)
    b = length(Xss)
    first_nonempty = findfirst(>(0), lengths)
    first_nonempty === nothing && error("Cannot batch an empty collection of states.")
    per_batch = Union{Nothing,Tuple}[isempty(Xs) ? nothing : batch_mask_preserving_elements(Xs, cmasks[i], lmasks[i]) for (i, Xs) in enumerate(Xss)]
    return combine_masked_batches(per_batch, maxlen, b)
end

"""
    validate_branchmask_cmask(state, branchmask, flowmask, padmask; context = "state")

Validate the BranchingFlows/Flowception invariant relating sequence-level
`branchmask` and component-level design masks.

For explicit `MaskedState` components, the component `cmask` is checked
directly. For plain components, `flowmask` is treated as the effective
component `cmask`. In either case, any live position (`padmask=1`) with
`branchmask=1` must have `cmask=1` in every component.
"""
function validate_branchmask_cmask(state, branchmask, flowmask, padmask; context = "state")
    for (component_index, component) in enumerate(state_tuple(state))
        component_cmask = component isa MaskedState ? component.cmask : flowmask
        size(component_cmask) == size(branchmask) || error("`$context` component $component_index has cmask size $(size(component_cmask)), expected $(size(branchmask)) to match branchmask.")
        invalid = branchmask .& padmask .& .!Bool.(component_cmask)
        any(invalid) || continue
        error("`$context` component $component_index has cmask=0 where branchmask=1. Branchable elements must be designable in every component.")
    end
    return nothing
end
