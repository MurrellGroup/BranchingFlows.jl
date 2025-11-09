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
