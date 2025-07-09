canonical_anchor_merge(S1::Tuple{Vararg{Flowfusion.UState}}, S2::Tuple{Vararg{Flowfusion.UState}}, w1, w2) = canonical_anchor_merge.(S1, S2, w1, w2)
canonical_anchor_merge(S1::MaskedState, S2::MaskedState, w1, w2) = canonical_anchor_merge(S1.S, S2.S, w1, w2)
canonical_anchor_merge(S1::ContinuousState, S2::ContinuousState, w1, w2) = ContinuousState((tensor(S1) * w1 + tensor(S2) * w2) / (w1 + w2))
canonical_anchor_merge(S1::ManifoldState, S2::ManifoldState, w1, w2) = ForwardBackward.interpolate(S1, S2, w2, w1)
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



#=
L = 1200
b = 2
locs = rand(Float32, 3, 1, L, b)
rots = rand(Float32, 3, 3, L, b)
aas = rand(1:21, L, b)

cmask = trues(L, b)
cmask[5:7,2] .= false
padmask = trues(L, b)
padmask[5:end,2] .= false

rotM = Rotations(3)

X1locs = MaskedState(ContinuousState(locs), cmask, padmask)
X1rots = MaskedState(ManifoldState(rotM, reshape(Array{Float32}.(Flowfusion.rand(rotM, L*b)), L, b)), cmask, padmask)
X1aas = MaskedState(DiscreteState(21, Flowfusion.onehotbatch(aas, 1:21)), cmask, padmask)
X1aas_unhot = MaskedState(DiscreteState(21, aas), cmask, padmask)
points = MaskedState(ContinuousState(rand(Float32, L, b)), cmask, padmask)

compound_state = (X1locs, X1rots, X1aas, points, X1aas_unhot);

tup1 = (element(X1locs, 1, 1), element(X1rots, 1, 1), element(X1aas_unhot, 1, 1), element(points, 1, 1), element(X1aas, 1, 1))
tup2 = (element(X1locs, 1, 2), element(X1rots, 1, 2), element(X1aas_unhot, 1, 2), element(points, 1, 2), element(X1aas, 1, 2))
avg1 = canonical_anchor_merge(tup1, tup2, 1, 0)
avg2 = canonical_anchor_merge(tup1, tup2, 0, 1)
isapprox.(tensor.(avg1), tensor.(tup1))
isapprox.(tensor.(avg2), tensor.(tup2))
#Note: this is not meant to match for the AAs, which use the "last token" as the dummy/masked token for the anchors.
=#
