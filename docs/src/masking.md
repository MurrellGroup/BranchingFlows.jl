```@meta
CurrentModule = BranchingFlows
```

# Masking And Context

This page explains the masking semantics used by both the original
Branching Flows path and the Flowception paths.

The short version is:

- `branchmask` controls topology changes
- `flowmask` controls whether an element still evolves under the base process
- `padmask` marks valid sequence positions
- component-level `cmask` and `lmask` live inside `Flowfusion.MaskedState`
- an element is never allowed to be branchable if any component is
  non-designable there

That last rule is the key invariant:

```julia
branchmask[i] == true  =>  every component has cmask[i] == true
```

For plain, non-`MaskedState` components, `flowmask` acts as the effective
component `cmask` at the public API level.

## Vocabulary

The package uses two layers of masking.

### Sequence-level masks

These live directly on `BranchingState` and `FlowceptionState`.

| Mask | Meaning |
| --- | --- |
| `branchmask` | Whether topology may change at that position. In Branching Flows this means split/delete eligibility. In Flowception this means insertion eligibility. |
| `flowmask` | Whether the base process should continue updating that element. |
| `padmask` | Whether the position is a real element instead of padding. |

### Component-level masks

These live inside `Flowfusion.MaskedState(component, cmask, lmask)`.

| Mask | Meaning |
| --- | --- |
| `cmask` | Whether that specific component is designable / allowed to move toward its target. |
| `lmask` | The component-local live mask carried through batching and masking operations. |

In mixed states, different components may have different `cmask`s, but only
when `branchmask == false` at those positions.

## Public constructor rules

The public state constructors are `BranchingState` and `FlowceptionState`.
Their full API signatures are documented on the home page and the Flowception
page; this page focuses on the shared masking semantics.

Their masking rules are:

1. `padmask == false` means "ignore this position".
2. `flowmask == false` freezes the base-process evolution for that element.
3. `branchmask == true` is only legal if every component is designable there.
4. If a component is plain, not a `MaskedState`, then `flowmask` is the
   effective component design mask for that component.
5. If a component is already a `MaskedState`, its explicit `cmask/lmask` are
   preserved.

So the public API behaves as if plain components had been wrapped explicitly
with component masks derived from the sequence masks.

## Plain states vs explicit `MaskedState`

### Plain component

If you pass a plain state component, like `ContinuousState`, you are choosing
the simple API: sequence-level masks control that component.

```julia
using BranchingFlows, Flowfusion

state = ContinuousState(reshape(Float32[1, 2, 3], 1, :))

X = BranchingState(
    state,
    ones(Int, 3);
    flowmask = Bool[true, false, true],
    branchmask = Bool[true, false, false],
    padmask = trues(3),
)
```

Interpretation:

- position 1 is flowing and branchable
- position 2 is frozen and not branchable
- position 3 is flowing but not branchable

If you changed `branchmask[2]` to `true`, the constructor would throw, because
for a plain component the effective component `cmask` is `flowmask`.

### Explicit `MaskedState` component

If you need component-specific design masks, wrap that component explicitly.

```julia
using BranchingFlows, Flowfusion

locs = MaskedState(
    ContinuousState(randn(Float32, 2, 3)),
    trues(3),
    trues(3),
)

toks = MaskedState(
    DiscreteState(21, Int[5, 8, 13]),
    Bool[true, false, false],
    trues(3),
)

X = BranchingState(
    (locs, toks),
    ones(Int, 3);
    flowmask = trues(3),
    branchmask = falses(3),
    padmask = trues(3),
)
```

This is legal because `branchmask` is off everywhere. The token component is
only designable at the first position, while the location component is
designable everywhere.

## Legal and illegal patterns

### Legal: different component `cmask`s when `branchmask == false`

```julia
using BranchingFlows, Flowfusion

state = (
    MaskedState(ContinuousState(randn(Float32, 1, 2)), trues(2), trues(2)),
    MaskedState(ContinuousState(randn(Float32, 1, 2)), Bool[true, false], trues(2)),
)

X = BranchingState(
    state,
    ones(Int, 2);
    branchmask = Bool[true, false],
    flowmask = trues(2),
    padmask = trues(2),
)
```

This is legal because the partially masked second component only appears at the
non-branchable position.

### Illegal: branchable but non-designable

```julia
using BranchingFlows, Flowfusion

bad_state = (
    MaskedState(ContinuousState(randn(Float32, 1, 2)), trues(2), trues(2)),
    MaskedState(ContinuousState(randn(Float32, 1, 2)), Bool[true, false], trues(2)),
)

BranchingState(
    bad_state,
    ones(Int, 2);
    branchmask = trues(2),
    flowmask = trues(2),
    padmask = trues(2),
)
```

This throws, because the second component has `cmask[2] == false` while
`branchmask[2] == true`.

### Illegal for plain components: `flowmask == false` and `branchmask == true`

```julia
using BranchingFlows, Flowfusion

BranchingState(
    ContinuousState(reshape(Float32[1, 2], 1, :)),
    ones(Int, 2);
    flowmask = Bool[false, true],
    branchmask = trues(2),
    padmask = trues(2),
)
```

This also throws. For a plain component, `flowmask` is its effective
component `cmask`, so position `1` would be branchable but non-designable.

## Context recipes

The masking recipes below are the ones to keep in mind.

| Situation | Plain component? | `flowmask` | `branchmask` | Extra requirement |
| --- | --- | --- | --- | --- |
| ordinary generated element | yes | `true` | `true` or `false` | none |
| frozen passive context | yes | `false` | `false` | none |
| frozen but still branchable context | no, not safely | `false` | `true` | every component must be an explicit `MaskedState` with `cmask=true` |
| partially designable tuple state | yes | any | `false` | component-specific masking must be explicit via `MaskedState` |
| padding | yes | any | any | `padmask=false` |

That third row is the subtle one. If you want an element that does not evolve
under the base process but is still allowed to branch/insert, you must make
all components explicit `MaskedState`s and keep their `cmask` equal to `true`.

## Branching Flows examples

### Bridge preserves explicit component masks

This is the minimal deterministic setup used in the test suite.

```julia
using BranchingFlows, Flowfusion

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
```

The important thing here is that the second component mask survives the bridge:

```julia
vec(ts.Xt.state[2].cmask[:, 1]) == Bool[true, false]
vec(ts.X1anchor[2].cmask[:, 1]) == Bool[true, false]
```

### Plain components still freeze under `flowmask`

```julia
using BranchingFlows, Flowfusion

base_state = BranchingState(
    (ContinuousState(reshape(Float32[0, 10], 1, 2, 1)),),
    ones(Int, 2, 1);
    branchmask = falses(2, 1),
    flowmask = reshape(Bool[false, true], 2, 1),
    padmask = trues(2, 1),
)

hat = (
    (ContinuousState(fill(5f0, 1, 2, 1)),),
    fill(-100f0, 2, 1),
    fill(-100f0, 2, 1),
)

next_state = Flowfusion.step(
    CoalescentFlow((Deterministic(),), BranchingFlows.Uniform(0f0, 1f0)),
    base_state,
    hat,
    0f0,
    0.25f0,
)
```

The first element stays fixed because `flowmask[1] == false`, while the second
continues to evolve.

## Flowception examples

### Same masking vocabulary, plus `local_t`

`FlowceptionState` uses the same masking rules as `BranchingState`, but adds
per-element local time:

```julia
using BranchingFlows, Flowfusion

X = FlowceptionState(
    MaskedState(ContinuousState(randn(Float32, 2, 4)), trues(4), trues(4)),
    ones(Int, 4);
    local_t = fill(0.25f0, 4),
    branchmask = trues(4),
    flowmask = trues(4),
    padmask = trues(4),
)
```

### Frozen but branchable Flowception context requires explicit masks

```julia
using BranchingFlows, Flowfusion

locs = MaskedState(ContinuousState(randn(Float32, 2, 3)), trues(3), trues(3))
toks = MaskedState(DiscreteState(3, Int[1, 2, 3]), trues(3), trues(3))

X = FlowceptionState(
    (locs, toks),
    ones(Int, 3);
    local_t = ones(Float32, 3),
    flowmask = falses(3),
    branchmask = trues(3),
    padmask = trues(3),
)
```

This is legal: the elements are not denoising, but they are still allowed to
host insertions because every component remains designable.

If you tried the same mask pattern with plain components, the constructor would
throw.

### Plain Flowception components still freeze correctly

```julia
using BranchingFlows, Flowfusion

base_state = FlowceptionState(
    (ContinuousState(reshape(Float32[0, 10], 1, 2, 1)),),
    reshape([1, 1], 2, 1);
    local_t = zeros(Float32, 2, 1),
    branchmask = falses(2, 1),
    flowmask = reshape(Bool[false, true], 2, 1),
    padmask = trues(2, 1),
)

P = FlowceptionFlow((Deterministic(),), () -> ContinuousState(fill(-1f0, 1, 1)))
hat = ((ContinuousState(fill(5f0, 1, 2, 1)),), fill(-100f0, 2, 1))

next_state = Flowfusion.step(P, base_state, hat, 0f0, 0.25f0)
```

Operationally:

- the first element stays fixed
- the second element moves toward the predicted endpoint
- `local_t` only advances for the second element

## What the bridges and steps preserve

The package preserves explicit component masks through:

- `branching_bridge`
- `flowception_bridge`
- `directional_flowception_bridge`
- `Flowfusion.step(::CoalescentFlow, ...)`
- `Flowfusion.step(::FlowceptionFlow, ...)`
- `Flowfusion.step(::DirectionalFlowceptionFlow, ...)`

For plain components, these paths explicitly wrap the component using the
relevant sequence-level masks before batching or stepping. That is why plain
components still obey `flowmask`, while explicit `MaskedState` components keep
their own `cmask/lmask`.

## Internal helper reference

These are internal utilities, but they are the relevant mask-preserving entry
points if you extend the package internals.

```@docs
BranchingFlows.masked_element
BranchingFlows.masked_tuple
BranchingFlows.batch_masked
BranchingFlows.pad_batch_masked
BranchingFlows.validate_branchmask_cmask
```
