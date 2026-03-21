```@meta
CurrentModule = BranchingFlows
```

# Flowception

`BranchingFlows.jl` now ships two Flowception-style variable-length interfaces
on top of `Flowfusion.jl`:

- `FlowceptionFlow`: the current working Flowception path, with rightward
  insertions and per-element local denoising times.
- `DirectionalFlowceptionFlow`: a separate bidirectional extension that keeps
  the same state and bridge structure, but predicts left/right insertion heads
  and pools them using `groupings`.

This page documents both APIs, how they map to the Flowception construction,
and how they relate to the original Branching Flows code path.

## Papers

- Branching Flows paper:
  [Branching Flows: Discrete, Continuous, and Manifold Flow Matching with Splits and Deletions](https://arxiv.org/abs/2511.09465)
- Flowception paper:
  [Flowception](https://arxiv.org/abs/2512.11438)

The original Branching Flows implementation in this package is unchanged.
Flowception support is an additional code path that reuses the same masking,
batching, and `groupings` conventions where possible.

For the detailed masking rules, including when `flowmask=false,
branchmask=true` is legal and how plain components differ from explicit
`MaskedState`s, see [`Masking And Context`](masking.md).

## Which interface should I use?

- Use `FlowceptionFlow` if you want the current Flowception implementation that
  is closest to the paper setup already validated in this repository.
- Use `DirectionalFlowceptionFlow` if you want a separate experimental path for
  left/right insertions without changing the working `FlowceptionFlow`
  behavior.

The two Flowception paths intentionally share the same `FlowceptionState`, base
process bridge logic, local-time semantics, and scheduler hooks.

## Core idea

Flowception extends a standard base process with variable-length generation.
There are two time coordinates:

- global time `τg ∈ [0, total_time]`, which controls visibility / reveal events
- per-element local time `local_t ∈ [0, 1]`, which controls how long a visible
  element continues denoising

Once an element has accumulated one unit of local time, its own base-process
state stops changing. Sequence length can still change around it via
insertions.

`BranchingFlows.jl` represents this with a sequence state that looks very much
like `BranchingState`, but with an extra `local_t` field.

Both Flowception constructors also take `total_time`:

- `total_time`: total global generation horizon

This defaults to `2`, which reproduces the original behavior:

- elements can be revealed / spawned over the first `total_time - 1 = 1` unit
- each revealed element then gets exactly one unit of local denoising time

If `total_time = 10`, insertions and reveals are spread over `[0, 9]`, while
each element still only denoises for one local-time unit after it appears.

## `FlowceptionState`

`FlowceptionState` is the common state type for both Flowception paths.

In practice the fields mean:

- `state`: one Flowfusion state or a tuple of Flowfusion states, usually
  wrapped in `MaskedState`
- `groupings`: per-position group ids; insertions never cross a group boundary
- `local_t`: per-element base-process time in `[0, 1]`
- `branchmask`: insertion permission mask
- `flowmask`: whether the element is still denoising
- `padmask`: valid sequence positions

This mirrors the way `BranchingState` carries `groupings`, `branchmask`,
`flowmask`, and `padmask`.

The detailed legality rules are the same as for `BranchingState`:

- plain components use `flowmask` as their effective component design mask
- explicit `MaskedState` components keep their own `cmask/lmask`
- `branchmask=true` requires every component to be designable at that position

See [`Masking And Context`](masking.md) for worked examples.

## Groupings and multiple chains

`groupings` is the mechanism that keeps different chains or regions separate.
The same convention is used in `BranchingState`, `BranchChain`, and the
Flowception paths here.

For Flowception this means:

- insertions never cross a group boundary
- each group behaves like an independent chain / segment
- for multi-chain protein settings, `groupings` should usually be the chain id

For `DirectionalFlowceptionFlow`, chain boundaries are handled explicitly:

- same-group interior gap: pooled from `right[i]` and `left[i+1]`
- group start: uses `left[first]`
- group end: uses `right[last]`

So between the last residue of chain A and the first residue of chain B there
are two distinct boundary insertion opportunities, not one shared gap.

## Scheduler hooks

Both Flowception paths expose the same scheduler API through
`linear_scheduler`, `linear_scheduler_derivative`, and
`linear_scheduler_inverse`.

These define the reveal schedule `κ`, its derivative, and its inverse. The
defaults are the linear scheduler used in the Flowception paper, but custom
schedules can be passed to either constructor as long as the inverse matches
the scheduler used during training.

## `FlowceptionFlow`

`FlowceptionFlow` is the current working Flowception path.

### Model contract

For `FlowceptionFlow`, the model should consume `(t, Xt)` and return:

1. endpoint predictions for the wrapped base process `P.P`
2. a right-insertion head on the visible-token axis

Typical output shape:

```julia
X1hat, hat_insertions = model(ts.t, ts.Xt)
```

where:

- `X1hat` matches the structure of `P.P`
- `hat_insertions` has shape `(L, B)` after squeezing the feature axis

### Bridge contract

The training bridge is `flowception_bridge`.

`flowception_bridge(P, X1s, times; nstart=1)` returns a named tuple with:

- `t`: sampled global times `τg`
- `Xt`: batched `FlowceptionState`
- `X1anchor`: visible target elements aligned with `Xt`
- `insertions_target`: missing-element counts to the right of each visible
  position
- `splits_target`: compatibility alias for `insertions_target`

`nstart` controls how many elements per group are guaranteed visible before the
reveal schedule starts hiding later elements.

`times` is clamped to `[0, P.total_time]`.

### Loss contract

For `FlowceptionFlow`, the insertion head uses the standard masked count loss:

```julia
X1hat, hat_insertions = model(ts.t, ts.Xt)

loc_loss = floss(P.P[1], X1hat[1], ts.X1anchor[1], scalefloss(P.P[1], ts.Xt.local_t))
disc_loss = floss(P.P[2], X1hat[2], onehot(ts.X1anchor[2]), scalefloss(P.P[2], ts.Xt.local_t))
ins_loss = floss(
    P,
    hat_insertions,
    ts.insertions_target,
    ts.Xt.padmask .* ts.Xt.branchmask,
    scalefloss(P, reshape(ts.t, 1, :)),
)
```

### Sampling contract

During `gen`, `FlowceptionFlow`:

1. advances `local_t` for active elements
2. steps the wrapped base process using `local_t`, not the raw global time
3. samples rightward insertions from the insertion head
4. initializes inserted elements from `birth_sampler`

Inserted elements are fresh source-prior births; they are not parent copies.
Insertion events only occur over the global interval `[0, P.total_time - 1]`;
the final unit of global time is reserved for already-visible elements to
finish their local denoising.

## `DirectionalFlowceptionFlow`

`DirectionalFlowceptionFlow` is a separate extension that leaves
`FlowceptionFlow` untouched. Its bridge is
`directional_flowception_bridge`.

### Why it exists

This path is meant for cases where you want insertions/extensions to either
side of an element while still keeping the model token-centric.

Instead of decoding an explicit slot tensor, the model predicts two insertion
channels per visible token:

- left insertion head
- right insertion head

Same-group adjacent pairs are then pooled into physical insertion slots using
vectorized operations on `groupings`.

### Model contract

For `DirectionalFlowceptionFlow`, the model should return:

1. endpoint predictions `X1hat` for the wrapped base process
2. directional insertion logits with shape `(2, L, B)` or a tuple
   `(left_logits, right_logits)`

Typical model return:

```julia
X1hat, hat_insertions = model(ts.t, ts.Xt)
```

where:

- `hat_insertions[1, :, :]` predicts insertions to the left of each token
- `hat_insertions[2, :, :]` predicts insertions to the right of each token

### Bridge contract

`directional_flowception_bridge` returns:

- `t`
- `Xt`
- `X1anchor`
- `insertions_target` with shape `(2, L, B)`
- `left_insertions_target`
- `right_insertions_target`

As with `flowception_bridge`, `times` is clamped to `[0, P.total_time]`.

The target semantics are:

- `insertions_target[1, i, b]`: hidden elements between the previous visible
  token in the same group and token `i`, or between the group start and token
  `i` if `i` is the first visible token in its group
- `insertions_target[2, i, b]`: hidden elements between token `i` and the next
  visible token in the same group, or between token `i` and the group end if
  `i` is the last visible token in its group

### Loss contract

This is the main API difference relative to `FlowceptionFlow`.

For `DirectionalFlowceptionFlow`, the loss needs the full `FlowceptionState`
instead of just a flat mask, because the pooling is defined by `Xt.groupings`:

```julia
X1hat, hat_insertions = model(ts.t, ts.Xt)

loc_loss = floss(P.P[1], X1hat[1], ts.X1anchor[1], scalefloss(P.P[1], ts.Xt.local_t))
disc_loss = floss(P.P[2], X1hat[2], onehot(ts.X1anchor[2]), scalefloss(P.P[2], ts.Xt.local_t))
ins_loss = floss(
    P,
    hat_insertions,
    ts.insertions_target,
    ts.Xt,
    scalefloss(P, reshape(ts.t, 1, :)),
)
```

Internally the loss does:

- left-boundary slots from `left[first]`
- right-boundary slots from `right[last]`
- same-group interior slots from the mean of `right[i]` and `left[i+1]` in
  rate space

### Sampling contract

Sampling mirrors the same directional semantics:

- group starts may spawn left insertions
- group interiors may spawn pooled insertions between adjacent same-group
  tokens
- group ends may spawn right insertions

Inserted elements inherit the group id of the slot they were created in.

## Minimal examples

### Minimal `FlowceptionFlow` setup

```julia
using BranchingFlows, Flowfusion, ForwardBackward, Distributions

birth_sampler(_) = (
    ContinuousState(randn(Float32, 2, 1)),
    DiscreteState(3, [3]),
)

P = FlowceptionFlow(
    (BrownianMotion(0.05f0), Flowfusion.DistNoisyInterpolatingDiscreteFlow()),
    birth_sampler,
    total_time = 2f0,
)

function make_target()
    locs = MaskedState(ContinuousState(randn(Float32, 2, 12)), trues(12), trues(12))
    toks = MaskedState(DiscreteState(3, rand(1:2, 12)), trues(12), trues(12))
    FlowceptionState((locs, toks), ones(Int, 12))
end

ts = flowception_bridge(P, [make_target() for _ in 1:8], Uniform(0f0, P.total_time))
```

### Minimal `DirectionalFlowceptionFlow` setup

```julia
using BranchingFlows, Flowfusion, ForwardBackward, Distributions

birth_sampler(_) = (
    ContinuousState(randn(Float32, 2, 1)),
    DiscreteState(3, [3]),
)

P = DirectionalFlowceptionFlow(
    (BrownianMotion(0.05f0), Flowfusion.DistNoisyInterpolatingDiscreteFlow()),
    birth_sampler,
    total_time = 2f0,
)

function make_target()
    locs = MaskedState(ContinuousState(randn(Float32, 2, 12)), trues(12), trues(12))
    toks = MaskedState(DiscreteState(3, rand(1:2, 12)), trues(12), trues(12))
    FlowceptionState((locs, toks), ones(Int, 12))
end

ts = directional_flowception_bridge(P, [make_target() for _ in 1:8], Uniform(0f0, P.total_time))
```

## Demos

There are two end-to-end toy demos in `examples/`:

- `examples/flowception_demo.jl`
- `examples/directional_flowception_demo.jl`

Both demos:

- bootstrap `MurrellGroupRegistry` when needed
- use `ONIONop.jl` on `fix-flash-attention-padding`
- use `Onion.jl` on `proteins`
- default to GPU if CUDA is available
- train with `Muon`
- use a linear warmdown schedule

Useful environment variables:

- `CUDA_VISIBLE_DEVICES=1` to keep the demo on the second GPU
- `FLOWCEPTION_DEMO_SKIP_PKG=true` or
  `DIRECTIONAL_FLOWCEPTION_DEMO_SKIP_PKG=true` to skip package bootstrap
- `FLOWCEPTION_DEMO_TOTAL_TIME` or `DIRECTIONAL_FLOWCEPTION_DEMO_TOTAL_TIME` to
  change the global horizon while keeping each element's local denoising window
  at length `1`
- `*_ITERS`, `*_WARMDOWN_STEPS`, `*_BATCH`, `*_DIM`, `*_DEPTH`, `*_LR`,
  `*_NSAMPLES`, `*_SAMPLE_DT` to control training and sampling

## Relationship to the original Branching Flows path

The original Branching Flows API is still:

- `CoalescentFlow`
- `BranchingState`
- `branching_bridge`

Nothing in the Flowception work changes that behavior.

The main shared design choices are:

- same general batching conventions
- same `groupings` semantics
- same style of `branchmask`, `flowmask`, and `padmask`
- same use of `Flowfusion` as the base-process backend

## Full API reference

```@docs
FlowceptionFlow
DirectionalFlowceptionFlow
FlowceptionState
flowception_bridge
directional_flowception_bridge
linear_scheduler
linear_scheduler_derivative
linear_scheduler_inverse
```
