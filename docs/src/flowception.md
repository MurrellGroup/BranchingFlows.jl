```@meta
CurrentModule = BranchingFlows
```

# Flowception

This page documents the `FlowceptionFlow` implementation added on top of
`BranchingFlows.jl`.

## What was added

The new API is intentionally parallel to the existing Branching Flows API:

- `FlowceptionFlow`: Flowception-style variable-length process wrapper.
- `FlowceptionState`: sequence state with `local_t` in addition to the usual
  sequence masks.
- `flowception_bridge`: training bridge that samples visible subsequences and
  slot-level insertion targets.

The original `CoalescentFlow`, `BranchingState`, and `branching_bridge` code
paths are unchanged.

## Paper-to-code mapping

The implementation follows the Flowception paper's core construction while
reusing the same style of bookkeeping as `BranchingFlows.jl`.

### State

`FlowceptionState` stores:

- `state`: tuple of Flowfusion states, usually wrapped in `MaskedState`.
- `groupings`: segment ids. Insertions never cross group boundaries.
- `local_t`: per-element denoising times in `[0, 1]`.
- `branchmask`: whether insertions are allowed to the right of an element.
- `flowmask`: whether the element is still denoising.
- `padmask`: valid sequence positions.

This mirrors `BranchingState`, except that Flowception needs explicit
per-element local times rather than a single bridge time plus descendant counts.

### Training bridge

`flowception_bridge(P, X1s, times; nstart=1)` does the Flowception training
sampling step:

1. Sample a global extended time `τg`.
2. Sample reveal delays for each flowable target element using the scheduler
   inverse.
3. Remove not-yet-visible elements.
4. Bridge visible elements from the source prior to their targets using the
   per-element `local_t`.
5. Compute `insertions_target`, the number of missing elements to the right of each
   visible slot.

The returned named tuple mirrors `branching_bridge`:

- `t`: sampled global extended times.
- `Xt`: batched `FlowceptionState`.
- `X1anchor`: visible targets aligned with `Xt`.
- `insertions_target`: slot-level insertion counts.

`X1anchor` uses the existing BranchingFlows naming convention even though, for
Flowception, these are direct visible targets rather than coalescent anchors.

### Sampling step

`Flowfusion.step(P::FlowceptionFlow, Xt, hat, s1, s2)` performs one global-time
step:

1. Advance `local_t` for currently active elements.
2. Denoise existing elements with the wrapped Flowfusion process `P.P`, using
   `local_t` as the per-element time argument.
3. Sample adjacent insertions to the right using Poisson counts based on the
   scheduler hazard and the model's insertion head.
4. Initialize inserted elements from `birth_sampler`.

This reuses the same adjacent insertion convention as the existing
BranchingFlows sampler, but with Flowception's source-prior births instead of
duplicate-the-parent semantics.

## Minimal usage

```julia
using BranchingFlows, Flowfusion, ForwardBackward

birth_sampler(_) = (
    ContinuousState(randn(Float32, 2, 1)),
    DiscreteState(3, [3]),
)

P = FlowceptionFlow(
    (BrownianMotion(0.05f0), Flowfusion.DistNoisyInterpolatingDiscreteFlow()),
    birth_sampler,
)

function make_target()
    locs = MaskedState(ContinuousState(randn(Float32, 2, 12)), trues(12), trues(12))
    toks = MaskedState(DiscreteState(3, rand(1:2, 12)), trues(12), trues(12))
    FlowceptionState((locs, toks), ones(Int, 12))
end

ts = flowception_bridge(P, [make_target() for _ in 1:8], Uniform(0f0, 2f0))
```

At that point a model can consume `ts.t` and `ts.Xt` and predict:

- per-element target states for the wrapped base process,
- per-slot insertion logits/counts.

Training looks very similar to the existing Branching Flows demo:

```julia
X1hat, hat_insertions = model(ts.t, ts.Xt)

loc_loss = floss(P.P[1], X1hat[1], ts.X1anchor[1], scalefloss(P.P[1], ts.Xt.local_t))
tok_loss = floss(P.P[2], X1hat[2], onehot(ts.X1anchor[2]), scalefloss(P.P[2], ts.Xt.local_t))
ins_loss = floss(P, hat_insertions, ts.insertions_target, ts.Xt.padmask .* ts.Xt.branchmask, scalefloss(P, reshape(ts.t, 1, :)))
```

## Conditioning and masking

The Flowception paper distinguishes active and passive context frames. In this
implementation the same effect is expressed with the existing BranchingFlows
mask vocabulary:

- active context frame: `flowmask = false`, `branchmask = true`
- passive context frame: `flowmask = false`, `branchmask = false`
- generated frame: `flowmask = true`, `branchmask = true`

`groupings` can be used to split a sequence into independent insertion regions.
Insertions never cross a group boundary.

## Scheduler hooks

By default the implementation uses the linear scheduler from the paper:

- `linear_scheduler`
- `linear_scheduler_derivative`
- `linear_scheduler_inverse`

These can be replaced when constructing `FlowceptionFlow`, as long as the
inverse is consistent with the scheduler used for training.

## Demo

The toy end-to-end example lives in:

- `examples/flowception_demo.jl`

It is designed to be self-contained:

1. It adds `MurrellGroupRegistry`.
2. It installs `ONIONop.jl` from the `fix-flash-attention-padding` branch.
3. It installs `Onion.jl` from the `proteins` branch.
4. It develops the local checkout of `BranchingFlows.jl`.
5. It can run on GPU by default.
6. It trains with `Muon` and a linear warmdown schedule by default.

To keep it on the second GPU:

```bash
CUDA_VISIBLE_DEVICES=1 julia examples/flowception_demo.jl
```

For repeated runs in an already-prepared environment, set
`FLOWCEPTION_DEMO_SKIP_PKG=true` to skip the demo's self-bootstrap step.
You can also override `FLOWCEPTION_DEMO_ITERS`, `FLOWCEPTION_DEMO_WARMDOWN_STEPS`,
and `FLOWCEPTION_DEMO_LR` from the environment.

## API reference

```@docs
FlowceptionFlow
FlowceptionState
flowception_bridge
linear_scheduler
linear_scheduler_derivative
linear_scheduler_inverse
```
