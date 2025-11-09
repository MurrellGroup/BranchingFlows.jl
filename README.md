# BranchingFlows

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/BranchingFlows.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/BranchingFlows.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/BranchingFlows.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/BranchingFlows.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/BranchingFlows.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/BranchingFlows.jl)

<img width="1382" height="698" alt="Image" src="https://github.com/user-attachments/assets/12d4e6c2-1157-4b16-be80-80de02b2dac5" />

## Overview

BranchingFlows gives a streamlined generator–matching (GM) formulation for
variable–cardinality systems that evolve in continuous time and undergo binary
split events. A sample is an ordered list whose length changes as elements
split; between split times each element follows a continuous Markov generator
(flow, diffusion, or jumps).

We condition on side information `Z = (x₁, T)`, where `x₁` is an ordered
terminal list from the data and `T` is a rooted binary planar topology on these
leaves. For each `(x₁, T)` we construct conditional bridges that respect the
split schedule implied by `T`, and train a parametric generator to match the
true marginal generator via GM.

Key ideas:
- Remaining split budget: for a time `t`, each present lineage `i` carries a
  count `m_i(t; T)` equal to the number of leaves in its cut–subtree minus one
  (the familiar “descendants minus one”). The total budget is `K(t) = Σ_i m_i(t;T)`.
- Hazard and intensities: with a time–varying hazard `h(t)` and survival
  `S(t) = exp(-∫ h)`, the total split rate is `K(t) h(t)` and the per–lineage
  intensity is `λ_i(t) = h(t) m_i(t; T)`.
- Equivalent i.i.d. schedule: attach to each internal node an independent time
  with density `ρ(t) = h(t) S(t)` and sort; executing splits at those times on
  the lineages that still contain the nodes reproduces the same event–time law.
- Unscaled supervision: train the split head to predict `m_i(t; T)` (unscaled);
  multiply by `h(t)` only at simulation time. This factors out the hazard and
  guarantees consistency with the forward process.

## Mathematics (sketch)

Let `(X_t)_{t∈[0,1]}` be a time–inhomogeneous Markov process. Between split
times, each lineage follows a continuous generator `F_t` on its state. Split
events are governed by the budget `K(t)` and hazard `h(t)`:

- Total split rate: `K(t) h(t)`
- Per–lineage split intensity: `λ_i(t) = h(t) m_i(t; T)`

This forward description implies an equivalent i.i.d. assignment of internal–node
times with density `ρ(t) = h(t) S(t)`, giving a convenient scheduler for
training and sampling.

Conditional bridges: for each `(x₁, T)` we build a bridge that reaches the
ordered terminal list at `t=1` and respects the split times of `T`. A practical
implementation places an “anchor” at each internal–node time `τ`, e.g.
`a_τ = w x_τ^(ℓ) + (1-w) x_τ^(r)` for children `(ℓ, r)`, and evaluates the
conditional generator `F_t^{bridge}` between events.

Learning objective: parameterize `F_t^θ` (velocity/diffusion/jump heads) and a
split head `\tilde{μ}_θ(x_t^{(i)}, t)`. Use a GM Bregman divergence for the
continuous part and a Poisson/Bregman loss for the split head with target
`m_i(t; T)`. At test time, simulate with `λ_i(t) = h(t) \tilde{μ}_θ(x_t^{(i)}, t)`.

## Key types and APIs

- `CoalescentFlow(P, branch_time_dist[, policy])`: wraps underlying process `P`
  and adds coalescence dynamics with a pluggable policy. See `merging.jl` for
  policies and traits.
- `BranchingState(state, groupings)`: container for states and group IDs; groups
  restrict which elements may coalesce.
- `branching_bridge(P, X0sampler, X1s, times; ...)`: batch bridge sampler that
  returns masked mini-batches of `(Xt, X1anchor)` plus supervision targets.

## Coalescence policies

Policies live in `src/merging.jl`. Highlights:
- `SequentialUniform()` (sequential): uniformly merge adjacent free intragroup
  pairs.
- `BalancedSequential(alpha)` (sequential): prefer smaller adjacent clusters.
- `CorrelatedSequential(boost, radius)` (sequential): upweight merges near the
  most recent merge location.
- `distance_weighted_coalescence(; state_index, temperature, squared)`
  (non-sequential): weight pairs by a distance kernel; nearest-first fallback.
- `LastToNearest(state_index)` (sequential with append-on-split): sample a group
  proportional to its free size, then merge that group’s last element into its
  nearest same-group neighbor; forward splits append at that group’s end.

You can write your own by implementing:
```julia
select_coalescence(policy, nodes; time)
max_coalescences(policy, nodes)
init!(policy, nodes)              # optional
update!(policy, nodes, i, j, k)   # optional
```

The split insertion mode is controlled by the trait:
```julia
should_append_on_split(policy) -> Bool
```

## Example

```julia
using BranchingFlows, Distributions

P = CoalescentFlow(BrownianMotion(0.1f0), Uniform(0.0f0, 1.0f0), last_to_nearest_coalescence(state_index=1))

# X1s is a Vector{BranchingState} with masked states and groupings
times = rand(Float32, length(X1s))
out = branching_bridge(P, X0sampler, X1s, times; coalescence_factor=0.8)

# forward step during sampling/inference (batch size 1)
X_next = Flowfusion.step(P, X_t, (X1_targets, event_logits), s1, s2)
```

## When to use which policy

- Use Sequential policies when your model leverages sequence order (e.g., text,
  proteins by residue index, or when adjacency encodes locality).
- Use NonSequential policies for sets/point clouds where order is irrelevant; be
  careful that your architecture is permutation-invariant.
- `LastToNearest` is useful when you conceptually “append” new elements and want
  backward coalescence to consume the sequence tails group-by-group.

## Extending

Implement fast candidate enumeration for complex data, e.g., kNN via a KD-tree,
and pass it as the `pairs` generator to `distance_weighted_coalescence`. For
stateful or correlated strategies, use `init!`/`update!` to cache neighbors or
bookkeep last merge locations.
