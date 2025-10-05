# BranchingFlows

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/BranchingFlows.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/BranchingFlows.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/BranchingFlows.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/BranchingFlows.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/BranchingFlows.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/BranchingFlows.jl)

## Overview

BranchingFlows implements generator matching for variable-cardinality states via
tree-structured coalescent/branching dynamics. Unlike classical flow matching,
which transports fixed-size tensors, this framework allows elements to be
inserted and coalesced along the path, enabling generative modeling of ordered
sequences, sets, and mixed discrete/continuous structures (e.g., molecules,
proteins) where topology itself evolves.

At a high level:
- Backward-time: we start from observations at time 1 and sample a stochastic
  coalescent forest of event times and pairings (per a policy). Between events
  we run bridges under an underlying state-space process (e.g., Brownian,
  manifold dynamics) to time `t` and assemble training pairs `(Xt, X1anchor)`
  with descendant counts.
- Forward-time: we simulate `Flowfusion.step` under the same process and sample
  split events according to the learned generator. Insertions occur either
  adjacently or at the end of each group block depending on the policy trait.

This unified view lets you learn distributions over objects that grow/shrink via
insert/delete-like operations while maintaining a consistent diffusion-like
generator.

## Mathematics (sketch)

Let `X_s` be the evolving state over `s ∈ [0,1]`. Between events the marginal
follows a user-chosen diffusion or Markov process `P`, and at random event times
`τ_k` (sampled from `branch_time_dist`) two elements coalesce (backward) or a
single element splits (forward). For training, we construct pairs by running a
bridge from known boundary `X_1` to `t` along sampled trees and minimize a
generator matching loss between model velocity and bridge velocity at `t`.

Split intensity target at time `t` for an element with `splits` descendants is
`λ*(t) = splits` (unscaled hazard). During forward simulation we multiply by the
truncated branch time density at `t` (and map logits through `split_transform`)
to obtain Poisson mean for split counts within `[s₁,s₂]`.

Coalescence selection is governed by a policy `π` (Sequential or NonSequential):
given current nodes, `π` returns a pair `(i,j)` to merge and an upper bound on
how many merges remain. This keeps the forest sampling modular and allows
custom, data-dependent rules.

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
