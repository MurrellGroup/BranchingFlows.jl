```@meta
CurrentModule = BranchingFlows
```

# BranchingFlows

Documentation for [BranchingFlows](https://github.com/MurrellGroup/BranchingFlows.jl).

The package now exposes two closely related variable-length interfaces:

- `CoalescentFlow` / `branching_bridge`: the original Branching Flows forest-based construction.
- `FlowceptionFlow` / `flowception_bridge`: the working Flowception path with
  per-element local times and rightward insertions.
- `DirectionalFlowceptionFlow` / `directional_flowception_bridge`: a separate
  bidirectional extension with left/right insertion heads pooled using
  `groupings`.

For the Flowception-specific API, paper links, grouping semantics, and toy
examples, see
[`Flowception`](flowception.md).

## Core API

```@docs
CoalescentFlow
BranchingState
branching_bridge
```

```@index
```
