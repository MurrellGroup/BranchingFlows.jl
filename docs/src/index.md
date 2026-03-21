```@meta
CurrentModule = BranchingFlows
```

# BranchingFlows

Documentation for [BranchingFlows](https://github.com/MurrellGroup/BranchingFlows.jl).

The package now exposes two closely related variable-length interfaces:

- `CoalescentFlow` / `branching_bridge`: the original Branching Flows forest-based construction.
- `FlowceptionFlow` / `flowception_bridge`: a left-to-right Flowception-style insertion-and-denoising construction with per-element local times.

For the Flowception-specific API, paper mapping, and toy example, see
[`Flowception`](flowception.md).

```@index
```

```@autodocs
Modules = [BranchingFlows]
```
