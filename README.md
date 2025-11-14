# BranchingFlows.jl

[![arXiv](https://img.shields.io/badge/arXiv-2511.09465-B31B1B.svg)](https://arxiv.org/abs/2511.09465)

## Graphical Abstract
<img width="1382" height="698" alt="Image" src="https://github.com/user-attachments/assets/12d4e6c2-1157-4b16-be80-80de02b2dac5" />

## Video Abstract
<video src="https://github.com/user-attachments/assets/36b20824-6b71-491b-8040-436369f8d4fe" controls></video>
<video src="https://github.com/user-attachments/assets/a95a126a-22e6-4fd3-aa41-c5918489d5ec" controls></video>

BranchingFlows.jl implements Branching Flows: a generator-matching framework for variable-length generation across continuous, manifold, discrete, and multimodal states. Elements evolve along a pre-sampled binary forest; they duplicate (insert) and may be deleted according to time-inhomogeneous hazards, while a base Markov generator transports states along each branch and guarantees termination at the data sample at t=1.

Unlike fixed-length diffusion/flow models, Branching Flows learns when to create and remove elements. Training uses Generator Matching with simple targets: the expected number of future splits and the probability of terminal deletion for each element, plus a standard base-process objective. The package is designed to compose with [Flowfusion.jl](https://github.com/MurrellGroup/Flowfusion.jl), supporting continuous, manifold, and discrete components.

## Why Branching Flows?

- Variable-length generation without autoregression
- Works for continuous, manifold, discrete, and mixed state spaces
- Stable training targets
- Manipulate the conditional paths by changing how you sample forests, and anchors.

## How it works (high level)

1) Sample Z:
We draw `(X₁, Z)` where `X₁ ~ q` and `Z = (X₁^{+ø}, X₀, 𝒯, 𝒜)` encodes:
- `X₀`: initial elements (count and states),  
- `X₁^{+ø}`: data with “to-be-deleted” elements inserted,  
- `𝒯`: a forest of binary plane trees mapping each `X₀` root to leaves of `X₁^{+ø}`,  
- `𝒜`: anchors for all internal/leaf nodes (surviving leaves’ anchors equal the data elements).

2) Conditional paths  
Conditioned on `Z`, each element follows the base generator along its branch. At split times (sampled from a hazard `H_split` with intensity `h_split(t)·(w_i−1)`), the element duplicates and the copies decouple. If a leaf is flagged “deleted,” a deletion hazard `H_del` applies along that branch; otherwise it is zero. This ensures termination at `X₁`.

3) Training targets and loss  
For each element at time t, the model predicts:
- expected remaining splits by t=1 (counting target),
- probability of terminal deletion (binary target),
- base-process field/velocity/logits (application dependent).

The per-element loss is a sum of Bregman divergences:
- counting (Poisson-style) for splits,  
- cross-entropy for deletion probability,  
- a separable base-process divergence (e.g., MSE/DFM).  

These targets are linear in the conditional generator and valid under Generator Matching.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/MurrellGroup/BranchingFlows.jl")
```

## Demo

For a full end-to-end demo (incl. a small Transformer and all losses), see:
- `examples/demo.jl`

## Core API

- `CoalescentFlow(P, branch_time_dist, policy=SequentialUniform(), deletion_time_dist=Uniform(0,1))`  
  Wraps base process(es) `P` and injects branching/deletion with time hazards.  
  The policy controls which elements coalesce when constructing forests for training bridges.

- `BranchingState(state, groupings; del=zeros(Bool,…), ids, branchmask, flowmask, padmask)`  
  Batched state with per-element group IDs; elements only coalesce within groups.

- `branching_bridge(P, X0sampler, X1s, times; use_branching_time_prob=0, length_mins=nothing, deletion_pad=0, coalescence_factor=1.0, merger=canonical_anchor_merge, coalescence_policy=P.coalescence_policy)`  
  Samples a forest per `(X₁, t)`, runs base-process bridges along branches, and returns batched bridge states plus training targets:
  - `Xt`, `X1anchor`, `splits_target`, `del`, `descendants`, `prev_coalescence`, and masks.

### Utilities

- Anchor strategies  
  - `canonical_anchor_merge`: weighted average/geodesic for continuous/manifold; mask for discrete.  
  - `select_anchor_merge`: copy-a-child alternative (no interpolation).

- Deletion augmentation for `X₁`  
  - `uniform_del_insertions(X1, p)`  
  - `fixedcount_del_insertions(X1, num_events)`  
  - `group_fixedcount_del_insertions(X1, group_num_events::Dict)`

- Coalescence policy  
  - `SequentialUniform()`: uniformly coalesce adjacent eligible pairs within groups.

## Practical notes

- Grouping: elements only coalesce within a group (e.g., per protein chain, or per designable segment).  
- `length_mins` lets you enforce per-group minimum lengths when sampling forests.  
- `deletion_pad` inserts “to-be-deleted” elements into `X₁` during training to ensure `|X₁^{+ø}| ≥ |X₀|` per group.  
- For continuous states, anchors can be sampled via geodesic interpolation; for discrete states.

## Applications tested so far

- QM9 molecules: joint atom positions + labels.
- Antibody sequences: discrete-only.
- Proteins (BF-ChainStorm): multimodal (Euclidean + SO(3) frames + AA labels), unconditional and conditioned generation; supports “unknown-length infix” design.

## Citing

```bibtex
@misc{nordlinder2025branchingflowsdiscretecontinuous,
      title={Branching Flows: Discrete, Continuous, and Manifold Flow Matching with Splits and Deletions}, 
      author={Hedwig Nora Nordlinder and Lukas Billera and Jack Collier Ryder and Anton Oresten and Aron Stålmarck and Theodor Mosetti Björk and Ben Murrell},
      year={2025},
      eprint={2511.09465},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2511.09465}, 
}
```

## License

MIT (see `LICENSE`).
