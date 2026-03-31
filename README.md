# dynanets

Sandbox for experimenting with neural networks that can change structure over time, including NAS-style search and explicit adaptation methods.

## Project Direction

`dynanets` is a research sandbox first and a benchmark arena later.

The immediate goal is not to create a leaderboard as quickly as possible. The immediate goal is to make it easy to take an idea from a paper, express it inside a shared framework, and compare it under increasingly standardized conditions. That means the project is currently focused on representation, extensibility, reproducibility, and fairer comparison primitives.

The longer-term direction is a more competition-like benchmark environment: methods prepare under shared constraints, run under the same conditions, and are compared with explicit budgets, consistent reporting, and reproducible protocols. We are not fully there yet. Right now, the system is better described as a standardized research gym than a finished competition stadium.

A simple way to think about the roadmap is:

- first: can we express the method cleanly?
- next: can we compare methods fairly?
- later: can we benchmark families of methods under strict shared rules?

## Current Scope

The current system supports:

- YAML-defined experiments with pluggable datasets, models, metrics, adaptations, search methods, and workflows
- fixed and dynamic MLP baselines in PyTorch
- fixed CNN baselines with explicit CNN architecture specs and graph export
- paper-inspired dynamic adaptation through Net2Wider-style widening and Net2Deeper-style insertion
- paper-inspired stagewise execution through an AdaNet-style candidate-selection workflow
- paper-inspired pruning through a Han-style "Learning both Weights and Connections" approximation
- paper-inspired NAS through regularized evolution and random search
- comparison reports with markdown, CSV/JSON, plots, architecture diagrams, and stage histories
- explicit architecture specs, mutation primitives, and typed adaptation events for the current MLP family
- an explicit `MLPSearchSpace` separate from the search algorithms
- a generic architecture-graph artifact path for reporting and comparison
- a unified experiment execution layer so train, search, staged workflows, and routing workflows produce a common report surface
- workflow-aware training with `single_stage`, `scheduled`, `adanet_rounds`, routing workflows, and pruning/compression workflows
- protocol manifests for benchmark tracks, starting with the Synthetic seed track
- image dataset support through a runnable synthetic image benchmark and an optional MNIST adapter
- automatic device selection through `runtime.device: auto`, with runtime environment details included in reports

## Phase Status

### Phase 1: Core Stability

Completed.

Phase 1 stabilized the core with:

- explicit config parsing and validation
- build-time compatibility checks for unsupported component combinations
- clearer code-level contracts for model, adaptation, and search interfaces
- contributor documentation for adding a new paper-inspired method

See [docs/adding-methods.md](docs/adding-methods.md) for the current extension workflow.
See [docs/first-paper-batch.md](docs/first-paper-batch.md) for the first paper-family planning target.
See [docs/benchmark-roadmap.md](docs/benchmark-roadmap.md) for the active 20-paper benchmark program.

### Phase 2: Explicit Architecture And Search Representation

Completed.

Phase 2 added:

- `MLPArchitectureSpec` and layer specs in `src/dynanets/architecture/`
- an MLP builder that materializes PyTorch networks from the spec
- mutation primitives over MLP specs
- dynamic MLP updates that use spec mutation as their structural source of truth
- an explicit `MLPSearchSpace` used by search methods

This means architecture representation, mutation, and search are now separated more cleanly for the current MLP family.

### Phase 3: Dynamic Adaptation Framework

In progress.

Phase 3 currently includes:

- multiple dynamic adaptation event types, including growth, pruning, insertion, and removal
- typed adaptation events instead of loose payload dicts
- capability-aware adaptation compatibility checks between methods and models
- structured adaptation history recorded in training summaries and reports
- effect summaries and before/after architecture snapshots for adaptation events
- parameter-count deltas in adaptation effect reporting

Phase 3 is moving the project from isolated dynamic tricks toward a reusable adaptation framework.

### Phase 4: Execution Workflows And Architecture Artifacts

Started.

Phase 4 currently includes:

- a generic architecture graph artifact alongside architecture specs
- shared architecture extraction utilities for train and search runs
- a unified experiment executor that normalizes train and search outputs into one report surface
- workflow-aware execution with stage summaries, including `single_stage`, `scheduled`, `adanet_rounds`, and `network_slimming`
- benchmark protocol manifests and a protocol runner for track-style execution

This phase is about decoupling method execution shape from individual entry points so staged, iterative, or composite methods can fit more naturally later.

### Phase 5: CNN And Block Architecture Families

Started.

Phase 5 currently includes:

- `CNNArchitectureSpec` and CNN graph export
- fixed and batch-normalized PyTorch CNN baselines
- a runnable synthetic image benchmark for CNN smoke tests
- a `Network Slimming` workflow approximation for channel pruning on CNNs
- a synthetic-image benchmark protocol for the CNN Wave 1 preview
- optional MNIST dataset adapters and MNIST-ready CNN configs that activate when `torchvision` is installed

This phase is about getting the first image-ready architecture family into the sandbox so CNN-focused papers have a real landing zone.

### Phase 6: Sparsity, Constraints, And Compression Engine

Started.

Phase 6 currently includes:

- a shared `ConstraintEvaluator` for MLP and CNN architecture specs
- unified constraint summaries in compare and benchmark outputs
- constraint-aware architecture metadata for dynamic MLP and CNN pruning workflows
- constraint deltas in adaptation event histories, including parameter, FLOP, activation, and sparsity changes

This phase is about turning sparsity and model-cost accounting into framework-level infrastructure instead of one-off method logic.

### Phase 7: Routing And Conditional Computation

Started.

Phase 7 currently includes:

- a routed CNN family for width-routing and early-exit style experiments
- workflow-based approximations for `Dynamic Slimmable Network` and `Conditional Computation`
- route-aware execution metadata and route summaries in compare and benchmark reports
- a 5-seed MNIST routing preview protocol

This phase is about making input-dependent execution a first-class concern in the sandbox rather than forcing routing papers into pruning or baseline abstractions.

### Competition Stage

Not started.

The benchmark or "competition stadium" stage should come after the framework phases are mature enough to support fair comparison. In practice, that means we first need broader architecture support, richer training workflows, better persistence, stronger resource accounting, and clearer benchmark protocols. The current phases are what make that later stage possible.

## Running Experiments

```bash
pip install -e .[dev]
python -m dynanets.cli experiments/examples/fixed_mlp.yaml
python -m dynanets.cli experiments/examples/net2wider.yaml
python -m dynanets.cli experiments/examples/regularized_evolution.yaml
python -m dynanets.cli experiments/examples/fixed_cnn_patterns.yaml
```

Generate a comparison report:

```bash
python -m dynanets.compare experiments/examples/fixed_mlp.yaml experiments/examples/net2wider.yaml experiments/examples/regularized_evolution.yaml --output-dir reports/paper_methods
python -m dynanets.compare experiments/examples/fixed_cnn_patterns.yaml experiments/examples/wide_cnn_patterns.yaml --output-dir reports/phase5_cnn_smoke
python -m dynanets.compare experiments/examples/fixed_cnn_patterns.yaml experiments/examples/wide_cnn_patterns_bn.yaml experiments/examples/network_slimming_patterns.yaml --output-dir reports/phase5_network_slimming
```

Generate a multi-seed benchmark summary:

```bash
python -m dynanets.benchmark experiments/examples/fixed_mlp_spirals10.yaml experiments/examples/gradmax_spirals10.yaml --output-dir reports/example_benchmark
```

Run a benchmark protocol manifest:

```bash
python -m dynanets.benchmark_protocol benchmarks/track_a_synthetic_seed.yaml --output-dir reports/track_a_synthetic_seed
python -m dynanets.benchmark_protocol benchmarks/track_a_cnn_wave1_preview.yaml --output-dir reports/track_a_cnn_wave1_preview
python -m dynanets.benchmark_protocol benchmarks/track_b_mnist_phase7_preview.yaml --output-dir reports/track_b_mnist_phase7_preview
```

Run the current Wave 1 preview, including AdaNet-style staged growth:

```bash
python -m dynanets.compare experiments/examples/fixed_mlp_spirals10.yaml experiments/examples/gradmax_spirals10.yaml experiments/examples/adanet_spirals10.yaml experiments/examples/weights_connections_spirals10.yaml --output-dir reports/wave1_adanet_compare
python -m dynanets.benchmark_protocol benchmarks/track_a_wave1_preview.yaml --output-dir reports/track_a_wave1_preview
```

`runtime.device` defaults to `auto`. If PyTorch has CUDA support, dynanets will use the GPU automatically. If reports still show `device=cpu`, check the installed PyTorch build first; a `+cpu` build cannot use the GPU even when the machine has one.

MNIST support is optional at the moment. Install the vision extra before using the MNIST configs or the `track_b_mnist_wave1_preview` protocol:

```bash
pip install -e .[vision]
```

## Current Limitations

- Search and adaptation are separate experiment modes for now.
- The architecture and search space are still narrow and centered on MLPs.
- Dynamic mutations currently support only limited weight-transfer logic.
- Dynamic CNN support is still narrow and currently centers on pruning workflows plus the first routed-CNN preview methods.
- Search spaces are implemented only for MLPs.
- Workflow support is still early and currently focuses on `single_stage`, `scheduled`, `adanet_rounds`, and `network_slimming`.
- The protocol layer currently starts with the Synthetic seed track; MNIST and CIFAR tracks are still planned.

## Suggested Next Steps

1. Continue Phase 3 by hardening the current growth-family methods and adding the next pruning-style approximations.
2. Continue Phase 4 by expressing more methods as explicit workflows instead of simple schedules, with AdaNet now landed as the first staged-method reference.
3. Extend the CNN slice from fixed baselines to dynamic CNN-capable methods.
4. Move the new CNN path onto MNIST, then carry the same abstractions toward CIFAR.
5. Introduce stronger protocol constraints only after the framework can support fair method comparisons.




