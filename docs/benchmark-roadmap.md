# Benchmark Roadmap

This document turns the current roadmap into a repo-tracked program.

## Program Shape

- Full target: 20 paper-inspired methods
- Official first benchmark subset: 8 methods
- Fidelity: approximate sandbox-faithful implementations
- Dataset ladder: Synthetic -> MNIST -> CIFAR
- Baselines are tracked separately from the 20-paper target
- Static comparison families: SqueezeNet-style and CondenseNet-style compact CNNs

## Official v1 Subset

1. Dynamically Growing Neural Network Architecture for Lifelong Deep Learning on the Edge
2. GradMax
3. Lifelong Learning with Dynamically Expandable Networks
4. NeST
5. Adaptive Neural Network Structure Optimization Algorithm Based on Dynamic Nodes
6. AdaNet
7. Learning both Weights and Connections for Efficient Neural Networks
8. Network Slimming

## Current Seed Set

The current synthetic seed track is stored in [benchmarks/track_a_synthetic_seed.yaml](../benchmarks/track_a_synthetic_seed.yaml).
It hardens the currently implemented growth-family methods before the full v1 subset lands.

Included today:

- fixed MLP baseline
- Net2Wider dynamic baseline
- GradMax
- DEN
- NeST
- Dynamic Nodes
- Edge Growth
- Han-style "Learning both Weights and Connections" approximation
- AdaNet-style staged workflow on the Wave 1 preview track
- Network Slimming on the CNN track
- LayerMerge on the MNIST preview track
- SqueezeNet-style and CondenseNet-style compact CNN comparison families

Active preview protocols:

- [benchmarks/track_a_wave1_preview.yaml](../benchmarks/track_a_wave1_preview.yaml) for MLP-based Wave 1 methods on 10D two-spirals
- [benchmarks/track_a_cnn_wave1_preview.yaml](../benchmarks/track_a_cnn_wave1_preview.yaml) for CNN plus Network Slimming on synthetic image patterns
- [benchmarks/track_a_efficient_static_preview.yaml](../benchmarks/track_a_efficient_static_preview.yaml) for efficient static-family comparisons on synthetic image patterns
- [benchmarks/track_b_mnist_wave1_preview.yaml](../benchmarks/track_b_mnist_wave1_preview.yaml) for the MNIST-ready CNN Wave 1 preview
- [benchmarks/track_b_mnist_phase7_preview.yaml](../benchmarks/track_b_mnist_phase7_preview.yaml) for the routed-CNN routing preview on MNIST with 5 seeds
- [benchmarks/track_b_mnist_layermerge_preview.yaml](../benchmarks/track_b_mnist_layermerge_preview.yaml) for the LayerMerge preview on MNIST
- [benchmarks/track_c_cifar_static_preview.yaml](../benchmarks/track_c_cifar_static_preview.yaml) for CIFAR-10 static-family comparisons once CIFAR data is available locally

## Phase Status

### Phase 1

Complete.

### Phase 2

Complete.

### Phase 3

Active.

Current implementation focus:

- typed event coverage
- richer architecture state metadata
- benchmark-visible structural deltas
- hardening of existing growth and grow-prune methods
- weight-mask pruning support for the first sparsity-oriented methods

### Phase 4

Started.

Current implementation focus:

- workflow abstraction above plain train/search
- stage-aware summaries and reports
- protocol-driven benchmark execution
- workflow-ready method implementations for round-based or staged methods

Landed in this phase so far:

- `scheduled` workflow for staged adaptation and finetune runs
- `adanet_rounds` workflow for AdaNet-style candidate selection rounds
- `network_slimming` workflow for sparse-train, prune, and finetune execution
- `layermerge` workflow for merge-and-finetune execution
- protocol-driven Wave 1 and LayerMerge preview benchmarking

### Phase 5

Started.

Current implementation focus:

- `CNNArchitectureSpec` and CNN graph export
- fixed and batch-normalized CNN baselines
- runnable synthetic image and MNIST-ready CNN benchmarks
- optional CIFAR-10 and CIFAR-100 adapters gated by `torchvision`
- compact efficient CNN comparison families for SqueezeNet-style and CondenseNet-style experiments
- routed-ResNet family for harder routing/computation-efficiency benchmarks

Next unlocks inside this phase:

- broader dynamic CNN-capable methods
- real CIFAR execution in environments with local data or network access
- track-C protocol hardening for the static efficient families

### Phase 6

Started.

Current implementation focus:

- shared sparsity and constraint evaluation across MLP and CNN families
- constraint-aware compare and benchmark artifacts
- constraint deltas in adaptation and workflow event histories
- groundwork for first-class pruning and compression methods beyond one-off approximations
- LayerMerge as an explicit staged merge workflow

Next unlocks inside this phase:

- reusable mask-aware sparsity state
- structured pruning primitives for channels, neurons, and layers
- richer constraint evaluators for FLOPs, activation cost, and latency proxies

### Phase 7

Started.

Current implementation focus:

- routed CNN execution with route-aware metadata and cost summaries
- routing-family workflow approximations for Dynamic Slimmable, Conditional Computation, Channel Gating, SkipNet, Instance-wise Sparsity, and IamNN
- MNIST protocol previews with expanded 5-seed evaluation for routing methods and stronger gate-training objectives

Next unlocks inside this phase:

- richer gate modules and route objectives
- per-sample route accounting in benchmark artifacts
- route-quality polishing for the early-exit family
- CIFAR-100 routing-efficiency previews with Pareto-aware reporting and routed-ResNet baselines

### Phase 8

Started.

Current implementation focus:

- official benchmark protocols for Synthetic, MNIST, and CIFAR tracks
- stricter stadium-style reproducibility and acceptance checks
- leaderboard regeneration from manifest-only benchmark definitions
- protocol acceptance and leaderboard artifacts alongside summary reports

Current official manifests:

- [benchmarks/track_a_synthetic_official_v1.yaml](../benchmarks/track_a_synthetic_official_v1.yaml)
- [benchmarks/track_b_mnist_official_v1.yaml](../benchmarks/track_b_mnist_official_v1.yaml)
- [benchmarks/track_c_cifar_official_extended.yaml](../benchmarks/track_c_cifar_official_extended.yaml)

## Wave Plan

### Wave 0

Stabilize the existing growth-family methods and keep them as the Synthetic seed track.

### Wave 1

Implemented at the method/config level.

Current landing order completed:

- Han-style pruning as `weights_connections`
- AdaNet as the first staged workflow method
- the first CNN/block path
- Network Slimming on the CNN path
- Runtime Neural Pruning, MorphNet, ASFP, and PruneTrain on the CNN pruning track
- the official v1 subset code-complete for Synthetic and MNIST-ready execution

Wave 1 closes operationally when the official v1 subset is run under the final Phase 8 stadium protocols.

### Wave 2

Implemented at the method/config level for the current preview scope.

Included today:

- Layer-wise OBS approximation
- Runtime Neural Pruning approximation
- MorphNet approximation
- Asymptotic Soft Filter Pruning approximation
- PruneTrain approximation
- LayerMerge approximation

### Wave 3

In progress.

Current landing order:

- routed CNN execution and route-aware metadata are now in place
- Dynamic Slimmable and Conditional Computation MNIST previews are now runnable
- Channel Gating, SkipNet, Instance-wise Sparsity, and IamNN are all on the shared routing substrate
- the routing benchmark preview now uses 5 seeds for a more stable view

### Wave 4

Started.

Current landing order:

- LayerMerge preview benchmark landed on MNIST
- SqueezeNet-style and CondenseNet-style comparison families landed on the synthetic image track
- CIFAR-10 static-family configs and protocol manifest are in place

Next focus:

- bring CIFAR-10 data online for Track C
- tighten protocol outputs into publication-ready benchmark artifacts
- complete Phase 8 stadium packaging
