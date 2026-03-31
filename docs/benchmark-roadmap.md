# Benchmark Roadmap

This document turns the current roadmap into a repo-tracked program.

## Program Shape

- Full target: 20 paper-inspired methods
- Official first benchmark subset: 8 methods
- Fidelity: approximate sandbox-faithful implementations
- Dataset ladder: Synthetic -> MNIST -> CIFAR
- Baselines are tracked separately from the 20-paper target

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
- Network Slimming on the synthetic-image CNN preview track

Active preview protocols:

- [benchmarks/track_a_wave1_preview.yaml](../benchmarks/track_a_wave1_preview.yaml) for MLP-based Wave 1 methods on 10D two-spirals
- [benchmarks/track_a_cnn_wave1_preview.yaml](../benchmarks/track_a_cnn_wave1_preview.yaml) for CNN plus Network Slimming on synthetic image patterns
- [benchmarks/track_b_mnist_wave1_preview.yaml](../benchmarks/track_b_mnist_wave1_preview.yaml) for the MNIST-ready CNN Wave 1 preview once `torchvision` is installed
- [benchmarks/track_b_mnist_phase7_preview.yaml](../benchmarks/track_b_mnist_phase7_preview.yaml) for the first routed-CNN routing preview on MNIST with 5 seeds

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
- protocol-driven Wave 1 preview benchmarking

### Phase 5

Started.

Current implementation focus:

- `CNNArchitectureSpec` and CNN graph export
- fixed and batch-normalized CNN baselines
- a runnable synthetic image benchmark for CNN smoke tests
- a `Network Slimming` approximation on top of the CNN path
- synthetic-image and MNIST-ready protocol manifests for CNN Wave 1 methods
- an optional MNIST dataset adapter gated by `torchvision`

Next unlocks inside this phase:

- broader dynamic CNN-capable methods
- real MNIST execution in environments with `torchvision`
- compact CIFAR-ready CNN families

### Phase 6

Started.

Current implementation focus:

- shared sparsity and constraint evaluation across MLP and CNN families
- constraint-aware compare and benchmark artifacts
- constraint deltas in adaptation and workflow event histories
- groundwork for first-class pruning and compression methods beyond one-off approximations

Next unlocks inside this phase:

- reusable mask-aware sparsity state
- structured pruning primitives for channels, neurons, and layers
- richer constraint evaluators for FLOPs, activation cost, and latency proxies

### Phase 7

Started.

Current implementation focus:

- routed CNN execution with route-aware metadata and cost summaries
- first routing-family workflow approximations for Dynamic Slimmable and Conditional Computation
- MNIST protocol previews with expanded 5-seed evaluation for routing methods

Next unlocks inside this phase:

- richer gate modules and route objectives
- per-sample route accounting in benchmark artifacts
- additional routing papers such as Channel Gating and SkipNet

### Phase 8

Planned.

Next major unlocks:

- official benchmark protocols for Synthetic, MNIST, and CIFAR tracks
- stricter stadium-style reproducibility and acceptance checks

## Wave Plan

### Wave 0

Stabilize the existing growth-family methods and keep them as the Synthetic seed track.

### Wave 1

In progress.

Current landing order:

- Han-style pruning is now in place as `weights_connections`
- AdaNet is now in place as the first staged workflow method
- the first CNN/block path is now in place
- Network Slimming is now in place on the CNN path
- Runtime Neural Pruning, MorphNet, ASFP, and PruneTrain approximations are now extending the CNN pruning track
- the official v1 subset is code-complete at the method/config level

Wave 1 closes operationally when the official v1 subset is runnable on both Synthetic and MNIST protocol tracks in an environment with `torchvision` and CUDA-enabled PyTorch.

### Wave 2

Add deeper pruning/compression methods once CNN and sparsity support are ready.

### Wave 3

In progress.

Current landing order:

- routed CNN execution and route-aware metadata are now in place
- Dynamic Slimmable and Conditional Computation MNIST previews are now runnable
- the routing benchmark preview now uses 5 seeds for a more stable view

### Wave 4

Add LayerMerge and tighten protocol outputs into publication-ready benchmark artifacts.





