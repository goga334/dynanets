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

The active Wave 1 preview protocol is [benchmarks/track_a_wave1_preview.yaml](../benchmarks/track_a_wave1_preview.yaml).
It currently compares fixed MLP, GradMax, AdaNet, and weights-connections on the 10D two-spirals track.

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
- protocol-driven Wave 1 preview benchmarking

### Phases 5-8

Planned.

Next major unlocks:

- CNN and block architecture specs
- sparsity and compression engine
- routed execution and conditional computation
- official benchmark protocols for Synthetic, MNIST, and CIFAR tracks

## Wave Plan

### Wave 0

Stabilize the existing growth-family methods and keep them as the Synthetic seed track.

### Wave 1

In progress.

Current landing order:

- Han-style pruning is now in place as `weights_connections`
- AdaNet is now in place as the first staged workflow method
- Network Slimming follows once the first CNN/block path lands

Wave 1 closes when the official v1 subset is usable on Synthetic + MNIST through protocol manifests.

### Wave 2

Add deeper pruning/compression methods once CNN and sparsity support are ready.

### Wave 3

Add routing and conditional-computation families.

### Wave 4

Add LayerMerge and tighten protocol outputs into publication-ready benchmark artifacts.
