# First Paper Batch

This document maps the first planned comparison batch to framework capabilities. The goal is not to reproduce every paper exactly on the first pass. The goal is to keep the sandbox moving toward a design where these methods can be implemented approximately without ad hoc framework surgery.

## Method Families

### Growth And Expansion

These methods grow width, depth, blocks, or subnetworks over time:

- Dynamically Growing Neural Network Architecture for Lifelong Deep Learning on the Edge
- GradMax
- Lifelong Learning with Dynamically Expandable Networks
- NeST
- Adaptive Neural Network Structure Optimization Algorithm Based on Dynamic Nodes
- A self-organising network that grows when required
- AdaNet
- Always-Sparse Training by Growing Connections with Guided Stochastic Exploration

Framework implications:

- richer growth event vocabulary beyond width and layer insertion
- stagewise and lifelong training workflows
- support for grow-and-prune cycles
- support for task or regime changes over time
- architecture state snapshots that include more than hidden width

### Pruning And Compression

These methods remove weights, channels, filters, layers, or structured groups:

- Learning both Weights and Connections for Efficient Neural Networks
- The Combinatorial Brain Surgeon
- Network Slimming
- Runtime Neural Pruning
- Deep Compression
- MorphNet
- Learning to Prune Deep Neural Networks via Layer-wise Optimal Brain Surgeon
- Asymptotic Soft Filter Pruning
- Neuroplasticity-Based Pruning Method for Deep Convolutional Neural Networks
- CUP
- LayerMerge
- PruneTrain

Framework implications:

- structured pruning events at weight, neuron, channel, and layer level
- sparsity-aware metrics and reporting
- constraint-aware objectives such as parameter count and FLOPs
- post-prune fine-tuning workflows
- support for masks, merge operations, and soft pruning schedules

### Conditional Computation And Routing

These methods change the computation path per input or per runtime condition:

- SkipNet
- IamNN
- Conditional Computation in Neural Networks for faster models
- Dynamic Slimmable Network
- Learning Instance-wise Sparsity for Accelerating Deep Models
- Channel Gating Neural Networks

Framework implications:

- graph-style architectures and routing-aware execution
- per-instance decisions, gates, or policies
- dynamic inference metrics in addition to training metrics
- budget-aware evaluation under latency or activation cost
- support for runtime path traces in reports

### Efficient Static Baselines And Architecture Families

These methods are not always dynamic in the same sense, but they are important comparison families:

- SqueezeNet
- CondenseNet

Framework implications:

- CNN architecture families must become first-class citizens
- architecture specs must support stages, blocks, and grouped operations
- reports must compare dynamic methods against efficient static baselines, not only simple MLPs

## Cross-Cutting Capabilities We Still Need

Across this paper batch, the most important missing capabilities are:

1. CNN and block-based architecture specs
2. graph or routed execution models
3. structured sparsity and mask support
4. constraint-aware objectives and reporting
5. stagewise and lifelong training workflows
6. richer benchmark datasets beyond synthetic blobs
7. persistent run metadata for longer experiments

## Suggested Adoption Order

A practical order for onboarding the first paper batch is:

1. grow-and-prune methods over MLPs
Examples: NeST-inspired, GradMax-inspired, pruning baselines
2. CNN baselines and structured pruning over CNNs
Examples: Network Slimming, soft filter pruning, MorphNet-inspired approximations
3. lifelong and expandable methods
Examples: DEN-inspired, edge growth papers, AdaNet-inspired stagewise growth
4. conditional computation and routing methods
Examples: SkipNet, Dynamic Slimmable Network, Channel Gating

This order keeps the framework expanding gradually instead of jumping directly from MLP mutations to routing-heavy CNN methods.

## Near-Term Design Guidance

When adding new framework features, prefer changes that directly support multiple papers from this list. In practice, that means prioritizing:

- typed events that can generalize to grow, prune, merge, gate, and route operations
- architecture specs that can describe CNN blocks and graph structure
- training runners that can support staged and multi-phase optimization
- reports that capture structural deltas, constraint deltas, and runtime behavior
