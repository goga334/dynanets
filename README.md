# dynanets

Sandbox for experimenting with neural networks that can change structure over time, including NAS-style search and explicit adaptation methods.

## Goals

- Keep experiments modular: datasets, models, metrics, and adaptation/search methods should be swappable.
- Support dynamic-structure neural networks as a first-class concept, not an afterthought.
- Make research iteration easy: small configs, repeatable runs, clear extension points.
- Avoid locking the project into one NAS strategy, training loop, or dataset format too early.

## Development Plan

### Phase 1: Research Sandbox Foundation

- Define stable interfaces for datasets, models, metrics, adaptation methods, and search algorithms.
- Build experiment configuration and registration mechanisms.
- Add a minimal training/evaluation runner that wires components together.
- Keep implementation lightweight while the research workflow is still evolving.

### Phase 2: Baseline Experiment Stack

- Add standard supervised baselines for fixed-topology neural networks.
- Add at least one dynamic-structure model family.
- Add dataset adapters for tabular and image experiments.
- Add baseline metrics and logging.

### Phase 3: Dynamic Adaptation Layer

- Formalize adaptation events such as grow, prune, rewire, or replace-block.
- Track architecture state across training.
- Make adaptation policies callable from the training loop and from search algorithms.

### Phase 4: NAS Methods

- Add search spaces for neural architectures.
- Add search strategies such as random search, evolutionary search, and controller/policy-driven search.
- Separate architecture proposal from training/evaluation so methods remain composable.

### Phase 5: Experiment Management

- Add reproducible config files for runs.
- Add result serialization and run comparison.
- Add test coverage around registries, configs, and experiment assembly.

## Initial Structure

```text
src/dynanets/
  adaptation/   # structure-change methods and policies
  datasets/     # dataset adapters and loaders
  metrics/      # evaluation metrics
  models/       # fixed and dynamic neural network abstractions
  runners/      # train/eval orchestration
  search/       # NAS search spaces and algorithms
  config.py     # experiment configuration models
  experiment.py # experiment assembly
  registry.py   # plugin registry
experiments/
  README.md
  examples/
tests/
```

## Core Design Choices

- `DatasetFactory` creates train/validation/test splits from config.
- `NeuralModel` is the common model interface.
- `DynamicNeuralModel` extends the base model with architecture state and mutation hooks.
- `Metric` calculates evaluation outputs independently of the runner.
- `AdaptationMethod` decides how and when the structure changes during training.
- `SearchMethod` proposes model/adaptation configurations to evaluate.

This separation keeps dynamic adaptation and NAS connected, but not coupled:

- adaptation methods can modify a running model,
- search methods can choose architectures or adaptation policies,
- the runner can stay mostly generic.

## Baselines Implemented

- `gaussian_blobs`: synthetic classification dataset with train/validation/test splits.
- `torch_mlp_classifier`: fixed-topology PyTorch MLP classifier baseline.
- `dynamic_mlp_classifier`: PyTorch MLP baseline with mutable hidden width.
- `accuracy`: classification accuracy metric.
- `width_growth`: first adaptation method that expands hidden width on a fixed epoch schedule.

## Running Experiments

Install the package and run one of the example configs:

```bash
pip install -e .
python -m dynanets.cli experiments/examples/fixed_mlp.yaml
python -m dynanets.cli experiments/examples/minimal.yaml
```

`fixed_mlp.yaml` runs the fixed-topology baseline.
`minimal.yaml` runs the adaptive baseline with scheduled width growth.

## Suggested Next Steps

1. Replace full-batch training with minibatch data loaders.
2. Add a real dataset adapter such as MNIST or FashionMNIST.
3. Add result serialization and per-epoch logging.
4. Extend adaptation methods beyond width growth to pruning or rewiring.
5. Introduce a compact NAS search space over hidden depth, width, and adaptation policy.
