# dynanets

Sandbox for experimenting with neural networks that can change structure over time, including NAS-style search and explicit adaptation methods.

## Current Scope

The current system supports:

- YAML-defined experiments with pluggable datasets, models, metrics, adaptations, and search methods
- fixed and dynamic MLP baselines in PyTorch
- paper-inspired dynamic adaptation through Net2Wider-style widening
- paper-inspired NAS through regularized evolution search
- comparison reports with markdown, CSV/JSON, and plots

## Phase 1 Status

Phase 1 is about stabilizing the core before broadening the sandbox.

Completed in this phase:

- explicit config parsing and validation
- build-time compatibility checks for unsupported component combinations
- clearer code-level contracts for model, adaptation, and search interfaces
- contributor documentation for adding a new paper-inspired method

See [docs/adding-methods.md](docs/adding-methods.md) for the current extension workflow.

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
  runners/      # train/eval/search orchestration
  search/       # NAS search algorithms
  config.py     # experiment configuration models and validation
  experiment.py # experiment assembly and compatibility checks
  registry.py   # plugin registry
experiments/
  examples/
docs/
  adding-methods.md
reports/
tests/
```

## Running Experiments

Install the package and run one of the example configs:

```bash
pip install -e .[dev]
python -m dynanets.cli experiments/examples/fixed_mlp.yaml
python -m dynanets.cli experiments/examples/net2wider.yaml
python -m dynanets.cli experiments/examples/regularized_evolution.yaml
```

Generate a comparison report:

```bash
python -m dynanets.compare experiments/examples/fixed_mlp.yaml experiments/examples/net2wider.yaml experiments/examples/regularized_evolution.yaml --output-dir reports/paper_methods
```

## Current Limitations

- Search and adaptation are separate experiment modes for now.
- The architecture space is still narrow and centered on MLPs.
- Dynamic mutations are implemented at model level rather than through a generic architecture spec.
- Dataset support is still minimal.

## Suggested Next Steps

1. Introduce an explicit `ArchitectureSpec`.
2. Add mutation primitives over architecture specs.
3. Separate search space definitions from search algorithms.
4. Add a real image dataset such as MNIST.
5. Expand the method library with pruning and random search baselines.