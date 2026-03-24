# dynanets

Sandbox for experimenting with neural networks that can change structure over time, including NAS-style search and explicit adaptation methods.

## Current Scope

The current system supports:

- YAML-defined experiments with pluggable datasets, models, metrics, adaptations, and search methods
- fixed and dynamic MLP baselines in PyTorch
- paper-inspired dynamic adaptation through Net2Wider-style widening
- paper-inspired NAS through regularized evolution search
- comparison reports with markdown, CSV/JSON, and plots
- an explicit `MLPArchitectureSpec` for the current MLP family

## Phase 1 Status

Phase 1 stabilized the core with:

- explicit config parsing and validation
- build-time compatibility checks for unsupported component combinations
- clearer code-level contracts for model, adaptation, and search interfaces
- contributor documentation for adding a new paper-inspired method

See [docs/adding-methods.md](docs/adding-methods.md) for the current extension workflow.

## Phase 2 Status

Phase 2 has started with explicit architecture representation for the current MLP family.

Implemented so far:

- `MLPArchitectureSpec` and layer specs in `src/dynanets/architecture/`
- an MLP builder that materializes PyTorch networks from the spec
- internal refactoring of current MLP models to hold and update an architecture spec

This is the bridge toward future search spaces and mutation primitives.

## Running Experiments

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
- Dynamic mutations currently update model weights directly and only secondarily update the spec.
- Dataset support is still minimal.
- Multi-layer MLP specs are represented, but the dynamic mutation methods only operate on the first hidden layer today.

## Suggested Next Steps

1. Add mutation primitives over architecture specs.
2. Separate search space definitions from search algorithms.
3. Let dynamic methods mutate specs first, then rebuild models from those mutations.
4. Add a real image dataset such as MNIST.
5. Expand the method library with pruning and random search baselines.