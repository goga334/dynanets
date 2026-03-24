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
- an explicit `MLPSearchSpace` separate from the regularized evolution search algorithm

## Phase 1 Status

Phase 1 stabilized the core with:

- explicit config parsing and validation
- build-time compatibility checks for unsupported component combinations
- clearer code-level contracts for model, adaptation, and search interfaces
- contributor documentation for adding a new paper-inspired method

See [docs/adding-methods.md](docs/adding-methods.md) for the current extension workflow.

## Phase 2 Status

Phase 2 now includes:

- `MLPArchitectureSpec` and layer specs in `src/dynanets/architecture/`
- an MLP builder that materializes PyTorch networks from the spec
- mutation primitives over MLP specs
- dynamic MLP updates that use spec mutation as their structural source of truth
- an explicit `MLPSearchSpace` used by regularized evolution

This means architecture representation, mutation, and search are now separated more cleanly for the current MLP family.

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
- The architecture and search space are still narrow and centered on MLPs.
- Dynamic mutations currently support only first-hidden-layer weight transfer logic.
- Dataset support is still minimal.
- Search spaces are implemented only for MLPs.

## Suggested Next Steps

1. Start Phase 3 by broadening dynamic adaptation events beyond width growth.
2. Add pruning and layer insertion methods that operate over architecture specs.
3. Generalize search-space support beyond MLPs.
4. Add a real image dataset such as MNIST.
5. Expand reporting to render searched architecture specs more explicitly.