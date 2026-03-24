# Adding A Paper-Inspired Method

This document defines how a paper idea is represented in the current dynanets sandbox.

## Method Shapes

A paper-inspired method currently fits into one of these shapes:

- `model`: the paper mainly proposes an architecture family.
- `adaptation`: the paper modifies structure during training.
- `search`: the paper searches over candidate model configurations.

A full experiment config may combine:

- `dataset`
- `model`
- `metrics`
- `trainer`
- `runtime`
- optionally `adaptation`
- optionally `search`

Current limitation:

- `adaptation` and `search` cannot be enabled in the same config yet.

## Workflow

1. Decide which extension point best matches the paper.
2. Implement the Python class under the relevant package.
3. Register it in `src/dynanets/experiment.py`.
4. Add an example YAML config under `experiments/examples/`.
5. Add at least one test that exercises the new path.
6. Run the comparison pipeline if the method should be benchmarked.

## Adding A Model

Use this when the paper mainly proposes a new network family.

- Implement a `NeuralModel` or `DynamicNeuralModel` in `src/dynanets/models/`.
- If the model supports structural changes, implement `architecture_state()` and `apply_adaptation()`.
- Register it in `default_registries()`.
- Reference it in YAML under `model.name`.

## Adding An Adaptation Method

Use this when the paper changes structure during training.

- Implement `AdaptationMethod` in `src/dynanets/adaptation/`.
- Put the paper-specific control knobs in the constructor.
- Use `maybe_adapt()` to decide when to act.
- Return `AdaptationResult` with a stable `changes` payload.
- Ensure the target `DynamicNeuralModel` understands the adaptation payload.
- Register it in `default_registries()`.
- Reference it in YAML under `adaptation.name`.

## Adding A Search Method

Use this when the paper searches over candidate architectures or hyperparameters.

- Implement `SearchMethod` in `src/dynanets/search/`.
- Keep search-policy logic inside the search class.
- Call the provided `evaluate_candidate()` callback for each candidate.
- Return `SearchResult` with the best candidate and search history.
- Register it in `default_registries()`.
- Reference it in YAML under `search.name`.

## Config Rules

The current config validation expects:

- `name`, `dataset`, and `model` are required.
- `metrics` must contain at least one metric.
- `trainer.epochs` must be a positive integer.
- `runtime.seed` must be an integer when provided.
- `adaptation` requires a `DynamicNeuralModel`.
- `search` and `adaptation` cannot currently coexist in the same config.

## What A Paper Is In This Repo

Today a paper is represented by:

- one registered implementation class
- one YAML config that instantiates it
- optionally one or more comparison runs and reports

This is intentionally lightweight for now. Later phases should introduce explicit architecture specs, search spaces, and mutation primitives.