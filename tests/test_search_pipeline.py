from dynanets.config import ExperimentConfig
from dynanets.experiment import ExperimentBuilder, default_registries
from dynanets.runners.search import SearchRunner
from dynanets.search import MLPSearchSpace



def test_regularized_evolution_search_runs_end_to_end() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "search-test",
            "dataset": {"name": "gaussian_blobs", "params": {"train_size": 64, "validation_size": 32}},
            "model": {
                "name": "torch_mlp_classifier",
                "params": {"input_dim": 2, "hidden_dim": 4, "output_dim": 2, "lr": 0.01},
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "search": {
                "name": "regularized_evolution",
                "params": {
                    "hidden_dim_choices": [4, 8],
                    "activation_choices": ["relu", "tanh"],
                    "lr_choices": [0.01],
                    "cycles": 3,
                    "population_size": 2,
                    "sample_size": 2,
                    "seed": 7,
                    "metric": "accuracy",
                },
            },
            "trainer": {"epochs": 2},
        }
    )

    registries = default_registries()
    builder = ExperimentBuilder(
        datasets=registries["datasets"],
        models=registries["models"],
        metrics=registries["metrics"],
        adaptations=registries["adaptations"],
        searches=registries["searches"],
    )
    experiment = builder.build(config)

    result = SearchRunner().run(config=config, experiment=experiment, registries=registries)

    assert result.best_score >= 0.0
    assert len(result.evaluation_history) == 3
    assert result.best_model_params["hidden_dims"][0] in {4, 8}
    assert "architecture_spec" in result.evaluation_history[0]["metadata"]



def test_random_search_runs_end_to_end() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "random-search-test",
            "dataset": {"name": "gaussian_blobs", "params": {"train_size": 64, "validation_size": 32}},
            "model": {
                "name": "torch_mlp_classifier",
                "params": {"input_dim": 2, "hidden_dim": 4, "output_dim": 2, "lr": 0.01},
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "search": {
                "name": "random_search",
                "params": {
                    "hidden_dim_choices": [4, 8],
                    "activation_choices": ["relu", "tanh"],
                    "lr_choices": [0.01],
                    "cycles": 3,
                    "seed": 9,
                    "metric": "accuracy",
                },
            },
            "trainer": {"epochs": 2},
        }
    )

    registries = default_registries()
    builder = ExperimentBuilder(
        datasets=registries["datasets"],
        models=registries["models"],
        metrics=registries["metrics"],
        adaptations=registries["adaptations"],
        searches=registries["searches"],
    )
    experiment = builder.build(config)

    result = SearchRunner().run(config=config, experiment=experiment, registries=registries)

    assert result.best_score >= 0.0
    assert len(result.evaluation_history) == 3



def test_mlp_search_space_sampling_and_mutation() -> None:
    import random

    search_space = MLPSearchSpace(
        input_dim=2,
        output_dim=2,
        hidden_dim_choices=[4, 8],
        activation_choices=["relu", "tanh"],
        lr_choices=[0.01, 0.02],
    )
    rng = random.Random(3)

    proposal = search_space.sample(rng)
    mutated = search_space.mutate(proposal, rng)

    assert "architecture_spec" in proposal.metadata
    assert proposal.model_overrides["hidden_dims"][0] in {4, 8}
    assert mutated.model_overrides["hidden_dims"][0] in {4, 8}
    assert mutated.model_overrides["lr"] in {0.01, 0.02}