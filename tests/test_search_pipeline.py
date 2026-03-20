from dynanets.config import ExperimentConfig
from dynanets.experiment import ExperimentBuilder, default_registries
from dynanets.runners.search import SearchRunner



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
    assert result.best_model_params["hidden_dim"] in {4, 8}