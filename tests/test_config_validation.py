import pytest

from dynanets.config import ConfigValidationError, ExperimentConfig
from dynanets.experiment import ExperimentAssemblyError, ExperimentBuilder, default_registries



def test_config_requires_at_least_one_metric() -> None:
    with pytest.raises(ConfigValidationError):
        ExperimentConfig.from_dict(
            {
                "name": "invalid",
                "dataset": {"name": "gaussian_blobs", "params": {}},
                "model": {"name": "torch_mlp_classifier", "params": {"input_dim": 2, "hidden_dim": 4, "output_dim": 2}},
                "metrics": [],
                "trainer": {"epochs": 1},
            }
        )



def test_config_rejects_search_and_adaptation_together() -> None:
    with pytest.raises(ConfigValidationError):
        ExperimentConfig.from_dict(
            {
                "name": "invalid",
                "dataset": {"name": "gaussian_blobs", "params": {}},
                "model": {"name": "dynamic_mlp_classifier", "params": {"input_dim": 2, "hidden_dim": 4, "output_dim": 2}},
                "metrics": [{"name": "accuracy", "params": {}}],
                "adaptation": {"name": "net2wider", "params": {"every_n_epochs": 1, "grow_by": 2}},
                "search": {
                    "name": "regularized_evolution",
                    "params": {
                        "hidden_dim_choices": [4, 8],
                        "activation_choices": ["relu"],
                        "lr_choices": [0.01],
                    },
                },
                "trainer": {"epochs": 1},
            }
        )



def test_builder_rejects_adaptation_with_non_dynamic_model() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "invalid-build",
            "dataset": {"name": "gaussian_blobs", "params": {}},
            "model": {"name": "torch_mlp_classifier", "params": {"input_dim": 2, "hidden_dim": 4, "output_dim": 2}},
            "metrics": [{"name": "accuracy", "params": {}}],
            "adaptation": {"name": "net2wider", "params": {"every_n_epochs": 1, "grow_by": 2}},
            "trainer": {"epochs": 1},
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

    with pytest.raises(ExperimentAssemblyError):
        builder.build(config)