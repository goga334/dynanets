import pytest

from dynanets.config import ConfigValidationError, ExperimentConfig
from dynanets.experiment import ExperimentAssemblyError, ExperimentBuilder, default_registries
from dynanets.models.base import ArchitectureState, DynamicNeuralModel
from dynanets.registry import Registry
from dynanets.adaptation.base import AdaptationMethod, AdaptationResult


class UnsupportedEventModel(DynamicNeuralModel):
    def forward(self, inputs):
        return inputs

    def training_step(self, batch):
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, batch):
        return batch["inputs"]

    def architecture_state(self) -> ArchitectureState:
        return ArchitectureState(metadata={"hidden_dim": 4, "hidden_dims": [4]})

    def supported_event_types(self) -> set[str]:
        return {"grow_hidden"}

    def capabilities(self) -> dict[str, object]:
        return {"architecture_family": "test", "supported_event_types": ["grow_hidden"]}

    def apply_adaptation(self, event):
        return None


class InsertionOnlyAdaptation(AdaptationMethod):
    def supported_event_types(self) -> set[str]:
        return {"insert_hidden_layer"}

    def maybe_adapt(self, model, state, context):
        return AdaptationResult(applied=False, reason="test")



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



def test_builder_accepts_pruning_adaptation_with_dynamic_model() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "valid-prune-build",
            "dataset": {"name": "gaussian_blobs", "params": {}},
            "model": {"name": "dynamic_mlp_classifier", "params": {"input_dim": 2, "hidden_dim": 6, "output_dim": 2}},
            "metrics": [{"name": "accuracy", "params": {}}],
            "adaptation": {"name": "prune_hidden", "params": {"every_n_epochs": 1, "prune_by": 2, "min_hidden_dim": 4}},
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

    experiment = builder.build(config)

    assert experiment.adaptation is not None



def test_builder_rejects_unsupported_adaptation_event_types() -> None:
    registries = default_registries()
    custom_models: Registry = Registry()
    custom_models.register("unsupported_dynamic", UnsupportedEventModel)
    custom_adaptations: Registry = Registry()
    custom_adaptations.register("insert_only", InsertionOnlyAdaptation)

    builder = ExperimentBuilder(
        datasets=registries["datasets"],
        models=custom_models,
        metrics=registries["metrics"],
        adaptations=custom_adaptations,
        searches=registries["searches"],
    )
    config = ExperimentConfig.from_dict(
        {
            "name": "unsupported-events",
            "dataset": {"name": "gaussian_blobs", "params": {}},
            "model": {"name": "unsupported_dynamic", "params": {}},
            "metrics": [{"name": "accuracy", "params": {}}],
            "adaptation": {"name": "insert_only", "params": {}},
            "trainer": {"epochs": 1},
        }
    )

    with pytest.raises(ExperimentAssemblyError):
        builder.build(config)



def test_config_accepts_workflow_section() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "workflow-config",
            "dataset": {"name": "gaussian_blobs", "params": {}},
            "model": {"name": "dynamic_mlp_classifier", "params": {"input_dim": 2, "hidden_dim": 4, "output_dim": 2}},
            "metrics": [{"name": "accuracy", "params": {}}],
            "workflow": {
                "name": "scheduled",
                "params": {"stages": [{"name": "adapt", "epochs": 1, "adaptation_enabled": True}]},
            },
            "trainer": {"epochs": 1},
        }
    )

    assert config.workflow is not None
    assert config.workflow.name == "scheduled"



def test_config_accepts_runtime_device() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "device-config",
            "dataset": {"name": "gaussian_blobs", "params": {}},
            "model": {"name": "dynamic_mlp_classifier", "params": {"input_dim": 2, "hidden_dim": 4, "output_dim": 2}},
            "metrics": [{"name": "accuracy", "params": {}}],
            "trainer": {"epochs": 1},
            "runtime": {"device": "cpu"},
        }
    )

    assert config.runtime["device"] == "cpu"



def test_scheduled_workflow_rejects_epoch_mismatch() -> None:
    from dynanets.execution import ExperimentExecutor

    config = ExperimentConfig.from_dict(
        {
            "name": "workflow-mismatch",
            "dataset": {"name": "gaussian_blobs", "params": {"train_size": 32, "validation_size": 16}},
            "model": {"name": "dynamic_mlp_classifier", "params": {"input_dim": 2, "hidden_dim": 4, "output_dim": 2}},
            "metrics": [{"name": "accuracy", "params": {}}],
            "workflow": {
                "name": "scheduled",
                "params": {"stages": [{"name": "adapt", "epochs": 1, "adaptation_enabled": True}]},
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
        workflows=registries["workflows"],
    )
    experiment = builder.build(config)

    with pytest.raises(ValueError):
        ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)


def test_config_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ConfigValidationError):
        ExperimentConfig.from_dict(
            {
                "name": "invalid-batch-size",
                "dataset": {"name": "gaussian_blobs", "params": {}},
                "model": {"name": "torch_mlp_classifier", "params": {"input_dim": 2, "hidden_dim": 4, "output_dim": 2}},
                "metrics": [{"name": "accuracy", "params": {}}],
                "trainer": {"epochs": 1, "batch_size": 0},
            }
        )



def test_config_accepts_positive_eval_batch_size() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "valid-eval-batch-size",
            "dataset": {"name": "gaussian_blobs", "params": {}},
            "model": {"name": "torch_mlp_classifier", "params": {"input_dim": 2, "hidden_dim": 4, "output_dim": 2}},
            "metrics": [{"name": "accuracy", "params": {}}],
            "trainer": {"epochs": 1, "batch_size": 8, "eval_batch_size": 16},
        }
    )

    assert config.trainer["batch_size"] == 8
    assert config.trainer["eval_batch_size"] == 16
