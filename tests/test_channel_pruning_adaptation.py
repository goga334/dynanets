import pytest

from dynanets.config import ExperimentConfig
from dynanets.execution import ExperimentExecutor
from dynanets.experiment import ExperimentAssemblyError, ExperimentBuilder, default_registries


def _builder() -> ExperimentBuilder:
    registries = default_registries()
    return ExperimentBuilder(
        datasets=registries["datasets"],
        models=registries["models"],
        metrics=registries["metrics"],
        adaptations=registries["adaptations"],
        searches=registries["searches"],
        workflows=registries["workflows"],
    )


def test_channel_pruning_adaptation_runs_and_records_prune_event() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "channel-pruning-test",
            "dataset": {
                "name": "synthetic_image_patterns",
                "params": {"train_size": 128, "validation_size": 32, "test_size": 32, "seed": 7},
            },
            "model": {
                "name": "torch_cnn_classifier",
                "params": {
                    "input_channels": 1,
                    "input_size": [28, 28],
                    "num_classes": 10,
                    "conv_channels": [12, 24],
                    "classifier_hidden_dims": [32],
                    "use_batch_norm": True,
                    "lr": 0.001,
                },
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "adaptation": {
                "name": "channel_pruning",
                "params": {"every_n_epochs": 1, "prune_fraction": 0.25, "min_channels_per_block": 8},
            },
            "trainer": {"epochs": 2},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )
    registries = default_registries()
    builder = _builder()
    experiment = builder.build(config)

    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)

    assert result.summary.adaptation_history
    event = result.summary.adaptation_history[0]
    assert event["event_type"] == "prune_channels"
    assert event["effect_summary"]["channels_changed"] is True
    assert event["after_state"]["metadata"]["parameter_count"] < event["before_state"]["metadata"]["parameter_count"]


def test_builder_rejects_channel_pruning_without_batch_norm() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "channel-pruning-invalid-bn",
            "dataset": {
                "name": "synthetic_image_patterns",
                "params": {"train_size": 64, "validation_size": 16, "test_size": 16},
            },
            "model": {
                "name": "torch_cnn_classifier",
                "params": {
                    "input_channels": 1,
                    "input_size": [28, 28],
                    "num_classes": 10,
                    "conv_channels": [8, 16],
                    "classifier_hidden_dims": [16],
                    "use_batch_norm": False,
                    "lr": 0.001,
                },
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "adaptation": {
                "name": "channel_pruning",
                "params": {"every_n_epochs": 1, "prune_fraction": 0.2, "min_channels_per_block": 4},
            },
            "trainer": {"epochs": 2},
        }
    )

    with pytest.raises(ExperimentAssemblyError):
        _builder().build(config)
