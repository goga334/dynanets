import pytest

from dynanets.config import ExperimentConfig
from dynanets.execution import ExperimentExecutor
from dynanets.experiment import ExperimentAssemblyError, ExperimentBuilder, default_registries
from dynanets.models.torch_cnn import TorchCNNClassifier



def _build_experiment(config_dict: dict):
    config = ExperimentConfig.from_dict(config_dict)
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
    return config, experiment, registries



def test_torch_cnn_prune_channels_reduces_conv_widths() -> None:
    model = TorchCNNClassifier(
        input_channels=1,
        input_size=[28, 28],
        num_classes=10,
        conv_channels=[12, 24],
        classifier_hidden_dims=[32],
        use_batch_norm=True,
        lr=0.001,
        device="cpu",
    )
    before = model.structure_state()
    prune_info = model.prune_channels(prune_fraction=0.25, min_channels_per_block=8)
    after = model.structure_state()

    assert prune_info["pruned"] is True
    assert before["metadata"]["conv_channels"] == [12, 24]
    assert after["metadata"]["conv_channels"] == [9, 18]
    assert after["metadata"]["parameter_count"] < before["metadata"]["parameter_count"]



def test_network_slimming_workflow_runs_and_records_prune_event() -> None:
    config, experiment, registries = _build_experiment(
        {
            "name": "network-slimming-test",
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
            "workflow": {
                "name": "network_slimming",
                "params": {
                    "sparse_epochs": 2,
                    "finetune_epochs": 1,
                    "sparsity_strength": 0.001,
                    "prune_fraction": 0.25,
                    "min_channels_per_block": 8,
                },
            },
            "trainer": {"epochs": 3},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )

    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)
    report = result.to_report_item()

    assert result.mode == "train"
    assert report["metadata"]["method_type"] == "workflow"
    assert [stage["name"] for stage in result.summary.stage_history] == [
        "network_slimming_sparse_train",
        "network_slimming_finetune",
    ]
    assert result.summary.workflow_metadata["workflow_name"] == "network_slimming"
    assert result.summary.adaptation_history
    assert result.summary.adaptation_history[0]["event_type"] == "prune_channels"
    assert result.summary.adaptation_history[0]["metadata"]["paper"] == "network-slimming-approx"
    assert result.summary.adaptation_history[0]["effect_summary"]["channels_changed"] is True
    assert "prune_channels" in result.summary.adaptation_history[0]["model_capabilities"]["supported_event_types"]
    assert "workflow=network_slimming" in report["metadata"]["notes"]



def test_builder_rejects_network_slimming_without_batch_norm() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "network-slimming-invalid-bn",
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
            "workflow": {
                "name": "network_slimming",
                "params": {"sparse_epochs": 1, "finetune_epochs": 1},
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

    with pytest.raises(ExperimentAssemblyError):
        builder.build(config)



