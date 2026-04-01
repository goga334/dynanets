import pytest

from dynanets.config import ExperimentConfig
from dynanets.execution import ExperimentExecutor
from dynanets.experiment import ExperimentAssemblyError, ExperimentBuilder, default_registries
from dynanets.models.torch_cnn import TorchCNNClassifier


def _builder():
    registries = default_registries()
    return (
        registries,
        ExperimentBuilder(
            datasets=registries["datasets"],
            models=registries["models"],
            metrics=registries["metrics"],
            adaptations=registries["adaptations"],
            searches=registries["searches"],
            workflows=registries["workflows"],
        ),
    )


def test_torch_cnn_merge_classifier_layers_reduces_depth() -> None:
    model = TorchCNNClassifier(
        input_channels=1,
        input_size=[28, 28],
        num_classes=10,
        conv_channels=[12, 24],
        classifier_hidden_dims=[32, 16],
        use_batch_norm=True,
        lr=0.001,
        device="cpu",
    )

    before = model.structure_state()
    info = model.merge_classifier_layers(merge_index=0)
    after = model.structure_state()

    assert info["merged"] is True
    assert before["metadata"]["classifier_hidden_dims"] == [32, 16]
    assert after["metadata"]["classifier_hidden_dims"] == [16]
    assert after["metadata"]["parameter_count"] < before["metadata"]["parameter_count"]



def test_layermerge_workflow_runs_and_records_merge_event() -> None:
    registries, builder = _builder()
    config = ExperimentConfig.from_dict(
        {
            "name": "layermerge-test",
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
                    "classifier_hidden_dims": [32, 16],
                    "use_batch_norm": True,
                    "lr": 0.001,
                },
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "workflow": {
                "name": "layermerge",
                "params": {"pretrain_epochs": 2, "finetune_epochs": 1, "merge_index": 0},
            },
            "trainer": {"epochs": 3, "batch_size": 32, "eval_batch_size": 32},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )
    experiment = builder.build(config)

    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)

    assert result.summary.workflow_metadata["workflow_name"] == "layermerge"
    assert result.summary.stage_history[0]["name"] == "layermerge_pretrain"
    assert result.summary.stage_history[1]["name"] == "layermerge_finetune"
    assert result.summary.adaptation_history
    assert result.summary.adaptation_history[0]["event_type"] == "merge_hidden_layers"
    assert result.summary.adaptation_history[0]["effect_summary"]["hidden_layers_changed"] is True
    assert result.summary.workflow_metadata["after_classifier_hidden_dims"] == [16]



def test_builder_rejects_layermerge_with_single_classifier_layer() -> None:
    _, builder = _builder()
    config = ExperimentConfig.from_dict(
        {
            "name": "layermerge-invalid-single-layer",
            "dataset": {
                "name": "synthetic_image_patterns",
                "params": {"train_size": 64, "validation_size": 16, "test_size": 16, "seed": 7},
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
            "workflow": {"name": "layermerge", "params": {"pretrain_epochs": 1, "finetune_epochs": 1}},
            "trainer": {"epochs": 2},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )

    with pytest.raises(ExperimentAssemblyError):
        builder.build(config)
