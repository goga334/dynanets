import pytest

from dynanets.config import ExperimentConfig
from dynanets.execution import ExperimentExecutor
from dynanets.experiment import ExperimentAssemblyError, ExperimentBuilder, default_registries



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



def test_adanet_workflow_runs_rounds_and_records_selection() -> None:
    config, experiment, registries = _build_experiment(
        {
            "name": "adanet-workflow-test",
            "dataset": {"name": "gaussian_blobs", "params": {"train_size": 64, "validation_size": 32, "seed": 11}},
            "model": {
                "name": "dynamic_mlp_classifier",
                "params": {"input_dim": 2, "hidden_dim": 4, "output_dim": 2, "lr": 0.01},
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "workflow": {
                "name": "adanet_rounds",
                "params": {
                    "warmup_epochs": 1,
                    "rounds": 2,
                    "candidate_epochs": 1,
                    "finetune_epochs": 1,
                    "grow_by": 2,
                    "insert_width": 4,
                    "max_hidden_dim": 8,
                    "max_hidden_layers": 3,
                    "complexity_penalty": 0.01,
                    "include_identity_candidate": False,
                },
            },
            "trainer": {"epochs": 4},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )

    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)
    report = result.to_report_item()

    assert result.mode == "train"
    assert report["metadata"]["method_type"] == "workflow"
    assert [stage["name"] for stage in result.summary.stage_history] == [
        "adanet_warmup",
        "adanet_round_1",
        "adanet_round_2",
        "adanet_consolidate",
    ]
    assert result.summary.workflow_metadata["workflow_name"] == "adanet_rounds"
    assert result.summary.workflow_metadata["round_count"] == 2
    assert result.summary.workflow_metadata["total_candidate_evaluations"] >= 4
    assert len(result.summary.workflow_metadata["rounds"]) == 2
    assert result.summary.adaptation_history
    assert result.summary.adaptation_history[0]["event_type"] in {"net2wider", "insert_hidden_layer"}
    assert result.architecture_spec is not None
    assert "workflow=adanet_rounds" in report["metadata"]["notes"]



def test_builder_rejects_adanet_workflow_with_non_dynamic_model() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "adanet-invalid-model",
            "dataset": {"name": "gaussian_blobs", "params": {"train_size": 64, "validation_size": 32}},
            "model": {
                "name": "torch_mlp_classifier",
                "params": {"input_dim": 2, "hidden_dim": 4, "output_dim": 2, "lr": 0.01},
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "workflow": {
                "name": "adanet_rounds",
                "params": {
                    "warmup_epochs": 1,
                    "rounds": 1,
                    "candidate_epochs": 1,
                    "finetune_epochs": 1,
                },
            },
            "trainer": {"epochs": 3},
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
