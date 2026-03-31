from dynanets.config import ExperimentConfig
from dynanets.execution import ExperimentExecutor
from dynanets.experiment import ExperimentBuilder, default_registries



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



def test_experiment_executor_returns_training_artifacts() -> None:
    config, experiment, registries = _build_experiment(
        {
            "name": "executor-train-test",
            "dataset": {"name": "gaussian_blobs", "params": {"train_size": 64, "validation_size": 32}},
            "model": {
                "name": "dynamic_mlp_classifier",
                "params": {"input_dim": 2, "hidden_dim": 4, "output_dim": 2, "lr": 0.05},
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "adaptation": {
                "name": "width_growth",
                "params": {"every_n_epochs": 1, "grow_by": 2, "max_hidden_dim": 8},
            },
            "trainer": {"epochs": 3},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )

    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)
    report = result.to_report_item()

    assert result.mode == "train"
    assert result.architecture_spec is not None
    assert result.architecture_graph is not None
    assert result.architecture_graph["metadata"]["family"] == "mlp"
    assert report["metadata"]["method_type"] == "dynamic"
    assert report["adaptations_applied"] >= 1
    assert len(report["stage_history"]) == 1
    assert "device=cpu" in report["metadata"]["notes"]
    assert "requested_device=cpu" in report["metadata"]["notes"]
    assert "cuda_available=" in report["metadata"]["notes"]
    assert report["metadata"]["runtime_environment"]["resolved_device"] == "cpu"
    assert report["constraints"]["parameter_count"] > 0
    assert report["constraints"]["forward_flop_proxy"] > 0



def test_experiment_executor_supports_scheduled_workflow() -> None:
    config, experiment, registries = _build_experiment(
        {
            "name": "executor-workflow-test",
            "dataset": {"name": "gaussian_blobs", "params": {"train_size": 64, "validation_size": 32}},
            "model": {
                "name": "dynamic_mlp_classifier",
                "params": {"input_dim": 2, "hidden_dim": 4, "output_dim": 2, "lr": 0.05},
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "adaptation": {
                "name": "width_growth",
                "params": {"every_n_epochs": 1, "grow_by": 2, "max_hidden_dim": 8},
            },
            "workflow": {
                "name": "scheduled",
                "params": {
                    "stages": [
                        {"name": "adapt", "epochs": 2, "adaptation_enabled": True},
                        {"name": "finetune", "epochs": 1, "adaptation_enabled": False},
                    ]
                },
            },
            "trainer": {"epochs": 3},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )

    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)
    report = result.to_report_item()

    assert len(result.summary.stage_history) == 2
    assert result.summary.stage_history[0]["name"] == "adapt"
    assert result.summary.stage_history[1]["adaptation_enabled"] is False
    assert result.summary.workflow_metadata["executed_total_epochs"] == 3
    assert report["stage_history"][1]["name"] == "finetune"



def test_experiment_executor_returns_search_artifacts() -> None:
    config, experiment, registries = _build_experiment(
        {
            "name": "executor-search-test",
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
            "runtime": {"seed": 9, "device": "cpu"},
        }
    )

    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)
    report = result.to_report_item()

    assert result.mode == "search"
    assert result.best_score is not None
    assert len(result.search_history) == 3
    assert result.architecture_spec is not None
    assert result.architecture_graph is not None
    assert report["metadata"]["method_type"] == "search"
    assert "device=cpu" in report["metadata"]["notes"]
    assert "requested_device=cpu" in report["metadata"]["notes"]
    assert report["metadata"]["runtime_environment"]["resolved_device"] == "cpu"
    assert report["constraints"]["parameter_count"] > 0
    assert report["constraints"]["activation_elements"] > 0
