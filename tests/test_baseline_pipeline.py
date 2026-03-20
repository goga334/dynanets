from dynanets.config import ExperimentConfig
from dynanets.experiment import ExperimentBuilder, default_registries
from dynanets.runners.train import TrainingRunner


def test_dynamic_baseline_pipeline_runs_end_to_end() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "test-run",
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
    dataset = experiment.dataset.build()

    summary = TrainingRunner().run(
        model=experiment.model,
        dataset=dataset,
        metrics=experiment.metrics,
        epochs=3,
        adaptation=experiment.adaptation,
    )

    assert len(summary.train_history) == 3
    assert len(summary.metric_history) == 3
    assert len(summary.adaptation_history) == 3
    assert any(item["applied"] for item in summary.adaptation_history)
