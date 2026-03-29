from dynanets.config import ExperimentConfig
from dynanets.experiment import ExperimentBuilder, default_registries
from dynanets.datasets.synthetic import ConcentricCirclesDatasetFactory, TwoSpiralsDatasetFactory
from dynanets.runners.train import TrainingRunner
from dynanets.runtime import set_global_seed



def test_two_spirals_factory_builds_expected_shapes() -> None:
    bundle = TwoSpiralsDatasetFactory(train_size=64, validation_size=32, test_size=16, seed=7).build()

    assert bundle.train.inputs.shape == (64, 2)
    assert bundle.validation is not None
    assert bundle.validation.inputs.shape == (32, 2)
    assert bundle.test is not None
    assert bundle.test.inputs.shape == (16, 2)
    assert bundle.metadata["dataset_name"] == "two_spirals"



def test_two_spirals_factory_supports_high_dimensional_embedding() -> None:
    bundle = TwoSpiralsDatasetFactory(
        train_size=100,
        validation_size=20,
        test_size=20,
        input_dim=20,
        seed=7,
    ).build()

    assert bundle.train.inputs.shape == (100, 20)
    assert bundle.validation is not None
    assert bundle.validation.inputs.shape == (20, 20)
    assert bundle.metadata["input_dim"] == 20



def test_concentric_circles_factory_builds_expected_shapes() -> None:
    bundle = ConcentricCirclesDatasetFactory(train_size=64, validation_size=32, test_size=16, seed=7).build()

    assert bundle.train.inputs.shape == (64, 2)
    assert bundle.validation is not None
    assert bundle.validation.targets.shape == (32,)
    assert bundle.metadata["dataset_name"] == "concentric_circles"



def test_dynamic_pipeline_runs_on_two_spirals() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "two-spirals-pipeline",
            "dataset": {"name": "two_spirals", "params": {"train_size": 128, "validation_size": 64, "test_size": 64, "seed": 7}},
            "model": {
                "name": "dynamic_mlp_classifier",
                "params": {"input_dim": 2, "hidden_dim": 8, "output_dim": 2, "lr": 0.01},
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "adaptation": {
                "name": "den",
                "params": {
                    "metric_name": "accuracy",
                    "patience": 1,
                    "min_delta": 0.01,
                    "grow_by": 2,
                    "max_hidden_dim": 12,
                    "max_expansions": 1,
                    "cooldown_epochs": 1,
                },
            },
            "trainer": {"epochs": 4},
            "runtime": {"seed": 7},
        }
    )

    set_global_seed(config.runtime["seed"])
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
        epochs=4,
        adaptation=experiment.adaptation,
    )

    assert len(summary.train_history) == 4
    assert len(summary.metric_history) == 4
    assert len(summary.adaptation_history) == 4
