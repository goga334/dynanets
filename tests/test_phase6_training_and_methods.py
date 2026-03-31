from dynanets.config import ExperimentConfig
from dynanets.execution import ExperimentExecutor
from dynanets.experiment import ExperimentBuilder, default_registries
from dynanets.runners.train import TrainingRunner



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



def test_training_runner_uses_minibatches_for_tensor_splits() -> None:
    registries, builder = _builder()
    config = ExperimentConfig.from_dict(
        {
            "name": "batched-train-test",
            "dataset": {"name": "gaussian_blobs", "params": {"train_size": 32, "validation_size": 16, "seed": 5}},
            "model": {
                "name": "dynamic_mlp_classifier",
                "params": {"input_dim": 2, "hidden_dim": 4, "output_dim": 2, "lr": 0.01},
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "trainer": {"epochs": 1, "batch_size": 8, "eval_batch_size": 4},
            "runtime": {"seed": 5, "device": "cpu"},
        }
    )
    experiment = builder.build(config)
    dataset = experiment.dataset.build()

    summary = TrainingRunner().run(
        model=experiment.model,
        dataset=dataset,
        metrics=experiment.metrics,
        epochs=1,
        trainer_config=dict(config.trainer),
    )

    assert len(summary.train_history) == 1
    assert len(summary.metric_history) == 1
    assert experiment.model.architecture_state().step == 4



def test_runtime_neural_pruning_runs_on_cnn() -> None:
    registries, builder = _builder()
    config = ExperimentConfig.from_dict(
        {
            "name": "runtime-pruning-test",
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
                "name": "runtime_neural_pruning",
                "params": {
                    "start_epoch": 2,
                    "prune_every_n_epochs": 1,
                    "prune_fraction": 0.15,
                    "min_channels_per_block": 8,
                    "target_activation_reduction": 0.10,
                },
            },
            "trainer": {"epochs": 3, "batch_size": 32, "eval_batch_size": 32},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )
    experiment = builder.build(config)

    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)

    assert result.summary.adaptation_history
    applied = [item for item in result.summary.adaptation_history if item.get("applied")]
    assert applied
    assert applied[0]["event_type"] == "prune_channels"
    assert applied[0]["metadata"]["paper"] == "runtime-neural-pruning-approx"



def test_morphnet_workflow_runs_and_records_prune_event() -> None:
    registries, builder = _builder()
    config = ExperimentConfig.from_dict(
        {
            "name": "morphnet-test",
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
                "name": "morphnet",
                "params": {
                    "sparse_epochs": 2,
                    "finetune_epochs": 1,
                    "sparsity_strength": 0.001,
                    "target_flop_reduction": 0.20,
                    "max_prune_fraction": 0.25,
                    "min_channels_per_block": 8,
                },
            },
            "trainer": {"epochs": 3, "batch_size": 32, "eval_batch_size": 32},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )
    experiment = builder.build(config)

    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)

    assert result.summary.workflow_metadata["workflow_name"] == "morphnet"
    assert result.summary.adaptation_history
    assert result.summary.adaptation_history[0]["metadata"]["paper"] == "morphnet-approx"
    assert result.summary.adaptation_history[0]["effect_summary"]["channels_changed"] is True


def test_asfp_runs_on_cnn() -> None:
    registries, builder = _builder()
    config = ExperimentConfig.from_dict(
        {
            "name": "asfp-test",
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
                "name": "soft_filter_pruning",
                "params": {
                    "warmup_epochs": 1,
                    "prune_every_n_epochs": 1,
                    "total_prune_epochs": 3,
                    "min_prune_fraction": 0.05,
                    "final_prune_fraction": 0.20,
                    "min_channels_per_block": 8,
                },
            },
            "trainer": {"epochs": 4, "batch_size": 32, "eval_batch_size": 32},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )
    experiment = builder.build(config)

    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)

    applied = [item for item in result.summary.adaptation_history if item.get("applied")]
    assert applied
    assert applied[0]["metadata"]["paper"] == "asfp-approx"



def test_prunetrain_workflow_runs_and_records_reconfiguration() -> None:
    registries, builder = _builder()
    config = ExperimentConfig.from_dict(
        {
            "name": "prunetrain-test",
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
                "name": "prunetrain",
                "params": {
                    "reconfigure_epochs": [2, 3],
                    "prune_fraction": 0.15,
                    "min_channels_per_block": 8,
                },
            },
            "trainer": {"epochs": 4, "batch_size": 32, "eval_batch_size": 32},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )
    experiment = builder.build(config)

    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)

    assert result.summary.workflow_metadata["workflow_name"] == "prunetrain"
    assert len(result.summary.adaptation_history) == 2
    assert result.summary.adaptation_history[0]["metadata"]["paper"] == "prunetrain-approx"
