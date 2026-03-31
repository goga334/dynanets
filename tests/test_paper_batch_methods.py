from dynanets.config import ExperimentConfig
from dynanets.experiment import ExperimentBuilder, default_registries
from dynanets.runners.train import TrainingRunner
from dynanets.runtime import set_global_seed



def _run_training(config: ExperimentConfig):
    set_global_seed(config.runtime.get("seed"))
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
    dataset = experiment.dataset.build()
    return experiment.workflow.execute(
        model=experiment.model,
        dataset=dataset,
        metrics=experiment.metrics,
        training_runner=TrainingRunner(),
        adaptation=experiment.adaptation,
        epochs=int(config.trainer.get("epochs", 1)),
        trainer_config=dict(config.trainer),
    )



def test_gradmax_emits_growth_event() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "gradmax-test",
            "dataset": {"name": "gaussian_blobs", "params": {"train_size": 64, "validation_size": 32, "seed": 11}},
            "model": {"name": "dynamic_mlp_classifier", "params": {"input_dim": 2, "hidden_dim": 4, "output_dim": 2, "lr": 0.01}},
            "metrics": [{"name": "accuracy", "params": {}}],
            "adaptation": {
                "name": "gradmax",
                "params": {"every_n_epochs": 1, "grow_by": 2, "max_hidden_dim": 8, "grad_norm_threshold": 0.0},
            },
            "trainer": {"epochs": 2},
            "runtime": {"seed": 7},
        }
    )

    summary = _run_training(config)
    applied_events = [event for event in summary.adaptation_history if event["applied"]]

    assert applied_events
    assert applied_events[0]["event_type"] == "net2wider"
    assert applied_events[0]["metadata"]["paper"] == "gradmax-approx"



def test_den_expands_after_validation_plateau() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "den-test",
            "dataset": {"name": "gaussian_blobs", "params": {"train_size": 64, "validation_size": 32, "seed": 11}},
            "model": {"name": "dynamic_mlp_classifier", "params": {"input_dim": 2, "hidden_dim": 4, "output_dim": 2, "lr": 0.01}},
            "metrics": [{"name": "accuracy", "params": {}}],
            "adaptation": {
                "name": "den",
                "params": {
                    "metric_name": "accuracy",
                    "patience": 1,
                    "min_delta": 1.0,
                    "grow_by": 2,
                    "max_hidden_dim": 8,
                    "max_expansions": 1,
                    "cooldown_epochs": 0,
                },
            },
            "trainer": {"epochs": 3},
            "runtime": {"seed": 7},
        }
    )

    summary = _run_training(config)
    applied_events = [event for event in summary.adaptation_history if event["applied"]]

    assert applied_events
    assert applied_events[0]["event_type"] == "net2wider"
    assert applied_events[0]["metadata"]["paper"] == "den-approx"
    assert applied_events[0]["metadata"]["trigger"] == "validation_plateau"



def test_nest_runs_growth_then_pruning() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "nest-test",
            "dataset": {"name": "gaussian_blobs", "params": {"train_size": 64, "validation_size": 32, "seed": 11}},
            "model": {"name": "dynamic_mlp_classifier", "params": {"input_dim": 2, "hidden_dim": 6, "output_dim": 2, "lr": 0.01}},
            "metrics": [{"name": "accuracy", "params": {}}],
            "adaptation": {
                "name": "nest",
                "params": {
                    "grow_every_n_epochs": 1,
                    "grow_until_epoch": 2,
                    "grow_by": 2,
                    "max_hidden_dim": 10,
                    "prune_every_n_epochs": 1,
                    "prune_start_epoch": 3,
                    "prune_by": 1,
                    "min_hidden_dim": 6,
                },
            },
            "trainer": {"epochs": 4},
            "runtime": {"seed": 7},
        }
    )

    summary = _run_training(config)
    applied_event_types = [event["event_type"] for event in summary.adaptation_history if event["applied"]]

    assert "net2wider" in applied_event_types
    assert "prune_hidden" in applied_event_types
    assert applied_event_types[:2] == ["net2wider", "net2wider"]



def test_dynamic_nodes_generates_new_layer() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "dynamic-nodes-test",
            "dataset": {"name": "gaussian_blobs", "params": {"train_size": 64, "validation_size": 32, "seed": 11}},
            "model": {"name": "dynamic_mlp_classifier", "params": {"input_dim": 2, "hidden_dim": 6, "output_dim": 2, "lr": 0.01}},
            "metrics": [{"name": "accuracy", "params": {}}],
            "adaptation": {
                "name": "dynamic_nodes",
                "params": {
                    "target_accuracy": 0.999,
                    "generation_width": 6,
                    "max_layers": 2,
                    "prune_every_n_epochs": 10,
                    "prune_by": 1,
                    "min_hidden_dim": 6,
                },
            },
            "trainer": {"epochs": 2},
            "runtime": {"seed": 7},
        }
    )

    summary = _run_training(config)
    applied_events = [event for event in summary.adaptation_history if event["applied"]]

    assert applied_events
    assert applied_events[0]["event_type"] == "insert_hidden_layer"
    assert applied_events[0]["metadata"]["paper"] == "dynamic-nodes-approx"



def test_edge_growth_widens_under_low_accuracy() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "edge-growth-test",
            "dataset": {"name": "gaussian_blobs", "params": {"train_size": 64, "validation_size": 32, "seed": 11}},
            "model": {"name": "dynamic_mlp_classifier", "params": {"input_dim": 2, "hidden_dim": 6, "output_dim": 2, "lr": 0.01}},
            "metrics": [{"name": "accuracy", "params": {}}],
            "adaptation": {
                "name": "edge_growth",
                "params": {
                    "accuracy_floor": 0.999,
                    "grow_by": 2,
                    "max_hidden_dim": 8,
                    "max_layers": 2,
                    "low_accuracy_patience": 1,
                    "layer_width": 6,
                },
            },
            "trainer": {"epochs": 2},
            "runtime": {"seed": 7},
        }
    )

    summary = _run_training(config)
    applied_events = [event for event in summary.adaptation_history if event["applied"]]

    assert applied_events
    assert applied_events[0]["event_type"] == "net2wider"
    assert applied_events[0]["metadata"]["paper"] == "edge-growth-approx"



def test_weights_connections_emits_mask_event() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "weights-connections-test",
            "dataset": {"name": "gaussian_blobs", "params": {"train_size": 64, "validation_size": 32, "seed": 11}},
            "model": {"name": "dynamic_mlp_classifier", "params": {"input_dim": 2, "hidden_dim": 8, "output_dim": 2, "lr": 0.01}},
            "metrics": [{"name": "accuracy", "params": {}}],
            "adaptation": {
                "name": "weights_connections",
                "params": {
                    "start_epoch": 1,
                    "prune_every_n_epochs": 1,
                    "prune_fraction": 0.20,
                    "max_sparsity": 0.40,
                },
            },
            "workflow": {
                "name": "scheduled",
                "params": {
                    "stages": [
                        {"name": "prune", "epochs": 2, "adaptation_enabled": True},
                        {"name": "finetune", "epochs": 1, "adaptation_enabled": False},
                    ]
                },
            },
            "trainer": {"epochs": 3},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )

    summary = _run_training(config)
    applied_events = [event for event in summary.adaptation_history if event["applied"]]

    assert applied_events
    assert applied_events[0]["event_type"] == "apply_weight_mask"
    assert applied_events[0]["metadata"]["paper"] == "weights-connections-approx"
    assert applied_events[0]["effect_summary"]["weight_sparsity_delta"] is not None
    assert summary.stage_history[0]["name"] == "prune"

