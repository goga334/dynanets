from dynanets.config import ExperimentConfig
from dynanets.execution import ExperimentExecutor
from dynanets.experiment import ExperimentAssemblyError, ExperimentBuilder, default_registries
from dynanets.models.torch_routed_cnn import TorchRoutedCNNClassifier
import torch


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


def test_dynamic_slimmable_workflow_runs_and_records_route_summary() -> None:
    registries, builder = _builder()
    config = ExperimentConfig.from_dict(
        {
            "name": "dynamic-slimmable-test",
            "dataset": {
                "name": "synthetic_image_patterns",
                "params": {"train_size": 128, "validation_size": 32, "test_size": 32, "seed": 7},
            },
            "model": {
                "name": "torch_routed_cnn_classifier",
                "params": {
                    "input_channels": 1,
                    "input_size": [28, 28],
                    "num_classes": 10,
                    "conv_channels": [12, 24],
                    "lr": 0.001,
                    "width_multipliers": [1.0, 0.75, 0.5],
                    "eval_width_multipliers": [0.5, 0.75, 1.0],
                    "routing_policy": "dynamic_width",
                    "confidence_threshold": 0.8,
                    "min_confidence_threshold": 0.4,
                    "gate_metric": "margin",
                    "gate_budget_weight": 0.05,
                    "target_cost_ratio": 0.8,
                },
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "workflow": {"name": "dynamic_slimmable", "params": {}},
            "trainer": {"epochs": 2, "batch_size": 32, "eval_batch_size": 32},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )
    experiment = builder.build(config)

    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)

    assert result.summary.workflow_metadata["workflow_name"] == "dynamic_slimmable"
    assert result.metadata["route_summary"]["policy"] == "dynamic_width"
    assert result.metadata["route_summary"]["gate_metric"] == "margin"
    assert "mean_cost_ratio" in result.metadata["route_summary"]
    assert result.architecture_graph is not None


def test_conditional_computation_workflow_runs_and_records_route_summary() -> None:
    registries, builder = _builder()
    config = ExperimentConfig.from_dict(
        {
            "name": "conditional-computation-test",
            "dataset": {
                "name": "synthetic_image_patterns",
                "params": {"train_size": 128, "validation_size": 32, "test_size": 32, "seed": 7},
            },
            "model": {
                "name": "torch_routed_cnn_classifier",
                "params": {
                    "input_channels": 1,
                    "input_size": [28, 28],
                    "num_classes": 10,
                    "conv_channels": [12, 24],
                    "lr": 0.001,
                    "width_multipliers": [1.0, 0.75, 0.5],
                    "routing_policy": "early_exit",
                    "confidence_threshold": 0.9,
                    "min_confidence_threshold": 0.4,
                    "gate_metric": "margin",
                },
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "workflow": {"name": "conditional_computation", "params": {}},
            "trainer": {"epochs": 2, "batch_size": 32, "eval_batch_size": 32},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )
    experiment = builder.build(config)

    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)

    assert result.summary.workflow_metadata["workflow_name"] == "conditional_computation"
    assert result.metadata["route_summary"]["policy"] == "early_exit"
    assert result.metadata["route_summary"]["gate_metric"] == "margin"
    assert "early_exit_fraction" in result.metadata["route_summary"]
    assert result.architecture_graph is not None


def test_builder_rejects_dynamic_slimmable_with_wrong_policy() -> None:
    _, builder = _builder()
    config = ExperimentConfig.from_dict(
        {
            "name": "dynamic-slimmable-invalid",
            "dataset": {
                "name": "synthetic_image_patterns",
                "params": {"train_size": 64, "validation_size": 16, "test_size": 16, "seed": 7},
            },
            "model": {
                "name": "torch_routed_cnn_classifier",
                "params": {
                    "input_channels": 1,
                    "input_size": [28, 28],
                    "num_classes": 10,
                    "conv_channels": [12, 24],
                    "lr": 0.001,
                    "width_multipliers": [1.0, 0.75, 0.5],
                    "routing_policy": "early_exit",
                },
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "workflow": {"name": "dynamic_slimmable", "params": {}},
            "trainer": {"epochs": 2},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )

    try:
        builder.build(config)
    except ExperimentAssemblyError:
        return

    raise AssertionError("Expected ExperimentAssemblyError for mismatched routing policy")


def test_dynamic_width_gate_can_route_everything_to_smallest_width() -> None:
    model = TorchRoutedCNNClassifier(
        input_channels=1,
        input_size=(28, 28),
        num_classes=10,
        conv_channels=[12, 24],
        width_multipliers=[1.0, 0.75, 0.5],
        eval_width_multipliers=[0.5, 0.75, 1.0],
        routing_policy="dynamic_width",
        confidence_threshold=0.0,
        gate_metric="confidence",
        device="cpu",
    )
    batch = {"inputs": torch.randn(16, 1, 28, 28), "targets": torch.zeros(16, dtype=torch.long)}
    outputs = model.evaluate(batch)

    assert outputs.shape == (16, 10)
    summary = model.route_summary()
    assert summary["route_counts"]["0.5"] == 16
    assert summary["mean_cost_ratio"] < 1.0


def test_early_exit_gate_can_route_everything_to_early_head() -> None:
    model = TorchRoutedCNNClassifier(
        input_channels=1,
        input_size=(28, 28),
        num_classes=10,
        conv_channels=[12, 24],
        routing_policy="early_exit",
        confidence_threshold=0.0,
        gate_metric="confidence",
        device="cpu",
    )
    batch = {"inputs": torch.randn(16, 1, 28, 28), "targets": torch.zeros(16, dtype=torch.long)}
    outputs = model.evaluate(batch)

    assert outputs.shape == (16, 10)
    summary = model.route_summary()
    assert summary["early_exit_fraction"] == 1.0
    assert summary["mean_cost_ratio"] < 1.0


def test_channel_gating_workflow_runs_with_two_stages() -> None:
    registries, builder = _builder()
    config = ExperimentConfig.from_dict(
        {
            "name": "channel-gating-test",
            "dataset": {
                "name": "synthetic_image_patterns",
                "params": {"train_size": 128, "validation_size": 32, "test_size": 32, "seed": 7},
            },
            "model": {
                "name": "torch_routed_cnn_classifier",
                "params": {
                    "input_channels": 1,
                    "input_size": [28, 28],
                    "num_classes": 10,
                    "conv_channels": [12, 24],
                    "lr": 0.001,
                    "width_multipliers": [1.0, 0.75, 0.5],
                    "eval_width_multipliers": [0.5, 0.75, 1.0],
                    "routing_policy": "dynamic_width",
                    "gate_mode": "learned",
                    "confidence_threshold": 0.7,
                    "min_confidence_threshold": 0.4,
                    "gate_supervision_weight": 0.5,
                },
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "workflow": {"name": "channel_gating", "params": {"warmup_epochs": 1}},
            "trainer": {"epochs": 3, "batch_size": 32, "eval_batch_size": 32},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )
    experiment = builder.build(config)

    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)

    assert result.summary.workflow_metadata["workflow_name"] == "channel_gating"
    assert result.summary.workflow_metadata["stage_count"] == 2
    assert result.summary.stage_history[0]["name"] == "channel_gating_warmup"
    assert result.summary.stage_history[1]["name"] == "channel_gating_routing"


def test_skipnet_workflow_runs_with_two_stages() -> None:
    registries, builder = _builder()
    config = ExperimentConfig.from_dict(
        {
            "name": "skipnet-test",
            "dataset": {
                "name": "synthetic_image_patterns",
                "params": {"train_size": 128, "validation_size": 32, "test_size": 32, "seed": 7},
            },
            "model": {
                "name": "torch_routed_cnn_classifier",
                "params": {
                    "input_channels": 1,
                    "input_size": [28, 28],
                    "num_classes": 10,
                    "conv_channels": [12, 24],
                    "lr": 0.001,
                    "width_multipliers": [1.0, 0.75, 0.5],
                    "routing_policy": "early_exit",
                    "gate_mode": "learned",
                    "confidence_threshold": 0.5,
                    "min_confidence_threshold": 0.3,
                    "gate_supervision_weight": 0.5,
                },
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "workflow": {"name": "skipnet", "params": {"warmup_epochs": 1}},
            "trainer": {"epochs": 3, "batch_size": 32, "eval_batch_size": 32},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )
    experiment = builder.build(config)

    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)

    assert result.summary.workflow_metadata["workflow_name"] == "skipnet"
    assert result.summary.workflow_metadata["stage_count"] == 2
    assert result.summary.stage_history[0]["name"] == "skipnet_warmup"
    assert result.summary.stage_history[1]["name"] == "skipnet_routing"


def test_route_trace_is_recorded_for_dynamic_width_eval() -> None:
    model = TorchRoutedCNNClassifier(
        input_channels=1,
        input_size=(28, 28),
        num_classes=10,
        conv_channels=[12, 24],
        width_multipliers=[1.0, 0.75, 0.5],
        eval_width_multipliers=[0.5, 0.75, 1.0],
        routing_policy="dynamic_width",
        gate_mode="metric",
        confidence_threshold=0.0,
        route_trace_limit=4,
        device="cpu",
    )
    batch = {"inputs": torch.randn(10, 1, 28, 28), "targets": torch.zeros(10, dtype=torch.long)}
    _ = model.evaluate(batch)

    trace = model.route_trace()
    assert trace
    assert trace[-1]["policy"] == "dynamic_width"
    assert len(trace[-1]["trace_samples"]) == 4
    assert "width" in trace[-1]["trace_samples"][0]


def test_instance_wise_sparsity_workflow_runs_with_three_stages() -> None:
    registries, builder = _builder()
    config = ExperimentConfig.from_dict(
        {
            "name": "instance-wise-sparsity-test",
            "dataset": {
                "name": "synthetic_image_patterns",
                "params": {"train_size": 128, "validation_size": 32, "test_size": 32, "seed": 7},
            },
            "model": {
                "name": "torch_routed_cnn_classifier",
                "params": {
                    "input_channels": 1,
                    "input_size": [28, 28],
                    "num_classes": 10,
                    "conv_channels": [12, 24],
                    "lr": 0.001,
                    "width_multipliers": [1.0, 0.75, 0.5],
                    "eval_width_multipliers": [0.5, 0.75, 1.0],
                    "routing_policy": "dynamic_width",
                    "gate_mode": "learned",
                    "confidence_threshold": 0.72,
                    "min_confidence_threshold": 0.35,
                    "gate_supervision_weight": 0.5,
                    "gate_entropy_weight": 0.02,
                    "target_accept_rate": 0.4,
                },
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "workflow": {"name": "instance_wise_sparsity", "params": {"warmup_epochs": 1, "consolidation_epochs": 1}},
            "trainer": {"epochs": 4, "batch_size": 32, "eval_batch_size": 32},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )
    experiment = builder.build(config)

    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)

    assert result.summary.workflow_metadata["workflow_name"] == "instance_wise_sparsity"
    assert result.summary.workflow_metadata["stage_count"] == 3
    assert result.summary.stage_history[0]["name"] == "instance_wise_sparsity_warmup"
    assert result.summary.stage_history[1]["name"] == "instance_wise_sparsity_routing"
    assert result.summary.stage_history[2]["name"] == "instance_wise_sparsity_consolidation"
    assert result.metadata["route_trace"] is not None


def test_iamnn_workflow_runs_with_three_stages() -> None:
    registries, builder = _builder()
    config = ExperimentConfig.from_dict(
        {
            "name": "iamnn-test",
            "dataset": {
                "name": "synthetic_image_patterns",
                "params": {"train_size": 128, "validation_size": 32, "test_size": 32, "seed": 7},
            },
            "model": {
                "name": "torch_routed_cnn_classifier",
                "params": {
                    "input_channels": 1,
                    "input_size": [28, 28],
                    "num_classes": 10,
                    "conv_channels": [12, 24],
                    "lr": 0.001,
                    "routing_policy": "early_exit",
                    "gate_mode": "learned",
                    "confidence_threshold": 0.58,
                    "min_confidence_threshold": 0.3,
                    "gate_supervision_weight": 0.5,
                    "gate_entropy_weight": 0.01,
                    "target_accept_rate": 0.3,
                },
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "workflow": {"name": "iamnn", "params": {"warmup_epochs": 1, "consolidation_epochs": 1}},
            "trainer": {"epochs": 4, "batch_size": 32, "eval_batch_size": 32},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )
    experiment = builder.build(config)

    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)

    assert result.summary.workflow_metadata["workflow_name"] == "iamnn"
    assert result.summary.workflow_metadata["stage_count"] == 3
    assert result.summary.stage_history[0]["name"] == "iamnn_warmup"
    assert result.summary.stage_history[1]["name"] == "iamnn_routing"
    assert result.summary.stage_history[2]["name"] == "iamnn_consolidation"
    assert result.metadata["route_trace"] is not None
