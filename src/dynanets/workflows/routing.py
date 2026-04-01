from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dynanets.adaptation.base import AdaptationMethod
from dynanets.datasets.base import DatasetBundle
from dynanets.metrics.base import Metric
from dynanets.models.base import NeuralModel
from dynanets.models.torch_routed_cnn import TorchRoutedCNNClassifier
from dynanets.models.torch_routed_resnet import TorchRoutedResNetClassifier
from dynanets.runners.train import TrainingRunner, TrainingSummary
from dynanets.workflows.base import MethodWorkflow


def _require_routed_model(model: NeuralModel, *, routing_policy: str, workflow_name: str) -> TorchRoutedCNNClassifier | TorchRoutedResNetClassifier:
    if not isinstance(model, (TorchRoutedCNNClassifier, TorchRoutedResNetClassifier)):
        raise ValueError(f"{workflow_name} currently requires TorchRoutedCNNClassifier")
    if model.routing_policy != routing_policy:
        raise ValueError(f"{workflow_name} requires routing_policy='{routing_policy}'")
    return model


def _gate_snapshot(model: TorchRoutedCNNClassifier | TorchRoutedResNetClassifier) -> dict[str, Any]:
    snapshot = model.gate_config.to_dict()
    snapshot["early_exit_loss_weight"] = float(model.early_exit_loss_weight)
    return snapshot


def _apply_gate_overrides(model: TorchRoutedCNNClassifier | TorchRoutedResNetClassifier, **overrides: Any) -> None:
    for key, value in overrides.items():
        if value is None:
            continue
        if hasattr(model.gate_config, key):
            setattr(model.gate_config, key, value)
            continue
        if hasattr(model, key):
            setattr(model, key, value)
    if "threshold" in overrides and overrides["threshold"] is not None:
        model.confidence_threshold = float(overrides["threshold"])
    if "min_threshold" in overrides:
        model.min_confidence_threshold = None if overrides["min_threshold"] is None else float(overrides["min_threshold"])


def _restore_gate_snapshot(model: TorchRoutedCNNClassifier | TorchRoutedResNetClassifier, snapshot: dict[str, Any]) -> None:
    _apply_gate_overrides(model, **snapshot)


def _extend_stage(
    *,
    summary: TrainingSummary,
    stage_summary: TrainingSummary,
    stage_name: str,
    epoch_start: int,
    metadata: dict[str, Any],
    metric_name: str = "accuracy",
) -> None:
    summary.train_history.extend(stage_summary.train_history)
    summary.metric_history.extend(stage_summary.metric_history)
    summary.adaptation_history.extend(stage_summary.adaptation_history)
    summary.stage_history.append(
        {
            "name": stage_name,
            "epochs": len(stage_summary.train_history),
            "epoch_start": epoch_start,
            "epoch_end": epoch_start + len(stage_summary.train_history) - 1,
            "adaptation_enabled": False,
            "final_train_accuracy": stage_summary.train_history[-1].get("accuracy") if stage_summary.train_history else None,
            "final_val_accuracy": stage_summary.metric_history[-1].get(metric_name) if stage_summary.metric_history else None,
            "adaptations_applied": sum(1 for item in stage_summary.adaptation_history if item.get("applied")),
            "metadata": dict(metadata),
        }
    )


def _run_stage(
    *,
    summary: TrainingSummary,
    model: TorchRoutedCNNClassifier | TorchRoutedResNetClassifier,
    dataset: DatasetBundle,
    metrics: list[Metric],
    training_runner: TrainingRunner,
    trainer_config: dict[str, Any],
    cursor: int,
    stage_name: str,
    epochs: int,
    metadata: dict[str, Any],
) -> int:
    stage_summary = training_runner.run(
        model=model,
        dataset=dataset,
        metrics=metrics,
        epochs=epochs,
        adaptation=None,
        start_epoch=cursor,
        trainer_config=trainer_config,
    )
    _extend_stage(
        summary=summary,
        stage_summary=stage_summary,
        stage_name=stage_name,
        epoch_start=cursor + 1,
        metadata=metadata,
    )
    return cursor + epochs


def _execute_single_stage(
    *,
    workflow_name: str,
    model: TorchRoutedCNNClassifier | TorchRoutedResNetClassifier,
    dataset: DatasetBundle,
    metrics: list[Metric],
    training_runner: TrainingRunner,
    epochs: int,
    trainer_config: dict[str, Any],
    stage_metadata: dict[str, Any],
) -> TrainingSummary:
    summary = TrainingSummary()
    cursor = _run_stage(
        summary=summary,
        model=model,
        dataset=dataset,
        metrics=metrics,
        training_runner=training_runner,
        trainer_config=trainer_config,
        cursor=0,
        stage_name=f"{workflow_name}_train",
        epochs=epochs,
        metadata=stage_metadata,
    )
    summary.workflow_metadata = {
        "workflow_name": workflow_name,
        "configured_total_epochs": epochs,
        "executed_total_epochs": cursor,
        "stage_count": 1,
        "routing_policy": model.routing_policy,
        "gate_mode": model.gate_config.mode,
        "route_summary": model.route_summary(),
        "route_trace": model.route_trace(),
    }
    return summary


def _execute_two_stage_routing(
    *,
    workflow_name: str,
    model: TorchRoutedCNNClassifier | TorchRoutedResNetClassifier,
    dataset: DatasetBundle,
    metrics: list[Metric],
    training_runner: TrainingRunner,
    epochs: int,
    trainer_config: dict[str, Any],
    warmup_epochs: int,
    warmup_overrides: dict[str, Any],
    routing_metadata: dict[str, Any],
) -> TrainingSummary:
    if warmup_epochs <= 0 or warmup_epochs >= epochs:
        raise ValueError(f"{workflow_name} requires 0 < warmup_epochs < trainer.epochs")

    summary = TrainingSummary()
    snapshot = _gate_snapshot(model)
    cursor = 0

    _apply_gate_overrides(model, **warmup_overrides)
    cursor = _run_stage(
        summary=summary,
        model=model,
        dataset=dataset,
        metrics=metrics,
        training_runner=training_runner,
        trainer_config=trainer_config,
        cursor=cursor,
        stage_name=f"{workflow_name}_warmup",
        epochs=warmup_epochs,
        metadata={"phase": "warmup", **warmup_overrides},
    )

    _restore_gate_snapshot(model, snapshot)
    cursor = _run_stage(
        summary=summary,
        model=model,
        dataset=dataset,
        metrics=metrics,
        training_runner=training_runner,
        trainer_config=trainer_config,
        cursor=cursor,
        stage_name=f"{workflow_name}_routing",
        epochs=epochs - warmup_epochs,
        metadata={"phase": "routing", **routing_metadata},
    )

    summary.workflow_metadata = {
        "workflow_name": workflow_name,
        "configured_total_epochs": epochs,
        "executed_total_epochs": cursor,
        "stage_count": len(summary.stage_history),
        "routing_policy": model.routing_policy,
        "gate_mode": model.gate_config.mode,
        "warmup_epochs": warmup_epochs,
        "route_summary": model.route_summary(),
        "route_trace": model.route_trace(),
    }
    return summary


def _execute_three_stage_routing(
    *,
    workflow_name: str,
    model: TorchRoutedCNNClassifier | TorchRoutedResNetClassifier,
    dataset: DatasetBundle,
    metrics: list[Metric],
    training_runner: TrainingRunner,
    epochs: int,
    trainer_config: dict[str, Any],
    warmup_epochs: int,
    consolidation_epochs: int,
    warmup_overrides: dict[str, Any],
    routing_overrides: dict[str, Any],
    consolidation_overrides: dict[str, Any],
) -> TrainingSummary:
    if warmup_epochs <= 0 or consolidation_epochs <= 0 or (warmup_epochs + consolidation_epochs) >= epochs:
        raise ValueError(f"{workflow_name} requires warmup_epochs + consolidation_epochs < trainer.epochs")

    summary = TrainingSummary()
    snapshot = _gate_snapshot(model)
    cursor = 0

    _apply_gate_overrides(model, **warmup_overrides)
    cursor = _run_stage(
        summary=summary,
        model=model,
        dataset=dataset,
        metrics=metrics,
        training_runner=training_runner,
        trainer_config=trainer_config,
        cursor=cursor,
        stage_name=f"{workflow_name}_warmup",
        epochs=warmup_epochs,
        metadata={"phase": "warmup", **warmup_overrides},
    )

    _restore_gate_snapshot(model, snapshot)
    _apply_gate_overrides(model, **routing_overrides)
    routing_epochs = epochs - warmup_epochs - consolidation_epochs
    cursor = _run_stage(
        summary=summary,
        model=model,
        dataset=dataset,
        metrics=metrics,
        training_runner=training_runner,
        trainer_config=trainer_config,
        cursor=cursor,
        stage_name=f"{workflow_name}_routing",
        epochs=routing_epochs,
        metadata={"phase": "routing", **routing_overrides},
    )

    _restore_gate_snapshot(model, snapshot)
    _apply_gate_overrides(model, **consolidation_overrides)
    cursor = _run_stage(
        summary=summary,
        model=model,
        dataset=dataset,
        metrics=metrics,
        training_runner=training_runner,
        trainer_config=trainer_config,
        cursor=cursor,
        stage_name=f"{workflow_name}_consolidation",
        epochs=consolidation_epochs,
        metadata={"phase": "consolidation", **consolidation_overrides},
    )

    _restore_gate_snapshot(model, snapshot)
    summary.workflow_metadata = {
        "workflow_name": workflow_name,
        "configured_total_epochs": epochs,
        "executed_total_epochs": cursor,
        "stage_count": len(summary.stage_history),
        "routing_policy": model.routing_policy,
        "gate_mode": model.gate_config.mode,
        "warmup_epochs": warmup_epochs,
        "consolidation_epochs": consolidation_epochs,
        "route_summary": model.route_summary(),
        "route_trace": model.route_trace(),
    }
    return summary


@dataclass(slots=True)
class DynamicSlimmableWorkflow(MethodWorkflow):
    warmup_epochs: int = 2

    def execute(
        self,
        *,
        model: NeuralModel,
        dataset: DatasetBundle,
        metrics: list[Metric],
        training_runner: TrainingRunner,
        adaptation: AdaptationMethod | None,
        epochs: int,
        trainer_config: dict[str, Any],
    ) -> TrainingSummary:
        if adaptation is not None:
            raise ValueError("DynamicSlimmableWorkflow manages routing internally and does not accept adaptation")
        routed_model = _require_routed_model(model, routing_policy="dynamic_width", workflow_name="DynamicSlimmableWorkflow")
        snapshot = _gate_snapshot(routed_model)
        warmup_threshold = max(0.90, float(snapshot.get("threshold", 0.90)))
        return _execute_two_stage_routing(
            workflow_name="dynamic_slimmable",
            model=routed_model,
            dataset=dataset,
            metrics=metrics,
            training_runner=training_runner,
            epochs=epochs,
            trainer_config=trainer_config,
            warmup_epochs=self.warmup_epochs,
            warmup_overrides={
                "budget_weight": 0.0,
                "accept_rate_weight": 0.0,
                "target_cost_ratio": 1.0,
                "target_accept_rate": None,
                "threshold": warmup_threshold,
                "min_threshold": warmup_threshold,
            },
            routing_metadata={
                "routing_policy": routed_model.routing_policy,
                "width_multipliers": routed_model.width_multipliers,
                "gate_mode": routed_model.gate_config.mode,
                "target_cost_ratio": snapshot.get("target_cost_ratio"),
                "target_accept_rate": snapshot.get("target_accept_rate"),
            },
        )


@dataclass(slots=True)
class ConditionalComputationWorkflow(MethodWorkflow):
    warmup_epochs: int = 2

    def execute(
        self,
        *,
        model: NeuralModel,
        dataset: DatasetBundle,
        metrics: list[Metric],
        training_runner: TrainingRunner,
        adaptation: AdaptationMethod | None,
        epochs: int,
        trainer_config: dict[str, Any],
    ) -> TrainingSummary:
        if adaptation is not None:
            raise ValueError("ConditionalComputationWorkflow manages routing internally and does not accept adaptation")
        routed_model = _require_routed_model(model, routing_policy="early_exit", workflow_name="ConditionalComputationWorkflow")
        snapshot = _gate_snapshot(routed_model)
        warmup_threshold = max(0.94, float(snapshot.get("threshold", 0.94)))
        return _execute_two_stage_routing(
            workflow_name="conditional_computation",
            model=routed_model,
            dataset=dataset,
            metrics=metrics,
            training_runner=training_runner,
            epochs=epochs,
            trainer_config=trainer_config,
            warmup_epochs=self.warmup_epochs,
            warmup_overrides={
                "budget_weight": 0.0,
                "accept_rate_weight": 0.0,
                "target_cost_ratio": 1.0,
                "target_accept_rate": None,
                "threshold": warmup_threshold,
                "min_threshold": warmup_threshold,
                "early_exit_loss_weight": max(1.15, float(routed_model.early_exit_loss_weight)),
            },
            routing_metadata={
                "routing_policy": routed_model.routing_policy,
                "confidence_threshold": routed_model.confidence_threshold,
                "gate_mode": routed_model.gate_config.mode,
                "target_cost_ratio": snapshot.get("target_cost_ratio"),
                "target_accept_rate": snapshot.get("target_accept_rate"),
            },
        )


@dataclass(slots=True)
class ChannelGatingWorkflow(MethodWorkflow):
    warmup_epochs: int = 3

    def execute(
        self,
        *,
        model: NeuralModel,
        dataset: DatasetBundle,
        metrics: list[Metric],
        training_runner: TrainingRunner,
        adaptation: AdaptationMethod | None,
        epochs: int,
        trainer_config: dict[str, Any],
    ) -> TrainingSummary:
        if adaptation is not None:
            raise ValueError("ChannelGatingWorkflow manages routing internally and does not accept adaptation")
        routed_model = _require_routed_model(model, routing_policy="dynamic_width", workflow_name="ChannelGatingWorkflow")
        snapshot = _gate_snapshot(routed_model)
        warmup_threshold = max(0.92, float(snapshot.get("threshold", 0.92)))
        return _execute_two_stage_routing(
            workflow_name="channel_gating",
            model=routed_model,
            dataset=dataset,
            metrics=metrics,
            training_runner=training_runner,
            epochs=epochs,
            trainer_config=trainer_config,
            warmup_epochs=self.warmup_epochs,
            warmup_overrides={
                "budget_weight": 0.0,
                "target_cost_ratio": 1.0,
                "threshold": warmup_threshold,
                "min_threshold": warmup_threshold,
            },
            routing_metadata={
                "routing_policy": routed_model.routing_policy,
                "gate_mode": routed_model.gate_config.mode,
                "target_cost_ratio": snapshot.get("target_cost_ratio"),
            },
        )


@dataclass(slots=True)
class SkipNetWorkflow(MethodWorkflow):
    warmup_epochs: int = 3

    def execute(
        self,
        *,
        model: NeuralModel,
        dataset: DatasetBundle,
        metrics: list[Metric],
        training_runner: TrainingRunner,
        adaptation: AdaptationMethod | None,
        epochs: int,
        trainer_config: dict[str, Any],
    ) -> TrainingSummary:
        if adaptation is not None:
            raise ValueError("SkipNetWorkflow manages routing internally and does not accept adaptation")
        routed_model = _require_routed_model(model, routing_policy="early_exit", workflow_name="SkipNetWorkflow")
        snapshot = _gate_snapshot(routed_model)
        warmup_threshold = max(0.95, float(snapshot.get("threshold", 0.95)))
        return _execute_two_stage_routing(
            workflow_name="skipnet",
            model=routed_model,
            dataset=dataset,
            metrics=metrics,
            training_runner=training_runner,
            epochs=epochs,
            trainer_config=trainer_config,
            warmup_epochs=self.warmup_epochs,
            warmup_overrides={
                "budget_weight": 0.0,
                "target_cost_ratio": 1.0,
                "threshold": warmup_threshold,
                "min_threshold": warmup_threshold,
                "early_exit_loss_weight": max(1.2, float(routed_model.early_exit_loss_weight)),
            },
            routing_metadata={
                "routing_policy": routed_model.routing_policy,
                "gate_mode": routed_model.gate_config.mode,
                "target_cost_ratio": snapshot.get("target_cost_ratio"),
            },
        )


@dataclass(slots=True)
class InstanceWiseSparsityWorkflow(MethodWorkflow):
    warmup_epochs: int = 2
    consolidation_epochs: int = 2

    def execute(
        self,
        *,
        model: NeuralModel,
        dataset: DatasetBundle,
        metrics: list[Metric],
        training_runner: TrainingRunner,
        adaptation: AdaptationMethod | None,
        epochs: int,
        trainer_config: dict[str, Any],
    ) -> TrainingSummary:
        if adaptation is not None:
            raise ValueError("InstanceWiseSparsityWorkflow manages routing internally and does not accept adaptation")
        routed_model = _require_routed_model(model, routing_policy="dynamic_width", workflow_name="InstanceWiseSparsityWorkflow")
        snapshot = _gate_snapshot(routed_model)
        warmup_threshold = max(0.92, float(snapshot.get("threshold", 0.92)))
        routing_target = snapshot.get("target_accept_rate")
        if routing_target is None:
            routing_target = 0.45
        return _execute_three_stage_routing(
            workflow_name="instance_wise_sparsity",
            model=routed_model,
            dataset=dataset,
            metrics=metrics,
            training_runner=training_runner,
            epochs=epochs,
            trainer_config=trainer_config,
            warmup_epochs=self.warmup_epochs,
            consolidation_epochs=self.consolidation_epochs,
            warmup_overrides={
                "budget_weight": 0.0,
                "target_cost_ratio": 1.0,
                "target_accept_rate": None,
                "threshold": warmup_threshold,
                "min_threshold": warmup_threshold,
            },
            routing_overrides={
                "target_cost_ratio": min(float(snapshot.get("target_cost_ratio", 0.75)), 0.72),
                "target_accept_rate": routing_target,
            },
            consolidation_overrides={
                "target_cost_ratio": min(float(snapshot.get("target_cost_ratio", 0.75)), 0.68),
                "target_accept_rate": min(float(routing_target), 0.4),
                "threshold": snapshot.get("min_threshold") or snapshot.get("threshold"),
                "min_threshold": snapshot.get("min_threshold") or snapshot.get("threshold"),
            },
        )


@dataclass(slots=True)
class IamNNWorkflow(MethodWorkflow):
    warmup_epochs: int = 2
    consolidation_epochs: int = 2

    def execute(
        self,
        *,
        model: NeuralModel,
        dataset: DatasetBundle,
        metrics: list[Metric],
        training_runner: TrainingRunner,
        adaptation: AdaptationMethod | None,
        epochs: int,
        trainer_config: dict[str, Any],
    ) -> TrainingSummary:
        if adaptation is not None:
            raise ValueError("IamNNWorkflow manages routing internally and does not accept adaptation")
        routed_model = _require_routed_model(model, routing_policy="early_exit", workflow_name="IamNNWorkflow")
        snapshot = _gate_snapshot(routed_model)
        warmup_threshold = max(0.96, float(snapshot.get("threshold", 0.96)))
        target_accept_rate = snapshot.get("target_accept_rate")
        if target_accept_rate is None:
            target_accept_rate = 0.35
        return _execute_three_stage_routing(
            workflow_name="iamnn",
            model=routed_model,
            dataset=dataset,
            metrics=metrics,
            training_runner=training_runner,
            epochs=epochs,
            trainer_config=trainer_config,
            warmup_epochs=self.warmup_epochs,
            consolidation_epochs=self.consolidation_epochs,
            warmup_overrides={
                "budget_weight": 0.0,
                "target_cost_ratio": 1.0,
                "target_accept_rate": None,
                "threshold": warmup_threshold,
                "min_threshold": warmup_threshold,
                "early_exit_loss_weight": max(1.15, float(routed_model.early_exit_loss_weight)),
            },
            routing_overrides={
                "target_cost_ratio": min(float(snapshot.get("target_cost_ratio", 0.8)), 0.78),
                "target_accept_rate": target_accept_rate,
                "early_exit_loss_weight": max(1.05, float(routed_model.early_exit_loss_weight)),
            },
            consolidation_overrides={
                "target_cost_ratio": min(float(snapshot.get("target_cost_ratio", 0.8)), 0.72),
                "target_accept_rate": min(float(target_accept_rate), 0.3),
                "threshold": snapshot.get("min_threshold") or snapshot.get("threshold"),
                "min_threshold": snapshot.get("min_threshold") or snapshot.get("threshold"),
                "early_exit_loss_weight": float(routed_model.early_exit_loss_weight),
            },
        )
