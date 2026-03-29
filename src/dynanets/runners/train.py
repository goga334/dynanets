from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dynanets.adaptation.base import AdaptationMethod
from dynanets.datasets.base import DatasetBundle
from dynanets.metrics.base import Metric
from dynanets.models.base import DynamicNeuralModel, NeuralModel


@dataclass(slots=True)
class TrainingSummary:
    train_history: list[dict[str, float]] = field(default_factory=list)
    metric_history: list[dict[str, float]] = field(default_factory=list)
    adaptation_history: list[dict[str, Any]] = field(default_factory=list)
    stage_history: list[dict[str, Any]] = field(default_factory=list)
    workflow_metadata: dict[str, Any] = field(default_factory=dict)


class TrainingRunner:
    """Framework-agnostic orchestration scaffold for training experiments."""

    def run(
        self,
        model: NeuralModel,
        dataset: DatasetBundle,
        metrics: list[Metric],
        epochs: int = 1,
        adaptation: AdaptationMethod | None = None,
        start_epoch: int = 0,
    ) -> TrainingSummary:
        summary = TrainingSummary()

        for offset in range(epochs):
            epoch = start_epoch + offset
            train_result = self._run_epoch(model, dataset.train, epoch)
            summary.train_history.append(train_result)

            metric_result: dict[str, float] = {}
            if dataset.validation is not None:
                metric_result = self._evaluate(model, dataset.validation, metrics)
                summary.metric_history.append(metric_result)

            if adaptation is not None and isinstance(model, DynamicNeuralModel):
                before_state = model.architecture_state().to_dict()
                model_capabilities = model.capabilities()
                result = adaptation.maybe_adapt(
                    model=model,
                    state=model.architecture_state(),
                    context={
                        "epoch": epoch,
                        "train_result": train_result,
                        "validation_metrics": metric_result,
                    },
                )
                after_state = model.architecture_state().to_dict()
                effect_summary = self._build_effect_summary(before_state, after_state, result.applied)
                if result.event is not None:
                    result.event.before_state = before_state
                    result.event.after_state = after_state
                    result.event.model_capabilities = dict(model_capabilities)
                    result.event.effect_summary = effect_summary
                    summary.adaptation_history.append(result.event.to_dict())
                else:
                    summary.adaptation_history.append(
                        {
                            "epoch": epoch,
                            "event_type": None,
                            "params": {},
                            "metadata": dict(result.metadata),
                            "before_state": before_state,
                            "after_state": after_state,
                            "model_capabilities": dict(model_capabilities),
                            "effect_summary": effect_summary,
                            "applied": result.applied,
                            "reason": result.reason,
                        }
                    )

        return summary

    def _run_epoch(self, model: NeuralModel, split: Any, epoch: int) -> dict[str, float]:
        batch = {"inputs": split.inputs, "targets": split.targets, "epoch": epoch}
        return model.training_step(batch)

    def _evaluate(self, model: NeuralModel, split: Any, metrics: list[Metric]) -> dict[str, float]:
        predictions = model.evaluate({"inputs": split.inputs, "targets": split.targets})
        return {metric.name: metric.compute(predictions, split.targets) for metric in metrics}

    def _build_effect_summary(
        self,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
        applied: bool,
    ) -> dict[str, Any]:
        before_meta = before_state.get("metadata", {})
        after_meta = after_state.get("metadata", {})
        before_hidden_dims = list(before_meta.get("hidden_dims", []))
        after_hidden_dims = list(after_meta.get("hidden_dims", []))
        before_hidden_dim = before_meta.get("hidden_dim")
        after_hidden_dim = after_meta.get("hidden_dim")
        before_layers = before_meta.get("num_hidden_layers", len(before_hidden_dims))
        after_layers = after_meta.get("num_hidden_layers", len(after_hidden_dims))
        before_params = before_meta.get("parameter_count")
        after_params = after_meta.get("parameter_count")
        before_nonzero = before_meta.get("nonzero_parameter_count")
        after_nonzero = after_meta.get("nonzero_parameter_count")
        before_sparsity = before_meta.get("weight_sparsity")
        after_sparsity = after_meta.get("weight_sparsity")

        return {
            "applied": applied,
            "structural_change": before_state != after_state,
            "version_delta": int(after_state.get("version", 0)) - int(before_state.get("version", 0)),
            "step_delta": int(after_state.get("step", 0)) - int(before_state.get("step", 0)),
            "hidden_dim_delta": _numeric_delta(before_hidden_dim, after_hidden_dim),
            "num_hidden_layers_delta": _numeric_delta(before_layers, after_layers),
            "parameter_count_delta": _numeric_delta(before_params, after_params),
            "nonzero_parameter_count_delta": _numeric_delta(before_nonzero, after_nonzero),
            "weight_sparsity_delta": _float_delta(before_sparsity, after_sparsity),
            "hidden_dims_changed": before_hidden_dims != after_hidden_dims,
        }


def _numeric_delta(before: Any, after: Any) -> int | None:
    if before is None or after is None:
        return None
    return int(after) - int(before)



def _float_delta(before: Any, after: Any) -> float | None:
    if before is None or after is None:
        return None
    return float(after) - float(before)
