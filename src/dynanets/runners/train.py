from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

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
        trainer_config: dict[str, Any] | None = None,
    ) -> TrainingSummary:
        summary = TrainingSummary()
        trainer_config = dict(trainer_config or {})
        batch_size = _optional_positive_int(trainer_config.get("batch_size"))
        eval_batch_size = _optional_positive_int(trainer_config.get("eval_batch_size"))
        shuffle = bool(trainer_config.get("shuffle", True))

        for offset in range(epochs):
            epoch = start_epoch + offset
            train_result = self._run_epoch(model, dataset.train, epoch, total_epochs=start_epoch + epochs, batch_size=batch_size, shuffle=shuffle)
            summary.train_history.append(train_result)

            metric_result: dict[str, float] = {}
            if dataset.validation is not None:
                metric_result = self._evaluate(model, dataset.validation, metrics, batch_size=eval_batch_size)
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

    def _run_epoch(
        self,
        model: NeuralModel,
        split: Any,
        epoch: int,
        *,
        total_epochs: int,
        batch_size: int | None,
        shuffle: bool,
    ) -> dict[str, float]:
        if batch_size is None or not _supports_tensor_batches(split):
            batch = {"inputs": split.inputs, "targets": split.targets, "epoch": epoch, "total_epochs": total_epochs}
            return model.training_step(batch)

        totals: dict[str, float] = {}
        total_examples = 0
        for batch_inputs, batch_targets in self._iter_tensor_batches(split.inputs, split.targets, batch_size, shuffle=shuffle):
            batch = {"inputs": batch_inputs, "targets": batch_targets, "epoch": epoch, "total_epochs": total_epochs}
            result = model.training_step(batch)
            batch_examples = int(batch_targets.shape[0])
            total_examples += batch_examples
            for key, value in result.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch_examples

        if total_examples == 0:
            return {}
        return {key: value / total_examples for key, value in totals.items()}

    def _evaluate(
        self,
        model: NeuralModel,
        split: Any,
        metrics: list[Metric],
        *,
        batch_size: int | None,
    ) -> dict[str, float]:
        if batch_size is None or not _supports_tensor_batches(split):
            predictions = model.evaluate({"inputs": split.inputs, "targets": split.targets})
            return {metric.name: metric.compute(predictions, split.targets) for metric in metrics}

        prediction_batches: list[torch.Tensor] = []
        target_batches: list[torch.Tensor] = []
        for batch_inputs, batch_targets in self._iter_tensor_batches(split.inputs, split.targets, batch_size, shuffle=False):
            predictions = model.evaluate({"inputs": batch_inputs, "targets": batch_targets})
            prediction_batches.append(predictions.detach().cpu())
            target_batches.append(batch_targets.detach().cpu())

        if not prediction_batches:
            return {}

        predictions = torch.cat(prediction_batches, dim=0)
        targets = torch.cat(target_batches, dim=0)
        return {metric.name: metric.compute(predictions, targets) for metric in metrics}

    def _iter_tensor_batches(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int,
        *,
        shuffle: bool,
    ):
        indices = torch.randperm(inputs.shape[0]) if shuffle else torch.arange(inputs.shape[0])
        for start in range(0, int(indices.shape[0]), batch_size):
            batch_indices = indices[start : start + batch_size]
            yield inputs[batch_indices], targets[batch_indices]

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
        before_conv_channels = list(before_meta.get("conv_channels", []))
        after_conv_channels = list(after_meta.get("conv_channels", []))
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
        before_flops = before_meta.get("forward_flop_proxy")
        after_flops = after_meta.get("forward_flop_proxy")
        before_activations = before_meta.get("activation_elements")
        after_activations = after_meta.get("activation_elements")

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
            "forward_flop_proxy_delta": _numeric_delta(before_flops, after_flops),
            "activation_elements_delta": _numeric_delta(before_activations, after_activations),
            "hidden_dims_changed": before_hidden_dims != after_hidden_dims,
            "conv_channels_before": before_conv_channels,
            "conv_channels_after": after_conv_channels,
            "channels_changed": before_conv_channels != after_conv_channels,
        }


def _supports_tensor_batches(split: Any) -> bool:
    return isinstance(getattr(split, "inputs", None), torch.Tensor) and isinstance(getattr(split, "targets", None), torch.Tensor)


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _numeric_delta(before: Any, after: Any) -> int | None:
    if before is None or after is None:
        return None
    return int(after) - int(before)


def _float_delta(before: Any, after: Any) -> float | None:
    if before is None or after is None:
        return None
    return float(after) - float(before)

