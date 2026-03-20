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


class TrainingRunner:
    """Framework-agnostic orchestration scaffold for training experiments."""

    def run(
        self,
        model: NeuralModel,
        dataset: DatasetBundle,
        metrics: list[Metric],
        epochs: int = 1,
        adaptation: AdaptationMethod | None = None,
    ) -> TrainingSummary:
        summary = TrainingSummary()

        for epoch in range(epochs):
            train_result = self._run_epoch(model, dataset.train, epoch)
            summary.train_history.append(train_result)

            metric_result: dict[str, float] = {}
            if dataset.validation is not None:
                metric_result = self._evaluate(model, dataset.validation, metrics)
                summary.metric_history.append(metric_result)

            if adaptation is not None and isinstance(model, DynamicNeuralModel):
                state = model.architecture_state()
                result = adaptation.maybe_adapt(
                    model=model,
                    state=state,
                    context={
                        "epoch": epoch,
                        "train_result": train_result,
                        "validation_metrics": metric_result,
                    },
                )
                summary.adaptation_history.append(
                    {"epoch": epoch, "applied": result.applied, "changes": result.changes}
                )

        return summary

    def _run_epoch(self, model: NeuralModel, split: Any, epoch: int) -> dict[str, float]:
        batch = {"inputs": split.inputs, "targets": split.targets, "epoch": epoch}
        return model.training_step(batch)

    def _evaluate(self, model: NeuralModel, split: Any, metrics: list[Metric]) -> dict[str, float]:
        predictions = model.evaluate({"inputs": split.inputs, "targets": split.targets})
        return {metric.name: metric.compute(predictions, split.targets) for metric in metrics}
