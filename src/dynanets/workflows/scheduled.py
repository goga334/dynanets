from __future__ import annotations

from dataclasses import dataclass, field

from dynanets.adaptation.base import AdaptationMethod
from dynanets.datasets.base import DatasetBundle
from dynanets.metrics.base import Metric
from dynanets.models.base import NeuralModel
from dynanets.runners.train import TrainingRunner, TrainingSummary
from dynanets.workflows.base import MethodWorkflow, WorkflowStageConfig


@dataclass(slots=True)
class SingleStageWorkflow(MethodWorkflow):
    def execute(
        self,
        *,
        model: NeuralModel,
        dataset: DatasetBundle,
        metrics: list[Metric],
        training_runner: TrainingRunner,
        adaptation: AdaptationMethod | None,
        epochs: int,
    ) -> TrainingSummary:
        summary = training_runner.run(
            model=model,
            dataset=dataset,
            metrics=metrics,
            epochs=epochs,
            adaptation=adaptation,
            start_epoch=0,
        )
        summary.stage_history.append(
            {
                "name": "train",
                "epochs": epochs,
                "epoch_start": 1,
                "epoch_end": epochs,
                "adaptation_enabled": adaptation is not None,
                "final_train_accuracy": summary.train_history[-1].get("accuracy") if summary.train_history else None,
                "final_val_accuracy": summary.metric_history[-1].get("accuracy") if summary.metric_history else None,
                "adaptations_applied": sum(1 for item in summary.adaptation_history if item.get("applied")),
                "metadata": {},
            }
        )
        summary.workflow_metadata = {
            "configured_total_epochs": epochs,
            "executed_total_epochs": epochs,
            "stage_count": 1,
        }
        return summary


@dataclass(slots=True)
class ScheduledWorkflow(MethodWorkflow):
    stages: list[WorkflowStageConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("ScheduledWorkflow requires at least one stage")
        for stage in self.stages:
            if stage.epochs <= 0:
                raise ValueError("ScheduledWorkflow stages must have positive epochs")

    def execute(
        self,
        *,
        model: NeuralModel,
        dataset: DatasetBundle,
        metrics: list[Metric],
        training_runner: TrainingRunner,
        adaptation: AdaptationMethod | None,
        epochs: int,
    ) -> TrainingSummary:
        configured_total = sum(stage.epochs for stage in self.stages)
        if epochs != configured_total:
            raise ValueError(
                f"ScheduledWorkflow stages sum to {configured_total} epochs, but trainer.epochs is {epochs}"
            )

        summary = TrainingSummary()
        cursor = 0
        for stage in self.stages:
            stage_adaptation = adaptation if stage.adaptation_enabled else None
            stage_summary = training_runner.run(
                model=model,
                dataset=dataset,
                metrics=metrics,
                epochs=stage.epochs,
                adaptation=stage_adaptation,
                start_epoch=cursor,
            )
            summary.train_history.extend(stage_summary.train_history)
            summary.metric_history.extend(stage_summary.metric_history)
            summary.adaptation_history.extend(stage_summary.adaptation_history)
            summary.stage_history.append(
                {
                    "name": stage.name,
                    "epochs": stage.epochs,
                    "epoch_start": cursor + 1,
                    "epoch_end": cursor + stage.epochs,
                    "adaptation_enabled": stage.adaptation_enabled and adaptation is not None,
                    "final_train_accuracy": stage_summary.train_history[-1].get("accuracy") if stage_summary.train_history else None,
                    "final_val_accuracy": stage_summary.metric_history[-1].get("accuracy") if stage_summary.metric_history else None,
                    "adaptations_applied": sum(1 for item in stage_summary.adaptation_history if item.get("applied")),
                    "metadata": dict(stage.metadata),
                }
            )
            cursor += stage.epochs

        summary.workflow_metadata = {
            "configured_total_epochs": epochs,
            "executed_total_epochs": cursor,
            "stage_count": len(self.stages),
        }
        return summary


__all__ = ["ScheduledWorkflow", "SingleStageWorkflow"]
