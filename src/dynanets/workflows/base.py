from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from dynanets.adaptation.base import AdaptationMethod
from dynanets.datasets.base import DatasetBundle
from dynanets.metrics.base import Metric
from dynanets.models.base import NeuralModel
from dynanets.runners.train import TrainingRunner, TrainingSummary


@dataclass(slots=True)
class WorkflowStageConfig:
    name: str
    epochs: int
    adaptation_enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class MethodWorkflow(ABC):
    @abstractmethod
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
        raise NotImplementedError


__all__ = ["MethodWorkflow", "WorkflowStageConfig"]
