from __future__ import annotations

import torch

from dynanets.metrics.base import Metric


class AccuracyMetric(Metric):
    @property
    def name(self) -> str:
        return "accuracy"

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        predicted_labels = predictions.argmax(dim=1)
        return float((predicted_labels == targets).float().mean().item())
