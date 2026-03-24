from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ArchitectureState:
    """Runtime snapshot of a dynamic model architecture.

    `step` tracks training progress, `version` increments on structural changes,
    and `metadata` stores method-specific architecture details such as width.
    """

    step: int = 0
    version: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class NeuralModel(ABC):
    """Base runtime contract for trainable models used in dynanets experiments."""

    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def training_step(self, batch: Any) -> dict[str, float]:
        """Run one optimization step and return scalar logging values."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, batch: Any) -> Any:
        """Run inference for evaluation without mutating optimizer state."""
        raise NotImplementedError


class DynamicNeuralModel(NeuralModel, ABC):
    """Extension point for models whose structure may change during training."""

    @abstractmethod
    def architecture_state(self) -> ArchitectureState:
        raise NotImplementedError

    @abstractmethod
    def apply_adaptation(self, adaptation: dict[str, Any]) -> None:
        """Apply a structural mutation described by an adaptation payload."""
        raise NotImplementedError