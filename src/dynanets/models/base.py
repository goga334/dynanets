from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ArchitectureState:
    step: int = 0
    version: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class NeuralModel(ABC):
    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def training_step(self, batch: Any) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, batch: Any) -> Any:
        raise NotImplementedError


class DynamicNeuralModel(NeuralModel, ABC):
    @abstractmethod
    def architecture_state(self) -> ArchitectureState:
        raise NotImplementedError

    @abstractmethod
    def apply_adaptation(self, adaptation: dict[str, Any]) -> None:
        raise NotImplementedError
