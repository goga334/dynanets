from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from dynanets.models.base import ArchitectureState, DynamicNeuralModel


@dataclass(slots=True)
class AdaptationResult:
    applied: bool
    changes: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None


class AdaptationMethod(ABC):
    @abstractmethod
    def maybe_adapt(
        self,
        model: DynamicNeuralModel,
        state: ArchitectureState,
        context: dict[str, Any],
    ) -> AdaptationResult:
        raise NotImplementedError
