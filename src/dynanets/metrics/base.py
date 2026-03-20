from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Metric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def compute(self, predictions: Any, targets: Any) -> float:
        raise NotImplementedError
