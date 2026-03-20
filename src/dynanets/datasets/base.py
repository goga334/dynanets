from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class DataSplit:
    inputs: Any
    targets: Any


@dataclass(slots=True)
class DatasetBundle:
    train: DataSplit
    validation: DataSplit | None = None
    test: DataSplit | None = None
    metadata: dict[str, Any] | None = None


class DatasetFactory(ABC):
    @abstractmethod
    def build(self) -> DatasetBundle:
        raise NotImplementedError
