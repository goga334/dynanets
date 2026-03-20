from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dynanets.config import ExperimentConfig


@dataclass(slots=True)
class SearchProposal:
    model_overrides: dict[str, Any] = field(default_factory=dict)
    adaptation_overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateEvaluation:
    summary: Any
    score: float
    model_params: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    best_evaluation: CandidateEvaluation
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SearchMethod(ABC):
    @abstractmethod
    def run(
        self,
        config: ExperimentConfig,
        evaluate_candidate: Callable[[dict[str, Any]], CandidateEvaluation],
    ) -> SearchResult:
        raise NotImplementedError