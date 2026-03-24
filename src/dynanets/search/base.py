from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dynanets.config import ExperimentConfig


@dataclass(slots=True)
class SearchProposal:
    """Candidate override payload for search-space sampling or mutation."""

    model_overrides: dict[str, Any] = field(default_factory=dict)
    adaptation_overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateEvaluation:
    """Training result and score for one searched candidate."""

    summary: Any
    score: float
    model_params: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    """Final output of a search method, including best candidate and history."""

    best_evaluation: CandidateEvaluation
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SearchMethod(ABC):
    """Algorithm hook that explores candidate model parameterizations."""

    @abstractmethod
    def run(
        self,
        config: ExperimentConfig,
        evaluate_candidate: Callable[[dict[str, Any]], CandidateEvaluation],
    ) -> SearchResult:
        """Search over candidates and return the best observed evaluation."""
        raise NotImplementedError