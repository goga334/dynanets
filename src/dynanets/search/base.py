from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dynanets.config import ExperimentConfig


@dataclass(slots=True)
class SearchProposal:
    """Candidate proposal payload emitted by a search space."""

    model_overrides: dict[str, Any] = field(default_factory=dict)
    adaptation_overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateEvaluation:
    """Training result and score for one searched candidate."""

    summary: Any
    score: float
    proposal: SearchProposal
    model_params: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    """Final output of a search method, including best candidate and history."""

    best_evaluation: CandidateEvaluation
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SearchSpace(ABC):
    """Defines how candidate architectures are sampled and mutated."""

    @abstractmethod
    def sample(self, rng: Any) -> SearchProposal:
        raise NotImplementedError

    @abstractmethod
    def mutate(self, proposal: SearchProposal, rng: Any) -> SearchProposal:
        raise NotImplementedError


class SearchMethod(ABC):
    """Algorithm hook that explores candidate proposals from a SearchSpace."""

    @abstractmethod
    def run(
        self,
        config: ExperimentConfig,
        search_space: SearchSpace,
        evaluate_candidate: Callable[[SearchProposal], CandidateEvaluation],
    ) -> SearchResult:
        """Search over candidate proposals and return the best observed evaluation."""
        raise NotImplementedError