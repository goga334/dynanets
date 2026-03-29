from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from dynanets.config import ExperimentConfig
from dynanets.search.base import CandidateEvaluation, SearchMethod, SearchResult, SearchSpace


@dataclass(slots=True)
class RandomSearch(SearchMethod):
    cycles: int = 8
    seed: int = 42
    metric: str = "accuracy"

    def run(
        self,
        config: ExperimentConfig,
        search_space: SearchSpace,
        evaluate_candidate: Any,
    ) -> SearchResult:
        rng = random.Random(self.seed)
        best_evaluation: CandidateEvaluation | None = None
        history: list[dict[str, Any]] = []

        for cycle in range(self.cycles):
            proposal = search_space.sample(rng)
            evaluation = evaluate_candidate(proposal)
            entry = {
                "id": cycle,
                "origin": "random",
                "score": evaluation.score,
                "proposal": evaluation.proposal,
                "model_params": dict(evaluation.model_params),
                "metadata": dict(evaluation.metadata),
            }
            history.append(entry)
            if best_evaluation is None or evaluation.score > best_evaluation.score:
                best_evaluation = evaluation

        if best_evaluation is None:
            raise RuntimeError("Random search produced no candidates")

        return SearchResult(
            best_evaluation=best_evaluation,
            history=history,
            metadata={
                "search_method": "random_search",
                "metric": self.metric,
                "cycles": self.cycles,
            },
        )