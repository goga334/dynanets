from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any

from dynanets.config import ExperimentConfig
from dynanets.search.base import CandidateEvaluation, SearchMethod, SearchResult, SearchSpace


@dataclass(slots=True)
class RegularizedEvolutionSearch(SearchMethod):
    cycles: int = 8
    population_size: int = 4
    sample_size: int = 2
    seed: int = 42
    metric: str = "accuracy"

    def run(
        self,
        config: ExperimentConfig,
        search_space: SearchSpace,
        evaluate_candidate: Any,
    ) -> SearchResult:
        rng = random.Random(self.seed)
        population: deque[dict[str, Any]] = deque()
        history: list[dict[str, Any]] = []
        best_evaluation: CandidateEvaluation | None = None

        for cycle in range(self.cycles):
            if len(population) < self.population_size:
                proposal = search_space.sample(rng)
                origin = "random"
            else:
                sample = rng.sample(list(population), k=min(self.sample_size, len(population)))
                parent = max(sample, key=lambda item: item["score"])
                proposal = search_space.mutate(parent["proposal"], rng)
                origin = f"mutated-from-{parent['id']}"

            evaluation = evaluate_candidate(proposal)
            entry = {
                "id": cycle,
                "origin": origin,
                "score": evaluation.score,
                "proposal": evaluation.proposal,
                "model_params": dict(evaluation.model_params),
                "metadata": dict(evaluation.metadata),
            }
            history.append(entry)
            population.append({**entry, "evaluation": evaluation})
            if len(population) > self.population_size:
                population.popleft()

            if best_evaluation is None or evaluation.score > best_evaluation.score:
                best_evaluation = evaluation

        if best_evaluation is None:
            raise RuntimeError("Regularized evolution produced no candidates")

        return SearchResult(
            best_evaluation=best_evaluation,
            history=history,
            metadata={
                "search_method": "regularized_evolution",
                "metric": self.metric,
                "cycles": self.cycles,
                "population_size": self.population_size,
                "sample_size": self.sample_size,
            },
        )