from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any

from dynanets.config import ExperimentConfig
from dynanets.search.base import CandidateEvaluation, SearchMethod, SearchResult


@dataclass(slots=True)
class RegularizedEvolutionSearch(SearchMethod):
    hidden_dim_choices: list[int]
    activation_choices: list[str]
    lr_choices: list[float]
    cycles: int = 8
    population_size: int = 4
    sample_size: int = 2
    seed: int = 42
    metric: str = "accuracy"

    def run(
        self,
        config: ExperimentConfig,
        evaluate_candidate: Any,
    ) -> SearchResult:
        rng = random.Random(self.seed)
        population: deque[dict[str, Any]] = deque()
        history: list[dict[str, Any]] = []
        best_evaluation: CandidateEvaluation | None = None

        for cycle in range(self.cycles):
            if len(population) < self.population_size:
                candidate = self._random_candidate(rng)
                origin = "random"
            else:
                sample = rng.sample(list(population), k=min(self.sample_size, len(population)))
                parent = max(sample, key=lambda item: item["score"])
                candidate = self._mutate(parent["model_params"], rng)
                origin = f"mutated-from-{parent['id']}"

            evaluation = evaluate_candidate(candidate)
            entry = {
                "id": cycle,
                "origin": origin,
                "score": evaluation.score,
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

    def _random_candidate(self, rng: random.Random) -> dict[str, Any]:
        return {
            "hidden_dim": rng.choice(self.hidden_dim_choices),
            "activation": rng.choice(self.activation_choices),
            "lr": rng.choice(self.lr_choices),
        }

    def _mutate(self, parent: dict[str, Any], rng: random.Random) -> dict[str, Any]:
        child = dict(parent)
        field = rng.choice(["hidden_dim", "activation", "lr"])
        choices = {
            "hidden_dim": self.hidden_dim_choices,
            "activation": self.activation_choices,
            "lr": self.lr_choices,
        }[field]
        alternatives = [value for value in choices if value != child[field]]
        if alternatives:
            child[field] = rng.choice(alternatives)
        return child