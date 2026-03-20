from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dynanets.config import ExperimentConfig
from dynanets.experiment import Experiment
from dynanets.models.base import DynamicNeuralModel
from dynanets.runners.train import TrainingRunner, TrainingSummary
from dynanets.search.base import CandidateEvaluation, SearchResult


@dataclass(slots=True)
class SearchRunSummary:
    best_summary: TrainingSummary
    best_model_params: dict[str, Any]
    best_score: float
    evaluation_history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    final_hidden_dim: int | None = None


class SearchRunner:
    def run(
        self,
        config: ExperimentConfig,
        experiment: Experiment,
        registries: dict[str, Any],
    ) -> SearchRunSummary:
        if experiment.search is None:
            raise ValueError("SearchRunner requires an experiment with a search method")

        dataset = experiment.dataset.build()
        trainer = TrainingRunner()

        def evaluate_candidate(model_overrides: dict[str, Any]) -> CandidateEvaluation:
            model_params = dict(config.model.params)
            model_params.update(model_overrides)
            model = registries["models"].build(config.model.name, **model_params)
            summary = trainer.run(
                model=model,
                dataset=dataset,
                metrics=experiment.metrics,
                epochs=int(config.trainer.get("epochs", 1)),
                adaptation=experiment.adaptation,
            )
            score = self._score_summary(summary, metric_name=config.search.params.get("metric", "accuracy"))
            metadata = {}
            if isinstance(model, DynamicNeuralModel):
                metadata["final_hidden_dim"] = int(model.architecture_state().metadata.get("hidden_dim", 0))
            return CandidateEvaluation(
                summary=summary,
                score=score,
                model_params=model_params,
                metadata=metadata,
            )

        result: SearchResult = experiment.search.run(config, evaluate_candidate)
        final_hidden_dim = result.best_evaluation.metadata.get("final_hidden_dim")
        return SearchRunSummary(
            best_summary=result.best_evaluation.summary,
            best_model_params=result.best_evaluation.model_params,
            best_score=result.best_evaluation.score,
            evaluation_history=result.history,
            metadata=result.metadata,
            final_hidden_dim=final_hidden_dim,
        )

    def _score_summary(self, summary: TrainingSummary, metric_name: str) -> float:
        if not summary.metric_history:
            raise ValueError("Search evaluation requires validation metrics")
        values = [epoch_metrics[metric_name] for epoch_metrics in summary.metric_history if metric_name in epoch_metrics]
        if not values:
            raise ValueError(f"Metric '{metric_name}' was not found in the validation history")
        return max(values)