from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dynanets.architecture import extract_architecture_artifacts
from dynanets.config import ExperimentConfig
from dynanets.experiment import Experiment
from dynanets.models.base import DynamicNeuralModel
from dynanets.runners.train import TrainingRunner, TrainingSummary
from dynanets.runtime import prepare_factory_kwargs
from dynanets.search import MLPSearchSpace
from dynanets.search.base import CandidateEvaluation, SearchProposal, SearchResult


@dataclass(slots=True)
class SearchRunSummary:
    best_summary: TrainingSummary
    best_model_params: dict[str, Any]
    best_score: float
    evaluation_history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    final_hidden_dim: int | None = None
    architecture_spec: dict[str, Any] | None = None
    architecture_graph: dict[str, Any] | None = None


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
        search_space = self._build_search_space(config)

        def evaluate_candidate(proposal: SearchProposal) -> CandidateEvaluation:
            model_factory = registries["models"].get(config.model.name)
            model_params = dict(config.model.params)
            model_params.update(proposal.model_overrides)
            model_params = prepare_factory_kwargs(model_factory, model_params, runtime=config.runtime)
            model = model_factory(**model_params)
            summary = experiment.workflow.execute(
                model=model,
                dataset=dataset,
                metrics=experiment.metrics,
                training_runner=trainer,
                adaptation=experiment.adaptation,
                epochs=int(config.trainer.get("epochs", 1)),
            )
            score = self._score_summary(summary, metric_name=config.search.params.get("metric", "accuracy"))
            metadata = dict(proposal.metadata)
            spec_dict, graph_dict = extract_architecture_artifacts(model, name=config.name)
            if spec_dict is not None:
                metadata["architecture_spec"] = spec_dict
            if graph_dict is not None:
                metadata["architecture_graph"] = graph_dict
            if hasattr(model, "device"):
                metadata["device"] = str(model.device)
            if isinstance(model, DynamicNeuralModel):
                metadata["final_hidden_dim"] = int(model.architecture_state().metadata.get("hidden_dim", 0))
            return CandidateEvaluation(
                summary=summary,
                score=score,
                proposal=proposal,
                model_params=model_params,
                metadata=metadata,
            )

        result: SearchResult = experiment.search.run(config, search_space, evaluate_candidate)
        final_hidden_dim = result.best_evaluation.metadata.get("final_hidden_dim")
        return SearchRunSummary(
            best_summary=result.best_evaluation.summary,
            best_model_params=result.best_evaluation.model_params,
            best_score=result.best_evaluation.score,
            evaluation_history=result.history,
            metadata={**result.metadata, "device": result.best_evaluation.metadata.get("device")},
            final_hidden_dim=final_hidden_dim,
            architecture_spec=result.best_evaluation.metadata.get("architecture_spec"),
            architecture_graph=result.best_evaluation.metadata.get("architecture_graph"),
        )

    def _build_search_space(self, config: ExperimentConfig) -> MLPSearchSpace:
        params = config.search.params
        model_params = config.model.params
        return MLPSearchSpace(
            input_dim=int(model_params["input_dim"]),
            output_dim=int(model_params["output_dim"]),
            hidden_dim_choices=[int(value) for value in params["hidden_dim_choices"]],
            activation_choices=[str(value) for value in params["activation_choices"]],
            lr_choices=[float(value) for value in params["lr_choices"]],
        )

    def _score_summary(self, summary: TrainingSummary, metric_name: str) -> float:
        if not summary.metric_history:
            raise ValueError("Search evaluation requires validation metrics")
        values = [epoch_metrics[metric_name] for epoch_metrics in summary.metric_history if metric_name in epoch_metrics]
        if not values:
            raise ValueError(f"Metric '{metric_name}' was not found in the validation history")
        return max(values)
