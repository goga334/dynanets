from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dynanets.architecture import extract_architecture_artifacts
from dynanets.config import ExperimentConfig
from dynanets.constraints import ConstraintEvaluator
from dynanets.experiment import Experiment
from dynanets.models.base import DynamicNeuralModel
from dynanets.reporting import summarize_run
from dynanets.runners.search import SearchRunner
from dynanets.runners.train import TrainingRunner, TrainingSummary
from dynanets.runtime import format_runtime_environment, runtime_environment


@dataclass(slots=True)
class ExecutionResult:
    name: str
    mode: str
    summary: TrainingSummary
    final_hidden_dim: int | None = None
    architecture_spec: dict[str, Any] | None = None
    architecture_graph: dict[str, Any] | None = None
    constraint_summary: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    search_history: list[dict[str, Any]] = field(default_factory=list)
    best_model_params: dict[str, Any] = field(default_factory=dict)
    best_score: float | None = None

    def to_report_item(self) -> dict[str, Any]:
        return summarize_run(
            self.name,
            self.summary,
            final_hidden_dim=self.final_hidden_dim,
            metadata=self.metadata,
            architecture_spec=self.architecture_spec,
            architecture_graph=self.architecture_graph,
            constraints=self.constraint_summary,
        )


class ExperimentExecutor:
    def execute(
        self,
        *,
        config: ExperimentConfig,
        experiment: Experiment,
        registries: dict[str, Any],
    ) -> ExecutionResult:
        if experiment.search is not None:
            return self._execute_search(config=config, experiment=experiment, registries=registries)
        return self._execute_training(config=config, experiment=experiment)

    def _execute_search(
        self,
        *,
        config: ExperimentConfig,
        experiment: Experiment,
        registries: dict[str, Any],
    ) -> ExecutionResult:
        summary = SearchRunner().run(config=config, experiment=experiment, registries=registries)
        runtime_info = runtime_environment(
            config.runtime.get("device"),
            resolved=summary.metadata.get("device"),
        )
        notes = (
            f"best model params={summary.best_model_params}; "
            f"evaluations={len(summary.evaluation_history)}; "
            f"search={summary.metadata.get('search_method')}; "
            f"{format_runtime_environment(runtime_info)}"
        )
        constraint_summary = ConstraintEvaluator().evaluate(
            architecture_spec=summary.architecture_spec,
        ).to_dict() if summary.architecture_spec is not None else None
        return ExecutionResult(
            name=config.name,
            mode="search",
            summary=summary.best_summary,
            final_hidden_dim=summary.final_hidden_dim,
            architecture_spec=summary.architecture_spec,
            architecture_graph=summary.architecture_graph,
            constraint_summary=constraint_summary,
            metadata={
                "method_type": "search",
                "notes": notes,
                "runtime_environment": runtime_info,
            },
            search_history=summary.evaluation_history,
            best_model_params=summary.best_model_params,
            best_score=summary.best_score,
        )

    def _execute_training(
        self,
        *,
        config: ExperimentConfig,
        experiment: Experiment,
    ) -> ExecutionResult:
        dataset = experiment.dataset.build()
        summary = experiment.workflow.execute(
            model=experiment.model,
            dataset=dataset,
            metrics=experiment.metrics,
            training_runner=TrainingRunner(),
            adaptation=experiment.adaptation,
            epochs=int(config.trainer.get("epochs", 1)),
            trainer_config=dict(config.trainer),
        )
        final_hidden_dim = None
        if isinstance(experiment.model, DynamicNeuralModel):
            final_hidden_dim = int(experiment.model.architecture_state().metadata.get("hidden_dim", 0))
        architecture_spec, architecture_graph = extract_architecture_artifacts(experiment.model, name=config.name)
        state_metadata = _state_metadata_for_model(experiment.model)
        constraint_summary = ConstraintEvaluator().evaluate(
            architecture_spec=architecture_spec,
            metadata=state_metadata,
        ).to_dict() if architecture_spec is not None or state_metadata else None
        method_type = "baseline"
        if experiment.adaptation is not None:
            method_type = "dynamic"
        elif config.workflow is not None and config.workflow.name != "single_stage":
            method_type = "workflow"
        runtime_info = runtime_environment(
            config.runtime.get("device"),
            resolved=str(getattr(experiment.model, "device", None) or "cpu"),
        )
        route_summary = None
        route_summary_factory = getattr(experiment.model, "route_summary", None)
        if callable(route_summary_factory):
            route_summary = route_summary_factory()
        route_trace = None
        route_trace_factory = getattr(experiment.model, "route_trace", None)
        if callable(route_trace_factory):
            route_trace = route_trace_factory()
        notes_parts = []
        if experiment.adaptation is not None:
            notes_parts.append(f"adaptation={config.adaptation.name}")
        if config.workflow is not None:
            notes_parts.append(f"workflow={config.workflow.name}")
        if route_summary:
            notes_parts.append(f"route_summary={route_summary}")
        notes_parts.append(format_runtime_environment(runtime_info))
        return ExecutionResult(
            name=config.name,
            mode="train",
            summary=summary,
            final_hidden_dim=final_hidden_dim,
            architecture_spec=architecture_spec,
            architecture_graph=architecture_graph,
            constraint_summary=constraint_summary,
            metadata={
                "method_type": method_type,
                "notes": "; ".join(notes_parts) or None,
                "runtime_environment": runtime_info,
                "route_summary": route_summary,
                "route_trace": route_trace,
            },
        )


def _state_metadata_for_model(model: Any) -> dict[str, Any]:
    state_factory = getattr(model, "architecture_state", None)
    if callable(state_factory):
        state = state_factory()
        metadata = getattr(state, "metadata", None)
        if isinstance(metadata, dict):
            return dict(metadata)

    structure_factory = getattr(model, "structure_state", None)
    if callable(structure_factory):
        state_dict = structure_factory()
        if isinstance(state_dict, dict):
            metadata = state_dict.get("metadata")
            if isinstance(metadata, dict):
                return dict(metadata)

    return {}




