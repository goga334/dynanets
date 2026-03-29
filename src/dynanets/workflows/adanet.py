from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dynanets.adaptation.base import AdaptationMethod, AppliedAdaptationEvent
from dynanets.adaptation.events import AdaptationEvent
from dynanets.datasets.base import DatasetBundle
from dynanets.metrics.base import Metric
from dynanets.models.base import NeuralModel
from dynanets.models.torch_mlp import DynamicMLPClassifier
from dynanets.runners.train import TrainingRunner, TrainingSummary
from dynanets.workflows.base import MethodWorkflow


@dataclass(slots=True)
class AdaNetRoundsWorkflow(MethodWorkflow):
    rounds: int = 2
    candidate_epochs: int = 4
    warmup_epochs: int = 4
    finetune_epochs: int = 4
    grow_by: int = 4
    insert_width: int | None = None
    max_hidden_dim: int = 64
    max_hidden_layers: int = 3
    complexity_penalty: float = 0.02
    metric_name: str = "accuracy"
    include_identity_candidate: bool = True

    def __post_init__(self) -> None:
        if self.rounds <= 0:
            raise ValueError("AdaNetRoundsWorkflow requires rounds > 0")
        if self.candidate_epochs <= 0:
            raise ValueError("AdaNetRoundsWorkflow requires candidate_epochs > 0")
        if self.warmup_epochs < 0 or self.finetune_epochs < 0:
            raise ValueError("AdaNetRoundsWorkflow warmup/finetune epochs must be non-negative")
        if self.grow_by <= 0:
            raise ValueError("AdaNetRoundsWorkflow requires grow_by > 0")
        if self.max_hidden_dim <= 0 or self.max_hidden_layers <= 0:
            raise ValueError("AdaNetRoundsWorkflow max limits must be positive")
        if self.complexity_penalty < 0.0:
            raise ValueError("AdaNetRoundsWorkflow complexity_penalty must be non-negative")

    def execute(
        self,
        *,
        model: NeuralModel,
        dataset: DatasetBundle,
        metrics: list[Metric],
        training_runner: TrainingRunner,
        adaptation: AdaptationMethod | None,
        epochs: int,
    ) -> TrainingSummary:
        if adaptation is not None:
            raise ValueError("AdaNetRoundsWorkflow manages structure growth internally and does not accept adaptation")
        if not isinstance(model, DynamicMLPClassifier):
            raise ValueError("AdaNetRoundsWorkflow currently supports DynamicMLPClassifier only")

        configured_total = self.warmup_epochs + self.rounds * self.candidate_epochs + self.finetune_epochs
        if epochs != configured_total:
            raise ValueError(
                f"AdaNetRoundsWorkflow requires trainer.epochs={configured_total}, got {epochs}"
            )

        summary = TrainingSummary()
        cursor = 0
        round_records: list[dict[str, Any]] = []
        total_candidate_evaluations = 0
        total_candidate_training_epochs = 0

        if self.warmup_epochs > 0:
            warmup_summary = training_runner.run(
                model=model,
                dataset=dataset,
                metrics=metrics,
                epochs=self.warmup_epochs,
                adaptation=None,
                start_epoch=cursor,
            )
            self._extend_selected_stage(
                summary=summary,
                stage_summary=warmup_summary,
                stage_name="adanet_warmup",
                epoch_start=cursor + 1,
                metadata={"phase": "warmup"},
            )
            cursor += self.warmup_epochs

        for round_index in range(self.rounds):
            before_state = model.architecture_state().to_dict()
            base_parameter_count = max(1, int(before_state.get("metadata", {}).get("parameter_count", 1)))
            candidates = self._build_candidates(model)
            evaluations: list[dict[str, Any]] = []

            for candidate in candidates:
                candidate_summary = training_runner.run(
                    model=candidate["model"],
                    dataset=dataset,
                    metrics=metrics,
                    epochs=self.candidate_epochs,
                    adaptation=None,
                    start_epoch=cursor,
                )
                best_metric = self._best_metric(candidate_summary)
                final_metric = self._final_metric(candidate_summary)
                parameter_count = int(candidate["model"].architecture_state().metadata.get("parameter_count", 0))
                parameter_ratio = parameter_count / base_parameter_count
                selection_score = best_metric - self.complexity_penalty * parameter_ratio
                evaluations.append(
                    {
                        **candidate,
                        "summary": candidate_summary,
                        "best_metric": best_metric,
                        "final_metric": final_metric,
                        "parameter_count": parameter_count,
                        "parameter_ratio": parameter_ratio,
                        "selection_score": selection_score,
                        "hidden_dims": list(candidate["model"].hidden_dims),
                    }
                )

            winner = max(
                evaluations,
                key=lambda item: (item["selection_score"], item["best_metric"], -item["parameter_count"]),
            )
            model.load_from(winner["model"])
            after_state = model.architecture_state().to_dict()

            stage_metadata = {
                "phase": "selection",
                "round_index": round_index + 1,
                "selected_candidate_type": winner["candidate_type"],
                "selected_score": winner["selection_score"],
                "candidate_count": len(evaluations),
                "candidate_training_epochs": self.candidate_epochs * len(evaluations),
                "complexity_penalty": self.complexity_penalty,
                "candidates": [
                    {
                        "candidate_type": item["candidate_type"],
                        "best_metric": item["best_metric"],
                        "final_metric": item["final_metric"],
                        "selection_score": item["selection_score"],
                        "parameter_count": item["parameter_count"],
                        "parameter_ratio": item["parameter_ratio"],
                        "hidden_dims": item["hidden_dims"],
                    }
                    for item in evaluations
                ],
            }
            self._extend_selected_stage(
                summary=summary,
                stage_summary=winner["summary"],
                stage_name=f"adanet_round_{round_index + 1}",
                epoch_start=cursor + 1,
                metadata=stage_metadata,
            )

            if winner["event_type"] is not None:
                selected_event = AppliedAdaptationEvent(
                    epoch=cursor + self.candidate_epochs - 1,
                    event_type=winner["event_type"],
                    params=dict(winner["event_params"]),
                    metadata={
                        "paper": "adanet-approx",
                        "round_index": round_index + 1,
                        "candidate_type": winner["candidate_type"],
                        "selection_score": winner["selection_score"],
                        "best_metric": winner["best_metric"],
                        "complexity_penalty": self.complexity_penalty,
                    },
                    before_state=before_state,
                    after_state=after_state,
                    model_capabilities=model.capabilities(),
                    effect_summary=self._build_effect_summary(before_state, after_state),
                )
                summary.adaptation_history.append(selected_event.to_dict())

            round_records.append(
                {
                    "round_index": round_index + 1,
                    "selected_candidate_type": winner["candidate_type"],
                    "selected_score": winner["selection_score"],
                    "best_metric": winner["best_metric"],
                    "parameter_count": winner["parameter_count"],
                    "hidden_dims": winner["hidden_dims"],
                    "candidate_count": len(evaluations),
                }
            )
            total_candidate_evaluations += len(evaluations)
            total_candidate_training_epochs += self.candidate_epochs * len(evaluations)
            cursor += self.candidate_epochs

        if self.finetune_epochs > 0:
            finetune_summary = training_runner.run(
                model=model,
                dataset=dataset,
                metrics=metrics,
                epochs=self.finetune_epochs,
                adaptation=None,
                start_epoch=cursor,
            )
            self._extend_selected_stage(
                summary=summary,
                stage_summary=finetune_summary,
                stage_name="adanet_consolidate",
                epoch_start=cursor + 1,
                metadata={"phase": "consolidate"},
            )
            cursor += self.finetune_epochs

        summary.workflow_metadata = {
            "workflow_name": "adanet_rounds",
            "configured_total_epochs": epochs,
            "executed_total_epochs": cursor,
            "stage_count": len(summary.stage_history),
            "round_count": self.rounds,
            "total_candidate_evaluations": total_candidate_evaluations,
            "total_candidate_training_epochs": total_candidate_training_epochs,
            "rounds": round_records,
        }
        return summary

    def _build_candidates(self, model: DynamicMLPClassifier) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        if self.include_identity_candidate:
            candidates.append(
                {
                    "candidate_type": "keep",
                    "model": model.clone(),
                    "event_type": None,
                    "event_params": {},
                }
            )

        current_hidden_dim = int(model.architecture_state().metadata.get("hidden_dim", model.hidden_dim))
        if len(model.hidden_dims) == 1 and current_hidden_dim < self.max_hidden_dim:
            wider = model.clone()
            amount = min(self.grow_by, self.max_hidden_dim - current_hidden_dim)
            wider.apply_adaptation(AdaptationEvent(event_type="net2wider", params={"amount": amount, "seed": 42}))
            candidates.append(
                {
                    "candidate_type": "wider",
                    "model": wider,
                    "event_type": "net2wider",
                    "event_params": {"amount": amount, "seed": 42},
                }
            )

        if len(model.hidden_dims) < self.max_hidden_layers:
            deeper = model.clone()
            width = int(self.insert_width or model.hidden_dims[-1])
            event_params = {"layer_index": len(model.hidden_dims), "width": width, "init_mode": "identity"}
            deeper.apply_adaptation(AdaptationEvent(event_type="insert_hidden_layer", params=event_params))
            candidates.append(
                {
                    "candidate_type": "deeper",
                    "model": deeper,
                    "event_type": "insert_hidden_layer",
                    "event_params": event_params,
                }
            )

        if not candidates:
            candidates.append(
                {
                    "candidate_type": "keep",
                    "model": model.clone(),
                    "event_type": None,
                    "event_params": {},
                }
            )
        return candidates

    def _extend_selected_stage(
        self,
        *,
        summary: TrainingSummary,
        stage_summary: TrainingSummary,
        stage_name: str,
        epoch_start: int,
        metadata: dict[str, Any],
    ) -> None:
        summary.train_history.extend(stage_summary.train_history)
        summary.metric_history.extend(stage_summary.metric_history)
        summary.stage_history.append(
            {
                "name": stage_name,
                "epochs": len(stage_summary.train_history),
                "epoch_start": epoch_start,
                "epoch_end": epoch_start + len(stage_summary.train_history) - 1,
                "adaptation_enabled": False,
                "final_train_accuracy": stage_summary.train_history[-1].get("accuracy") if stage_summary.train_history else None,
                "final_val_accuracy": stage_summary.metric_history[-1].get(self.metric_name) if stage_summary.metric_history else None,
                "adaptations_applied": sum(1 for item in stage_summary.adaptation_history if item.get("applied")),
                "metadata": dict(metadata),
            }
        )

    def _best_metric(self, summary: TrainingSummary) -> float:
        if not summary.metric_history:
            return float("-inf")
        return max(float(item.get(self.metric_name, float("-inf"))) for item in summary.metric_history)

    def _final_metric(self, summary: TrainingSummary) -> float:
        if not summary.metric_history:
            return float("-inf")
        return float(summary.metric_history[-1].get(self.metric_name, float("-inf")))

    def _build_effect_summary(self, before_state: dict[str, Any], after_state: dict[str, Any]) -> dict[str, Any]:
        before_meta = before_state.get("metadata", {})
        after_meta = after_state.get("metadata", {})
        before_hidden_dims = list(before_meta.get("hidden_dims", []))
        after_hidden_dims = list(after_meta.get("hidden_dims", []))
        before_hidden_dim = before_meta.get("hidden_dim")
        after_hidden_dim = after_meta.get("hidden_dim")
        before_layers = before_meta.get("num_hidden_layers", len(before_hidden_dims))
        after_layers = after_meta.get("num_hidden_layers", len(after_hidden_dims))
        before_params = before_meta.get("parameter_count")
        after_params = after_meta.get("parameter_count")
        before_nonzero = before_meta.get("nonzero_parameter_count")
        after_nonzero = after_meta.get("nonzero_parameter_count")
        before_sparsity = before_meta.get("weight_sparsity")
        after_sparsity = after_meta.get("weight_sparsity")
        return {
            "applied": before_state != after_state,
            "structural_change": before_state != after_state,
            "version_delta": _numeric_delta(before_state.get("version"), after_state.get("version")),
            "step_delta": _numeric_delta(before_state.get("step"), after_state.get("step")),
            "hidden_dim_delta": _numeric_delta(before_hidden_dim, after_hidden_dim),
            "num_hidden_layers_delta": _numeric_delta(before_layers, after_layers),
            "parameter_count_delta": _numeric_delta(before_params, after_params),
            "nonzero_parameter_count_delta": _numeric_delta(before_nonzero, after_nonzero),
            "weight_sparsity_delta": _float_delta(before_sparsity, after_sparsity),
            "hidden_dims_changed": before_hidden_dims != after_hidden_dims,
        }


def _numeric_delta(before: Any, after: Any) -> int | None:
    if before is None or after is None:
        return None
    return int(after) - int(before)


def _float_delta(before: Any, after: Any) -> float | None:
    if before is None or after is None:
        return None
    return float(after) - float(before)
