from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dynanets.adaptation.base import AdaptationMethod, AppliedAdaptationEvent
from dynanets.datasets.base import DatasetBundle
from dynanets.metrics.base import Metric
from dynanets.models.base import NeuralModel
from dynanets.models.torch_cnn import TorchCNNClassifier
from dynanets.runners.train import TrainingRunner, TrainingSummary
from dynanets.workflows.base import MethodWorkflow


@dataclass(slots=True)
class LayerMergeWorkflow(MethodWorkflow):
    pretrain_epochs: int = 5
    finetune_epochs: int = 3
    merge_index: int = 0
    metric_name: str = "accuracy"

    def __post_init__(self) -> None:
        if self.pretrain_epochs <= 0:
            raise ValueError("LayerMergeWorkflow requires pretrain_epochs > 0")
        if self.finetune_epochs < 0:
            raise ValueError("LayerMergeWorkflow finetune_epochs must be non-negative")
        if self.merge_index < 0:
            raise ValueError("LayerMergeWorkflow merge_index must be non-negative")

    def execute(
        self,
        *,
        model: NeuralModel,
        dataset: DatasetBundle,
        metrics: list[Metric],
        training_runner: TrainingRunner,
        adaptation: AdaptationMethod | None,
        epochs: int,
        trainer_config: dict[str, Any],
    ) -> TrainingSummary:
        if adaptation is not None:
            raise ValueError("LayerMergeWorkflow manages layer merging internally and does not accept adaptation")
        if not isinstance(model, TorchCNNClassifier):
            raise ValueError("LayerMergeWorkflow currently supports TorchCNNClassifier only")
        if len(model.spec.classifier_hidden_dims) < 2:
            raise ValueError("LayerMergeWorkflow requires at least two classifier hidden layers")

        configured_total = self.pretrain_epochs + self.finetune_epochs
        if epochs != configured_total:
            raise ValueError(f"LayerMergeWorkflow requires trainer.epochs={configured_total}, got {epochs}")

        summary = TrainingSummary()
        cursor = 0

        pretrain_summary = training_runner.run(
            model=model,
            dataset=dataset,
            metrics=metrics,
            epochs=self.pretrain_epochs,
            adaptation=None,
            start_epoch=cursor,
            trainer_config=trainer_config,
        )
        self._extend_stage(
            summary=summary,
            stage_summary=pretrain_summary,
            stage_name="layermerge_pretrain",
            epoch_start=cursor + 1,
            metadata={"phase": "pretrain"},
        )
        cursor += self.pretrain_epochs

        before_state = model.structure_state()
        merge_info = model.merge_classifier_layers(merge_index=self.merge_index)
        after_state = model.structure_state()
        summary.adaptation_history.append(
            AppliedAdaptationEvent(
                epoch=cursor - 1,
                event_type="merge_hidden_layers",
                params={"merge_index": self.merge_index},
                metadata={
                    "paper": "layermerge-approx",
                    "before_classifier_hidden_dims": merge_info.get("before_classifier_hidden_dims"),
                    "after_classifier_hidden_dims": merge_info.get("after_classifier_hidden_dims"),
                },
                before_state=before_state,
                after_state=after_state,
                model_capabilities=model.capabilities(),
                effect_summary=self._build_effect_summary(before_state, after_state),
            ).to_dict()
        )

        if self.finetune_epochs > 0:
            finetune_summary = training_runner.run(
                model=model,
                dataset=dataset,
                metrics=metrics,
                epochs=self.finetune_epochs,
                adaptation=None,
                start_epoch=cursor,
                trainer_config=trainer_config,
            )
            self._extend_stage(
                summary=summary,
                stage_summary=finetune_summary,
                stage_name="layermerge_finetune",
                epoch_start=cursor + 1,
                metadata={"phase": "finetune"},
            )
            cursor += self.finetune_epochs

        summary.workflow_metadata = {
            "workflow_name": "layermerge",
            "configured_total_epochs": epochs,
            "executed_total_epochs": cursor,
            "stage_count": len(summary.stage_history),
            "merge_index": self.merge_index,
            "before_classifier_hidden_dims": before_state.get("metadata", {}).get("classifier_hidden_dims"),
            "after_classifier_hidden_dims": after_state.get("metadata", {}).get("classifier_hidden_dims"),
        }
        return summary

    def _extend_stage(
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

    def _build_effect_summary(self, before_state: dict[str, Any], after_state: dict[str, Any]) -> dict[str, Any]:
        before_meta = before_state.get("metadata", {})
        after_meta = after_state.get("metadata", {})
        before_hidden = list(before_meta.get("classifier_hidden_dims", []))
        after_hidden = list(after_meta.get("classifier_hidden_dims", []))
        return {
            "applied": before_state != after_state,
            "structural_change": before_state != after_state,
            "version_delta": _numeric_delta(before_state.get("version"), after_state.get("version")),
            "step_delta": _numeric_delta(before_state.get("step"), after_state.get("step")),
            "parameter_count_delta": _numeric_delta(before_meta.get("parameter_count"), after_meta.get("parameter_count")),
            "nonzero_parameter_count_delta": _numeric_delta(before_meta.get("nonzero_parameter_count"), after_meta.get("nonzero_parameter_count")),
            "weight_sparsity_delta": _float_delta(before_meta.get("weight_sparsity"), after_meta.get("weight_sparsity")),
            "forward_flop_proxy_delta": _numeric_delta(before_meta.get("forward_flop_proxy"), after_meta.get("forward_flop_proxy")),
            "activation_elements_delta": _numeric_delta(before_meta.get("activation_elements"), after_meta.get("activation_elements")),
            "classifier_hidden_dims_before": before_hidden,
            "classifier_hidden_dims_after": after_hidden,
            "hidden_layers_changed": before_hidden != after_hidden,
        }


def _numeric_delta(before: Any, after: Any) -> int | None:
    if before is None or after is None:
        return None
    return int(after) - int(before)


def _float_delta(before: Any, after: Any) -> float | None:
    if before is None or after is None:
        return None
    return float(after) - float(before)
