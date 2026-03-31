from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dynanets.adaptation.base import AdaptationMethod, AppliedAdaptationEvent
from dynanets.datasets.base import DatasetBundle
from dynanets.metrics.base import Metric
from dynanets.models.base import NeuralModel
from dynanets.models.torch_cnn import TorchCNNClassifier
from dynanets.runners.train import TrainingRunner, TrainingSummary
from dynanets.workflows.base import MethodWorkflow


@dataclass(slots=True)
class PruneTrainWorkflow(MethodWorkflow):
    reconfigure_epochs: list[int] = field(default_factory=lambda: [3, 6])
    prune_fraction: float = 0.10
    min_channels_per_block: int = 8
    metric_name: str = "accuracy"

    def __post_init__(self) -> None:
        if not self.reconfigure_epochs:
            raise ValueError("PruneTrainWorkflow requires at least one reconfigure epoch")
        normalized = sorted({int(epoch) for epoch in self.reconfigure_epochs})
        if normalized[0] <= 0:
            raise ValueError("PruneTrainWorkflow reconfigure epochs must be positive")
        if not 0.0 < self.prune_fraction < 1.0:
            raise ValueError("PruneTrainWorkflow prune_fraction must be in (0.0, 1.0)")
        if self.min_channels_per_block <= 0:
            raise ValueError("PruneTrainWorkflow min_channels_per_block must be positive")
        self.reconfigure_epochs = normalized

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
            raise ValueError("PruneTrainWorkflow manages pruning internally and does not accept adaptation")
        if not isinstance(model, TorchCNNClassifier):
            raise ValueError("PruneTrainWorkflow currently supports TorchCNNClassifier only")
        if not model.spec.use_batch_norm:
            raise ValueError("PruneTrainWorkflow requires a batch-normalized CNN")
        if self.reconfigure_epochs[-1] >= epochs:
            raise ValueError("PruneTrainWorkflow reconfigure epochs must be strictly less than trainer.epochs")

        summary = TrainingSummary()
        cursor = 0
        segment_start = 0

        for milestone in self.reconfigure_epochs + [epochs]:
            segment_epochs = milestone - segment_start
            stage_summary = training_runner.run(
                model=model,
                dataset=dataset,
                metrics=metrics,
                epochs=segment_epochs,
                adaptation=None,
                start_epoch=cursor,
                trainer_config=trainer_config,
            )
            self._extend_stage(
                summary=summary,
                stage_summary=stage_summary,
                stage_name=f"prunetrain_segment_{len(summary.stage_history) + 1}",
                epoch_start=cursor + 1,
                metadata={"phase": "train", "segment_end_epoch": milestone},
            )
            cursor += segment_epochs

            if milestone != epochs:
                before_state = model.structure_state()
                before_meta = before_state.get("metadata", {})
                prune_info = model.prune_channels(
                    prune_fraction=self.prune_fraction,
                    min_channels_per_block=self.min_channels_per_block,
                )
                after_state = model.structure_state()
                after_meta = after_state.get("metadata", {})
                if prune_info.get("pruned"):
                    summary.adaptation_history.append(
                        AppliedAdaptationEvent(
                            epoch=milestone - 1,
                            event_type="prune_channels",
                            params={
                                "prune_fraction": self.prune_fraction,
                                "min_channels_per_block": self.min_channels_per_block,
                            },
                            metadata={
                                "paper": "prunetrain-approx",
                                "strategy": "periodic_reconfiguration",
                                "reconfigure_epoch": milestone,
                                "before_conv_channels": prune_info.get("before_conv_channels"),
                                "after_conv_channels": prune_info.get("after_conv_channels"),
                                "kept_channels": prune_info.get("kept_channels"),
                            },
                            before_state=before_state,
                            after_state=after_state,
                            model_capabilities=model.capabilities(),
                            effect_summary=self._build_effect_summary(before_meta, after_meta, before_state, after_state),
                        ).to_dict()
                    )
            segment_start = milestone

        summary.workflow_metadata = {
            "workflow_name": "prunetrain",
            "configured_total_epochs": epochs,
            "executed_total_epochs": cursor,
            "stage_count": len(summary.stage_history),
            "reconfigure_epochs": list(self.reconfigure_epochs),
            "prune_fraction": self.prune_fraction,
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

    def _build_effect_summary(
        self,
        before_meta: dict[str, Any],
        after_meta: dict[str, Any],
        before_state: dict[str, Any],
        after_state: dict[str, Any],
    ) -> dict[str, Any]:
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
            "num_conv_blocks_delta": _numeric_delta(before_meta.get("num_conv_blocks"), after_meta.get("num_conv_blocks")),
            "conv_channels_before": list(before_meta.get("conv_channels", [])),
            "conv_channels_after": list(after_meta.get("conv_channels", [])),
            "channels_changed": list(before_meta.get("conv_channels", [])) != list(after_meta.get("conv_channels", [])),
        }


def _numeric_delta(before: Any, after: Any) -> int | None:
    if before is None or after is None:
        return None
    return int(after) - int(before)


def _float_delta(before: Any, after: Any) -> float | None:
    if before is None or after is None:
        return None
    return float(after) - float(before)
