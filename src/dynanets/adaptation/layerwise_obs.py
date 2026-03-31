from __future__ import annotations

from dataclasses import dataclass

from dynanets.adaptation.base import AdaptationEvent, AdaptationMethod, AdaptationResult, AppliedAdaptationEvent
from dynanets.models.base import ArchitectureState, DynamicNeuralModel


@dataclass(slots=True)
class LayerwiseOBSAdaptation(AdaptationMethod):
    """Approximate layer-wise OBS via iterative per-layer magnitude pruning."""

    start_epoch: int = 2
    prune_every_n_epochs: int = 1
    prune_fraction: float = 0.08
    max_sparsity: float = 0.60

    def supported_event_types(self) -> set[str]:
        return {"apply_weight_mask"}

    def maybe_adapt(
        self,
        model: DynamicNeuralModel,
        state: ArchitectureState,
        context: dict[str, object],
    ) -> AdaptationResult:
        epoch = int(context["epoch"])
        epoch_number = epoch + 1
        if epoch_number < self.start_epoch:
            return AdaptationResult(applied=False, reason="warmup-not-finished")
        if self.prune_every_n_epochs <= 0 or (epoch_number - self.start_epoch) % self.prune_every_n_epochs != 0:
            return AdaptationResult(applied=False, reason="schedule-not-reached")

        current_sparsity = float(state.metadata.get("weight_sparsity", 0.0))
        if current_sparsity >= self.max_sparsity - 1e-8:
            return AdaptationResult(applied=False, reason="max-sparsity-reached")

        threshold_fn = getattr(model, "layerwise_weight_thresholds", None)
        if not callable(threshold_fn):
            return AdaptationResult(applied=False, reason="model-does-not-support-layerwise-thresholds")

        target_sparsity = min(self.max_sparsity, current_sparsity + self.prune_fraction)
        thresholds_by_name = {
            str(name): float(value)
            for name, value in threshold_fn(self.prune_fraction).items()
        }
        event = AdaptationEvent(
            event_type="apply_weight_mask",
            params={
                "thresholds_by_name": thresholds_by_name,
                "target_sparsity": target_sparsity,
                "layerwise_prune_fraction": self.prune_fraction,
            },
            metadata={"paper": "layerwise-obs-approx", "strategy": "layerwise_magnitude_pruning"},
        )
        model.apply_adaptation(event)
        return AdaptationResult(
            applied=True,
            event=AppliedAdaptationEvent(
                epoch=epoch,
                event_type=event.event_type,
                params=dict(event.params),
                metadata={
                    "paper": "layerwise-obs-approx",
                    "strategy": "layerwise_magnitude_pruning",
                    "target_sparsity": round(target_sparsity, 4),
                    "layerwise_prune_fraction": self.prune_fraction,
                    "thresholds_by_name": {name: round(value, 8) for name, value in thresholds_by_name.items()},
                },
            ),
        )
