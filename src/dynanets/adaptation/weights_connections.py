from __future__ import annotations

from dataclasses import dataclass

from dynanets.adaptation.base import AdaptationEvent, AdaptationMethod, AdaptationResult, AppliedAdaptationEvent
from dynanets.models.base import ArchitectureState, DynamicNeuralModel


@dataclass(slots=True)
class WeightsConnectionsAdaptation(AdaptationMethod):
    """Approximate Han et al. via iterative global magnitude pruning with mask application."""

    start_epoch: int = 2
    prune_every_n_epochs: int = 1
    prune_fraction: float = 0.10
    max_sparsity: float = 0.70

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

        threshold_fn = getattr(model, "global_weight_threshold", None)
        if not callable(threshold_fn):
            return AdaptationResult(applied=False, reason="model-does-not-support-weight-thresholds")

        target_sparsity = min(self.max_sparsity, current_sparsity + self.prune_fraction)
        threshold = float(threshold_fn(target_sparsity))
        event = AdaptationEvent(
            event_type="apply_weight_mask",
            params={"threshold": threshold, "target_sparsity": target_sparsity},
            metadata={"paper": "weights-connections-approx", "strategy": "global_magnitude_pruning"},
        )
        model.apply_adaptation(event)
        return AdaptationResult(
            applied=True,
            event=AppliedAdaptationEvent(
                epoch=epoch,
                event_type=event.event_type,
                params=dict(event.params),
                metadata={
                    "paper": "weights-connections-approx",
                    "strategy": "global_magnitude_pruning",
                    "target_sparsity": round(target_sparsity, 4),
                    "threshold": threshold,
                },
            ),
        )
