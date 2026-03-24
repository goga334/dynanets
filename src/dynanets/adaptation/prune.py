from __future__ import annotations

from dataclasses import dataclass

from dynanets.adaptation.base import AdaptationEvent, AdaptationMethod, AdaptationResult, AppliedAdaptationEvent
from dynanets.models.base import ArchitectureState, DynamicNeuralModel


@dataclass(slots=True)
class WidthPruningAdaptation(AdaptationMethod):
    every_n_epochs: int = 3
    prune_by: int = 2
    min_hidden_dim: int = 4

    def maybe_adapt(
        self,
        model: DynamicNeuralModel,
        state: ArchitectureState,
        context: dict[str, object],
    ) -> AdaptationResult:
        epoch = int(context["epoch"])
        if self.every_n_epochs <= 0 or (epoch + 1) % self.every_n_epochs != 0:
            return AdaptationResult(applied=False, reason="schedule-not-reached")

        current_hidden = int(state.metadata.get("hidden_dim", 0))
        if current_hidden <= self.min_hidden_dim:
            return AdaptationResult(applied=False, reason="min-hidden-reached")

        amount = min(self.prune_by, current_hidden - self.min_hidden_dim)
        event = AdaptationEvent(
            event_type="prune_hidden",
            params={"amount": amount, "min_width": self.min_hidden_dim},
        )
        model.apply_adaptation(event)
        return AdaptationResult(
            applied=True,
            event=AppliedAdaptationEvent(
                epoch=epoch,
                event_type=event.event_type,
                params=dict(event.params),
                metadata={"hidden_dim": current_hidden - amount},
            ),
        )