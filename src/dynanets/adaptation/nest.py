from __future__ import annotations

from dataclasses import dataclass

from dynanets.adaptation.base import AdaptationEvent, AdaptationMethod, AdaptationResult, AppliedAdaptationEvent
from dynanets.models.base import ArchitectureState, DynamicNeuralModel


@dataclass(slots=True)
class NeSTAdaptation(AdaptationMethod):
    """Approximate NeST via a two-phase grow-and-prune schedule."""

    grow_every_n_epochs: int = 1
    grow_until_epoch: int = 3
    grow_by: int = 2
    max_hidden_dim: int = 16
    prune_every_n_epochs: int = 1
    prune_start_epoch: int = 4
    prune_by: int = 1
    min_hidden_dim: int = 4

    def supported_event_types(self) -> set[str]:
        return {"net2wider", "prune_hidden"}

    def maybe_adapt(
        self,
        model: DynamicNeuralModel,
        state: ArchitectureState,
        context: dict[str, object],
    ) -> AdaptationResult:
        epoch = int(context["epoch"])
        epoch_number = epoch + 1
        current_hidden = int(state.metadata.get("hidden_dim", 0))

        if epoch_number <= self.grow_until_epoch and self.grow_every_n_epochs > 0:
            if epoch_number % self.grow_every_n_epochs == 0 and current_hidden < self.max_hidden_dim:
                amount = min(self.grow_by, self.max_hidden_dim - current_hidden)
                event = AdaptationEvent(
                    event_type="net2wider",
                    params={"amount": amount},
                    metadata={"phase": "grow", "paper": "nest-approx"},
                )
                model.apply_adaptation(event)
                return AdaptationResult(
                    applied=True,
                    event=AppliedAdaptationEvent(
                        epoch=epoch,
                        event_type=event.event_type,
                        params=dict(event.params),
                        metadata={"hidden_dim": current_hidden + amount, "phase": "grow", "paper": "nest-approx"},
                    ),
                )

        if epoch_number >= self.prune_start_epoch and self.prune_every_n_epochs > 0:
            if (epoch_number - self.prune_start_epoch) % self.prune_every_n_epochs == 0 and current_hidden > self.min_hidden_dim:
                amount = min(self.prune_by, current_hidden - self.min_hidden_dim)
                event = AdaptationEvent(
                    event_type="prune_hidden",
                    params={"amount": amount, "min_width": self.min_hidden_dim},
                    metadata={"phase": "prune", "paper": "nest-approx"},
                )
                model.apply_adaptation(event)
                return AdaptationResult(
                    applied=True,
                    event=AppliedAdaptationEvent(
                        epoch=epoch,
                        event_type=event.event_type,
                        params=dict(event.params),
                        metadata={"hidden_dim": current_hidden - amount, "phase": "prune", "paper": "nest-approx"},
                    ),
                )

        return AdaptationResult(applied=False, reason="phase-condition-not-met")
