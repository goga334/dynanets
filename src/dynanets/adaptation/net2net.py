from __future__ import annotations

from dataclasses import dataclass

from dynanets.adaptation.base import AdaptationMethod, AdaptationResult
from dynanets.models.base import ArchitectureState, DynamicNeuralModel


@dataclass(slots=True)
class Net2WiderAdaptation(AdaptationMethod):
    every_n_epochs: int = 3
    grow_by: int = 4
    max_hidden_dim: int = 128
    seed: int = 42

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
        if current_hidden >= self.max_hidden_dim:
            return AdaptationResult(applied=False, reason="max-hidden-reached")

        amount = min(self.grow_by, self.max_hidden_dim - current_hidden)
        model.apply_adaptation(
            {
                "action": "net2wider",
                "amount": amount,
                "seed": self.seed + epoch,
            }
        )
        return AdaptationResult(
            applied=True,
            changes={"action": "net2wider", "amount": amount, "hidden_dim": current_hidden + amount},
        )