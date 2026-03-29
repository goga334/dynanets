from __future__ import annotations

from dataclasses import dataclass, field

from dynanets.adaptation.base import AdaptationEvent, AdaptationMethod, AdaptationResult, AppliedAdaptationEvent
from dynanets.models.base import ArchitectureState, DynamicNeuralModel


@dataclass(slots=True)
class GradMaxAdaptation(AdaptationMethod):
    """Approximate GradMax via gradient-triggered width growth.

    The original method uses gradient information to choose and initialize new
    neurons. In the current sandbox, we preserve the core idea by using the
    input-layer gradient norm as a growth signal and applying a standard hidden
    width expansion event when that signal is strong enough.
    """

    every_n_epochs: int = 1
    grow_by: int = 2
    max_hidden_dim: int = 32
    grad_norm_threshold: float = 0.05
    min_epochs_between_growth: int = 1
    _last_growth_epoch: int = field(default=-10_000, init=False, repr=False)

    def supported_event_types(self) -> set[str]:
        return {"net2wider"}

    def maybe_adapt(
        self,
        model: DynamicNeuralModel,
        state: ArchitectureState,
        context: dict[str, object],
    ) -> AdaptationResult:
        epoch = int(context["epoch"])
        if self.every_n_epochs <= 0 or (epoch + 1) % self.every_n_epochs != 0:
            return AdaptationResult(applied=False, reason="schedule-not-reached")
        if epoch - self._last_growth_epoch < self.min_epochs_between_growth:
            return AdaptationResult(applied=False, reason="cooldown-active")

        current_hidden = int(state.metadata.get("hidden_dim", 0))
        if current_hidden >= self.max_hidden_dim:
            return AdaptationResult(applied=False, reason="max-hidden-reached")

        train_result = context.get("train_result", {})
        grad_norm = float(getattr(train_result, "get", lambda *_: 0.0)("grad_norm_input_layer", 0.0))
        if grad_norm < self.grad_norm_threshold:
            return AdaptationResult(
                applied=False,
                reason="gradient-threshold-not-reached",
                metadata={"grad_norm_input_layer": grad_norm},
            )

        amount = min(self.grow_by, self.max_hidden_dim - current_hidden)
        event = AdaptationEvent(
            event_type="net2wider",
            params={"amount": amount},
            metadata={"trigger": "gradient_norm", "grad_norm_input_layer": grad_norm, "paper": "gradmax-approx"},
        )
        model.apply_adaptation(event)
        self._last_growth_epoch = epoch
        return AdaptationResult(
            applied=True,
            event=AppliedAdaptationEvent(
                epoch=epoch,
                event_type=event.event_type,
                params=dict(event.params),
                metadata={
                    "hidden_dim": current_hidden + amount,
                    "trigger": "gradient_norm",
                    "grad_norm_input_layer": grad_norm,
                    "paper": "gradmax-approx",
                },
            ),
        )
