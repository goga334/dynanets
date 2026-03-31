from __future__ import annotations

from dataclasses import dataclass, field

from dynanets.adaptation.base import AdaptationMethod, AdaptationResult
from dynanets.adaptation.channel_prune import _apply_channel_pruning
from dynanets.models.base import ArchitectureState, DynamicNeuralModel


@dataclass(slots=True)
class RuntimeNeuralPruningAdaptation(AdaptationMethod):
    """Approximate Runtime Neural Pruning via activation-cost-aware channel pruning."""

    start_epoch: int = 3
    prune_every_n_epochs: int = 2
    prune_fraction: float = 0.10
    min_channels_per_block: int = 8
    target_activation_reduction: float = 0.30
    _initial_activation_elements: int | None = field(default=None, init=False, repr=False)

    def supported_event_types(self) -> set[str]:
        return {"prune_channels"}

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

        current_activation = state.metadata.get("activation_elements")
        if current_activation is None:
            return AdaptationResult(applied=False, reason="missing-activation-cost")
        current_activation = int(current_activation)
        if self._initial_activation_elements is None:
            self._initial_activation_elements = current_activation

        target_activation = int(round(self._initial_activation_elements * (1.0 - self.target_activation_reduction)))
        if current_activation <= target_activation:
            return AdaptationResult(applied=False, reason="target-activation-cost-reached")

        return _apply_channel_pruning(
            model=model,
            state=state,
            epoch=epoch,
            prune_fraction=self.prune_fraction,
            min_channels_per_block=self.min_channels_per_block,
            strategy="activation_cost_channel_pruning",
            paper="runtime-neural-pruning-approx",
            extra_metadata={
                "activation_elements_before": current_activation,
                "target_activation_elements": target_activation,
                "target_activation_reduction": self.target_activation_reduction,
            },
        )
