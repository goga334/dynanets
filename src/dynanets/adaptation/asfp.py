from __future__ import annotations

from dataclasses import dataclass

from dynanets.adaptation.base import AdaptationMethod, AdaptationResult
from dynanets.adaptation.channel_prune import _apply_channel_pruning
from dynanets.models.base import ArchitectureState, DynamicNeuralModel


@dataclass(slots=True)
class AsymptoticSoftFilterPruningAdaptation(AdaptationMethod):
    """Approximate ASFP via progressively stronger channel pruning during training."""

    warmup_epochs: int = 2
    prune_every_n_epochs: int = 1
    total_prune_epochs: int = 6
    min_prune_fraction: float = 0.04
    final_prune_fraction: float = 0.20
    min_channels_per_block: int = 8

    def __post_init__(self) -> None:
        if self.warmup_epochs < 0:
            raise ValueError("AsymptoticSoftFilterPruningAdaptation warmup_epochs must be non-negative")
        if self.prune_every_n_epochs <= 0:
            raise ValueError("AsymptoticSoftFilterPruningAdaptation prune_every_n_epochs must be positive")
        if self.total_prune_epochs <= 0:
            raise ValueError("AsymptoticSoftFilterPruningAdaptation total_prune_epochs must be positive")
        if not 0.0 < self.min_prune_fraction <= self.final_prune_fraction < 1.0:
            raise ValueError("AsymptoticSoftFilterPruningAdaptation prune fractions must satisfy 0 < min <= final < 1")
        if self.min_channels_per_block <= 0:
            raise ValueError("AsymptoticSoftFilterPruningAdaptation min_channels_per_block must be positive")

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
        if epoch_number <= self.warmup_epochs:
            return AdaptationResult(applied=False, reason="warmup-not-finished")
        if (epoch_number - self.warmup_epochs - 1) % self.prune_every_n_epochs != 0:
            return AdaptationResult(applied=False, reason="schedule-not-reached")

        progressive_epoch = min(self.total_prune_epochs, epoch_number - self.warmup_epochs)
        progress = progressive_epoch / max(1, self.total_prune_epochs)
        prune_fraction = self.min_prune_fraction + (self.final_prune_fraction - self.min_prune_fraction) * (progress ** 2)

        return _apply_channel_pruning(
            model=model,
            state=state,
            epoch=epoch,
            prune_fraction=float(prune_fraction),
            min_channels_per_block=self.min_channels_per_block,
            strategy="asymptotic_progressive_channel_pruning",
            paper="asfp-approx",
            extra_metadata={
                "progress": round(progress, 4),
                "scheduled_prune_fraction": round(float(prune_fraction), 4),
                "total_prune_epochs": self.total_prune_epochs,
            },
        )
