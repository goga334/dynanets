from __future__ import annotations

from dataclasses import dataclass, field

from dynanets.adaptation.base import AdaptationEvent, AdaptationMethod, AdaptationResult, AppliedAdaptationEvent
from dynanets.models.base import ArchitectureState, DynamicNeuralModel


@dataclass(slots=True)
class ScheduledChannelPruningAdaptation(AdaptationMethod):
    every_n_epochs: int = 6
    prune_fraction: float = 0.15
    min_channels_per_block: int = 8

    def supported_event_types(self) -> set[str]:
        return {"prune_channels"}

    def maybe_adapt(
        self,
        model: DynamicNeuralModel,
        state: ArchitectureState,
        context: dict[str, object],
    ) -> AdaptationResult:
        epoch = int(context["epoch"])
        if self.every_n_epochs <= 0 or (epoch + 1) % self.every_n_epochs != 0:
            return AdaptationResult(applied=False, reason="schedule-not-reached")
        return _apply_channel_pruning(
            model=model,
            state=state,
            epoch=epoch,
            prune_fraction=self.prune_fraction,
            min_channels_per_block=self.min_channels_per_block,
            strategy="scheduled_channel_pruning",
            paper="channel-pruning-generic",
        )


ChannelPruningAdaptation = ScheduledChannelPruningAdaptation


@dataclass(slots=True)
class ValidationStallChannelPruningAdaptation(AdaptationMethod):
    warmup_epochs: int = 2
    patience: int = 1
    min_delta: float = 1e-3
    prune_fraction: float = 0.12
    min_channels_per_block: int = 8
    metric_name: str = "accuracy"
    cooldown_epochs: int = 1
    _best_score: float | None = field(default=None, init=False, repr=False)
    _epochs_since_improvement: int = field(default=0, init=False, repr=False)
    _last_prune_epoch: int | None = field(default=None, init=False, repr=False)

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

        metrics = context.get("validation_metrics", {})
        if not isinstance(metrics, dict) or self.metric_name not in metrics:
            return AdaptationResult(applied=False, reason="missing-validation-metric")
        score = float(metrics[self.metric_name])

        if self._best_score is None or score > (self._best_score + self.min_delta):
            self._best_score = score
            self._epochs_since_improvement = 0
            return AdaptationResult(applied=False, reason="metric-improved")

        self._epochs_since_improvement += 1
        if self._last_prune_epoch is not None and (epoch_number - self._last_prune_epoch) <= self.cooldown_epochs:
            return AdaptationResult(applied=False, reason="cooldown-active")
        if self._epochs_since_improvement < self.patience:
            return AdaptationResult(applied=False, reason="stall-patience-not-reached")

        result = _apply_channel_pruning(
            model=model,
            state=state,
            epoch=epoch,
            prune_fraction=self.prune_fraction,
            min_channels_per_block=self.min_channels_per_block,
            strategy="validation_stall_channel_pruning",
            paper="channel-pruning-generic",
            extra_metadata={
                "metric_name": self.metric_name,
                "best_score": self._best_score,
                "observed_score": score,
                "epochs_since_improvement": self._epochs_since_improvement,
            },
        )
        if result.applied:
            self._last_prune_epoch = epoch_number
            self._epochs_since_improvement = 0
        return result



def _apply_channel_pruning(
    *,
    model: DynamicNeuralModel,
    state: ArchitectureState,
    epoch: int,
    prune_fraction: float,
    min_channels_per_block: int,
    strategy: str,
    paper: str,
    extra_metadata: dict[str, object] | None = None,
) -> AdaptationResult:
    current_channels = list(state.metadata.get("conv_channels", []))
    if not current_channels:
        return AdaptationResult(applied=False, reason="missing-conv-channel-state")
    if all(channel <= min_channels_per_block for channel in current_channels):
        return AdaptationResult(applied=False, reason="min-channels-reached")

    event = AdaptationEvent(
        event_type="prune_channels",
        params={
            "prune_fraction": prune_fraction,
            "min_channels_per_block": min_channels_per_block,
        },
        metadata={"paper": paper, "strategy": strategy, **(extra_metadata or {})},
    )
    model.apply_adaptation(event)
    updated_channels = list(model.architecture_state().metadata.get("conv_channels", current_channels))
    if updated_channels == current_channels:
        return AdaptationResult(applied=False, reason="no-channel-change")

    metadata = {
        "paper": paper,
        "strategy": strategy,
        "before_conv_channels": current_channels,
        "after_conv_channels": updated_channels,
        **(extra_metadata or {}),
    }
    return AdaptationResult(
        applied=True,
        event=AppliedAdaptationEvent(
            epoch=epoch,
            event_type=event.event_type,
            params=dict(event.params),
            metadata=metadata,
        ),
    )


