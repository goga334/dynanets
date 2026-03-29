from __future__ import annotations

from dataclasses import dataclass, field

from dynanets.adaptation.base import AdaptationEvent, AdaptationMethod, AdaptationResult, AppliedAdaptationEvent
from dynanets.models.base import ArchitectureState, DynamicNeuralModel


@dataclass(slots=True)
class DENAdaptation(AdaptationMethod):
    """Approximate DEN via validation-stall-triggered expansion.

    The original DEN method includes task-aware expansion, duplication, and
    selective retraining. In the current sandbox we approximate the expandable
    network idea by widening the active hidden representation when validation
    improvement stalls.
    """

    metric_name: str = "accuracy"
    patience: int = 1
    min_delta: float = 0.0
    grow_by: int = 4
    max_hidden_dim: int = 32
    max_expansions: int = 2
    cooldown_epochs: int = 1
    _best_metric: float = field(default=float("-inf"), init=False, repr=False)
    _stalled_epochs: int = field(default=0, init=False, repr=False)
    _expansions: int = field(default=0, init=False, repr=False)
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
        validation_metrics = context.get("validation_metrics", {})
        current_metric = float(getattr(validation_metrics, "get", lambda *_: float("nan"))(self.metric_name, float("nan")))
        if current_metric != current_metric:
            return AdaptationResult(applied=False, reason="missing-validation-metric")

        if current_metric > self._best_metric + self.min_delta:
            self._best_metric = current_metric
            self._stalled_epochs = 0
            return AdaptationResult(applied=False, reason="validation-improved", metadata={self.metric_name: current_metric})

        self._stalled_epochs += 1
        if self._expansions >= self.max_expansions:
            return AdaptationResult(applied=False, reason="max-expansions-reached", metadata={self.metric_name: current_metric})
        if epoch - self._last_growth_epoch < self.cooldown_epochs:
            return AdaptationResult(applied=False, reason="cooldown-active", metadata={self.metric_name: current_metric})
        if self._stalled_epochs < self.patience:
            return AdaptationResult(
                applied=False,
                reason="validation-plateau-not-long-enough",
                metadata={self.metric_name: current_metric, "stalled_epochs": self._stalled_epochs},
            )

        current_hidden = int(state.metadata.get("hidden_dim", 0))
        if current_hidden >= self.max_hidden_dim:
            return AdaptationResult(applied=False, reason="max-hidden-reached", metadata={self.metric_name: current_metric})

        amount = min(self.grow_by, self.max_hidden_dim - current_hidden)
        event = AdaptationEvent(
            event_type="net2wider",
            params={"amount": amount},
            metadata={"trigger": "validation_plateau", "metric_name": self.metric_name, "paper": "den-approx"},
        )
        model.apply_adaptation(event)
        self._expansions += 1
        self._stalled_epochs = 0
        self._last_growth_epoch = epoch
        return AdaptationResult(
            applied=True,
            event=AppliedAdaptationEvent(
                epoch=epoch,
                event_type=event.event_type,
                params=dict(event.params),
                metadata={
                    "hidden_dim": current_hidden + amount,
                    "trigger": "validation_plateau",
                    self.metric_name: current_metric,
                    "expansion_index": self._expansions,
                    "paper": "den-approx",
                },
            ),
        )
