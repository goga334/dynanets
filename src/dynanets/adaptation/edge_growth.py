from __future__ import annotations

from dataclasses import dataclass, field

from dynanets.adaptation.base import AdaptationEvent, AdaptationMethod, AdaptationResult, AppliedAdaptationEvent
from dynanets.models.base import ArchitectureState, DynamicNeuralModel


@dataclass(slots=True)
class EdgeGrowthAdaptation(AdaptationMethod):
    """Approximate SONN-style lifelong growth for edge learning.

    The original work grows neurons and connections in response to new input
    structure over time. In the current sandbox, we approximate that behavior by
    widening under sustained low validation accuracy and inserting an extra
    hidden layer once the width budget is exhausted.
    """

    accuracy_floor: float = 0.96
    grow_by: int = 2
    max_hidden_dim: int = 12
    max_layers: int = 2
    low_accuracy_patience: int = 2
    layer_width: int = 8
    _low_accuracy_epochs: int = field(default=0, init=False, repr=False)

    def supported_event_types(self) -> set[str]:
        return {"net2wider", "insert_hidden_layer"}

    def maybe_adapt(
        self,
        model: DynamicNeuralModel,
        state: ArchitectureState,
        context: dict[str, object],
    ) -> AdaptationResult:
        epoch = int(context["epoch"])
        validation_metrics = context.get("validation_metrics", {})
        accuracy = float(getattr(validation_metrics, "get", lambda *_: 0.0)("accuracy", 0.0))
        hidden_dims = list(state.metadata.get("hidden_dims", []))
        current_hidden = int(state.metadata.get("hidden_dim", 0))

        if accuracy >= self.accuracy_floor:
            self._low_accuracy_epochs = 0
            return AdaptationResult(applied=False, reason="accuracy-floor-met", metadata={"accuracy": accuracy})

        self._low_accuracy_epochs += 1
        if self._low_accuracy_epochs < self.low_accuracy_patience:
            return AdaptationResult(
                applied=False,
                reason="low-accuracy-patience-not-met",
                metadata={"accuracy": accuracy, "low_accuracy_epochs": self._low_accuracy_epochs},
            )

        if current_hidden < self.max_hidden_dim:
            amount = min(self.grow_by, self.max_hidden_dim - current_hidden)
            event = AdaptationEvent(
                event_type="net2wider",
                params={"amount": amount},
                metadata={"trigger": "accuracy_floor", "paper": "edge-growth-approx"},
            )
            model.apply_adaptation(event)
            self._low_accuracy_epochs = 0
            return AdaptationResult(
                applied=True,
                event=AppliedAdaptationEvent(
                    epoch=epoch,
                    event_type=event.event_type,
                    params=dict(event.params),
                    metadata={
                        "trigger": "accuracy_floor",
                        "hidden_dim": current_hidden + amount,
                        "accuracy": accuracy,
                        "paper": "edge-growth-approx",
                    },
                ),
            )

        if len(hidden_dims) < self.max_layers:
            layer_index = len(hidden_dims)
            event = AdaptationEvent(
                event_type="insert_hidden_layer",
                params={"layer_index": layer_index, "width": self.layer_width},
                metadata={"trigger": "accuracy_floor", "paper": "edge-growth-approx"},
            )
            model.apply_adaptation(event)
            self._low_accuracy_epochs = 0
            updated_dims = list(hidden_dims)
            updated_dims.insert(layer_index, self.layer_width)
            return AdaptationResult(
                applied=True,
                event=AppliedAdaptationEvent(
                    epoch=epoch,
                    event_type=event.event_type,
                    params=dict(event.params),
                    metadata={
                        "trigger": "accuracy_floor",
                        "hidden_dims": updated_dims,
                        "accuracy": accuracy,
                        "paper": "edge-growth-approx",
                    },
                ),
            )

        return AdaptationResult(applied=False, reason="growth-budget-exhausted", metadata={"accuracy": accuracy})
