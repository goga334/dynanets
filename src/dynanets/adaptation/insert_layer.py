from __future__ import annotations

from dataclasses import dataclass

from dynanets.adaptation.base import AdaptationEvent, AdaptationMethod, AdaptationResult, AppliedAdaptationEvent
from dynanets.models.base import ArchitectureState, DynamicNeuralModel


@dataclass(slots=True)
class LayerInsertionAdaptation(AdaptationMethod):
    every_n_epochs: int = 3
    width: int = 8
    max_layers: int = 3
    layer_index: int = 1

    def supported_event_types(self) -> set[str]:
        return {"insert_hidden_layer"}

    def maybe_adapt(
        self,
        model: DynamicNeuralModel,
        state: ArchitectureState,
        context: dict[str, object],
    ) -> AdaptationResult:
        epoch = int(context["epoch"])
        if self.every_n_epochs <= 0 or (epoch + 1) % self.every_n_epochs != 0:
            return AdaptationResult(applied=False, reason="schedule-not-reached")

        hidden_dims = list(state.metadata.get("hidden_dims", []))
        if len(hidden_dims) >= self.max_layers:
            return AdaptationResult(applied=False, reason="max-layers-reached")

        insertion_index = min(self.layer_index, len(hidden_dims))
        event = AdaptationEvent(event_type="insert_hidden_layer", params={"layer_index": insertion_index, "width": self.width})
        model.apply_adaptation(event)
        updated_dims = list(hidden_dims)
        updated_dims.insert(insertion_index, self.width)
        return AdaptationResult(
            applied=True,
            event=AppliedAdaptationEvent(
                epoch=epoch,
                event_type=event.event_type,
                params=dict(event.params),
                metadata={"hidden_dims": updated_dims, "num_hidden_layers": len(updated_dims)},
            ),
        )