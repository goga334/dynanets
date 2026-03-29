from __future__ import annotations

from dataclasses import dataclass, field

from dynanets.adaptation.base import AdaptationEvent, AdaptationMethod, AdaptationResult, AppliedAdaptationEvent
from dynanets.models.base import ArchitectureState, DynamicNeuralModel


@dataclass(slots=True)
class DynamicNodesAdaptation(AdaptationMethod):
    """Approximate DNS/GA-DNS with generation then pruning phases.

    The paper builds hidden structure progressively until an accuracy target is
    met, then prunes redundant structure. In the current sandbox, we approximate
    this with repeated layer insertion during a generation phase followed by
    width pruning during a pruning phase.
    """

    target_accuracy: float = 0.95
    generation_width: int = 8
    max_layers: int = 3
    prune_every_n_epochs: int = 2
    prune_by: int = 1
    min_hidden_dim: int = 6
    _generation_complete: bool = field(default=False, init=False, repr=False)

    def supported_event_types(self) -> set[str]:
        return {"insert_hidden_layer", "prune_hidden"}

    def maybe_adapt(
        self,
        model: DynamicNeuralModel,
        state: ArchitectureState,
        context: dict[str, object],
    ) -> AdaptationResult:
        epoch = int(context["epoch"])
        epoch_number = epoch + 1
        validation_metrics = context.get("validation_metrics", {})
        accuracy = float(getattr(validation_metrics, "get", lambda *_: 0.0)("accuracy", 0.0))
        hidden_dims = list(state.metadata.get("hidden_dims", []))

        if not self._generation_complete:
            if accuracy >= self.target_accuracy:
                self._generation_complete = True
                return AdaptationResult(applied=False, reason="generation-target-reached", metadata={"accuracy": accuracy})
            if len(hidden_dims) >= self.max_layers:
                self._generation_complete = True
                return AdaptationResult(applied=False, reason="max-layers-reached", metadata={"accuracy": accuracy})

            layer_index = len(hidden_dims)
            event = AdaptationEvent(
                event_type="insert_hidden_layer",
                params={"layer_index": layer_index, "width": self.generation_width},
                metadata={"phase": "generation", "paper": "dynamic-nodes-approx"},
            )
            model.apply_adaptation(event)
            updated_dims = list(hidden_dims)
            updated_dims.insert(layer_index, self.generation_width)
            return AdaptationResult(
                applied=True,
                event=AppliedAdaptationEvent(
                    epoch=epoch,
                    event_type=event.event_type,
                    params=dict(event.params),
                    metadata={
                        "phase": "generation",
                        "hidden_dims": updated_dims,
                        "num_hidden_layers": len(updated_dims),
                        "paper": "dynamic-nodes-approx",
                    },
                ),
            )

        current_hidden = int(state.metadata.get("hidden_dim", 0))
        if self.prune_every_n_epochs > 0 and epoch_number % self.prune_every_n_epochs == 0 and current_hidden > self.min_hidden_dim:
            amount = min(self.prune_by, current_hidden - self.min_hidden_dim)
            event = AdaptationEvent(
                event_type="prune_hidden",
                params={"amount": amount, "min_width": self.min_hidden_dim},
                metadata={"phase": "pruning", "paper": "dynamic-nodes-approx"},
            )
            model.apply_adaptation(event)
            return AdaptationResult(
                applied=True,
                event=AppliedAdaptationEvent(
                    epoch=epoch,
                    event_type=event.event_type,
                    params=dict(event.params),
                    metadata={
                        "phase": "pruning",
                        "hidden_dim": current_hidden - amount,
                        "paper": "dynamic-nodes-approx",
                    },
                ),
            )

        return AdaptationResult(applied=False, reason="phase-condition-not-met", metadata={"accuracy": accuracy})
