import pytest
import torch

from dynanets.adaptation import AdaptationEvent, LayerwiseOBSAdaptation, ValidationStallChannelPruningAdaptation
from dynanets.config import ExperimentConfig
from dynanets.execution import ExperimentExecutor
from dynanets.experiment import ExperimentBuilder, default_registries
from dynanets.models.base import ArchitectureState, DynamicNeuralModel
from dynanets.models.torch_cnn import TorchCNNClassifier
from dynanets.models.torch_mlp import DynamicMLPClassifier
from dynanets.sparsity import resolve_keep_count


class _FakeDynamicCNN(DynamicNeuralModel):
    def __init__(self, channels: list[int]) -> None:
        self._state = ArchitectureState(step=0, version=0, metadata={"conv_channels": list(channels)})

    def forward(self, inputs):
        raise NotImplementedError

    def training_step(self, batch):
        raise NotImplementedError

    def evaluate(self, batch):
        raise NotImplementedError

    def architecture_state(self) -> ArchitectureState:
        return self._state

    def supported_event_types(self) -> set[str]:
        return {"prune_channels"}

    def capabilities(self) -> dict[str, object]:
        return {"supported_event_types": ["prune_channels"]}

    def apply_adaptation(self, event: AdaptationEvent) -> None:
        updated = [
            resolve_keep_count(width, prune_fraction=float(event.params["prune_fraction"]), min_count=int(event.params["min_channels_per_block"]))
            for width in self._state.metadata["conv_channels"]
        ]
        self._state.version += 1
        self._state.metadata["conv_channels"] = updated



def _builder() -> ExperimentBuilder:
    registries = default_registries()
    return ExperimentBuilder(
        datasets=registries["datasets"],
        models=registries["models"],
        metrics=registries["metrics"],
        adaptations=registries["adaptations"],
        searches=registries["searches"],
        workflows=registries["workflows"],
    )



def test_validation_stall_channel_pruning_triggers_after_metric_stall() -> None:
    model = _FakeDynamicCNN([24, 48])
    adaptation = ValidationStallChannelPruningAdaptation(
        warmup_epochs=1,
        patience=1,
        min_delta=0.01,
        prune_fraction=0.25,
        min_channels_per_block=8,
        cooldown_epochs=0,
    )

    result1 = adaptation.maybe_adapt(model, model.architecture_state(), {"epoch": 0, "validation_metrics": {"accuracy": 0.60}})
    result2 = adaptation.maybe_adapt(model, model.architecture_state(), {"epoch": 1, "validation_metrics": {"accuracy": 0.605}})
    result3 = adaptation.maybe_adapt(model, model.architecture_state(), {"epoch": 2, "validation_metrics": {"accuracy": 0.606}})

    assert result1.applied is False
    assert result2.applied is False
    assert result3.applied is True
    assert model.architecture_state().metadata["conv_channels"] == [18, 36]
    assert result3.event is not None
    assert result3.event.event_type == "prune_channels"
    assert result3.event.metadata["strategy"] == "validation_stall_channel_pruning"



def test_torch_cnn_supports_mask_aware_weight_pruning() -> None:
    model = TorchCNNClassifier(
        input_channels=1,
        input_size=[28, 28],
        num_classes=10,
        conv_channels=[8, 16],
        classifier_hidden_dims=[16],
        use_batch_norm=True,
        lr=0.001,
        device="cpu",
    )

    threshold = model.global_weight_threshold(0.2)
    model.apply_adaptation(
        AdaptationEvent(
            event_type="apply_weight_mask",
            params={"threshold": threshold, "target_sparsity": 0.2},
        )
    )
    state = model.architecture_state().metadata

    assert "apply_weight_mask" in model.supported_event_types()
    assert state["masked_weight_count"] > 0
    assert state["weight_sparsity"] > 0.0
    assert state["mask_state_names"]



def test_models_expose_layerwise_weight_thresholds() -> None:
    mlp = DynamicMLPClassifier(input_dim=2, hidden_dim=4, output_dim=2, activation="relu", lr=0.01, device="cpu")
    cnn = TorchCNNClassifier(
        input_channels=1,
        input_size=[28, 28],
        num_classes=10,
        conv_channels=[8, 16],
        classifier_hidden_dims=[16],
        use_batch_norm=True,
        lr=0.001,
        device="cpu",
    )

    mlp_thresholds = mlp.layerwise_weight_thresholds(0.1)
    cnn_thresholds = cnn.layerwise_weight_thresholds(0.1)

    assert set(mlp_thresholds) == {"linear_0.weight", "linear_1.weight"}
    assert set(cnn_thresholds) == {"conv_0.weight", "conv_1.weight", "linear_0.weight", "linear_1.weight"}



def test_layerwise_obs_adaptation_runs_on_cnn() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "layerwise-obs-cnn-test",
            "dataset": {
                "name": "synthetic_image_patterns",
                "params": {"train_size": 128, "validation_size": 32, "test_size": 32, "seed": 7},
            },
            "model": {
                "name": "torch_cnn_classifier",
                "params": {
                    "input_channels": 1,
                    "input_size": [28, 28],
                    "num_classes": 10,
                    "conv_channels": [12, 24],
                    "classifier_hidden_dims": [32],
                    "use_batch_norm": True,
                    "lr": 0.001,
                },
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "adaptation": {
                "name": "layerwise_obs",
                "params": {"start_epoch": 2, "prune_every_n_epochs": 1, "prune_fraction": 0.08, "max_sparsity": 0.3},
            },
            "trainer": {"epochs": 3},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )
    registries = default_registries()
    experiment = _builder().build(config)

    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)

    applied = [event for event in result.summary.adaptation_history if event.get("applied")]
    assert applied
    assert applied[0]["event_type"] == "apply_weight_mask"
    assert applied[0]["metadata"]["paper"] == "layerwise-obs-approx"
    assert "thresholds_by_name" in applied[0]["params"]
    assert result.constraint_summary is not None
    assert result.constraint_summary["masked_weight_count"] > 0
    assert result.constraint_summary["weight_sparsity"] > 0.0
