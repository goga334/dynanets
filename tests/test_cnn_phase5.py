import importlib.util

import pytest
import torch

from dynanets.architecture import (
    CNNArchitectureSpec,
    ConvBlockSpec,
    build_cnn_network,
    extract_architecture_artifacts,
)
from dynanets.config import ExperimentConfig
from dynanets.datasets.images import MNISTDatasetFactory, SyntheticImagePatternsDatasetFactory
from dynanets.execution import ExperimentExecutor
from dynanets.experiment import ExperimentBuilder, default_registries
from dynanets.models.torch_cnn import TorchCNNClassifier



def _build_experiment(config_dict: dict):
    config = ExperimentConfig.from_dict(config_dict)
    registries = default_registries()
    builder = ExperimentBuilder(
        datasets=registries["datasets"],
        models=registries["models"],
        metrics=registries["metrics"],
        adaptations=registries["adaptations"],
        searches=registries["searches"],
        workflows=registries["workflows"],
    )
    experiment = builder.build(config)
    return config, experiment, registries



def test_cnn_spec_roundtrip_and_graph() -> None:
    spec = CNNArchitectureSpec(
        input_channels=1,
        input_size=(28, 28),
        num_classes=10,
        blocks=[ConvBlockSpec(out_channels=16), ConvBlockSpec(out_channels=32, pool="avg")],
        classifier_hidden_dims=[64],
        activation="relu",
        metadata={"tag": "cnn-demo"},
    )
    spec.validate()

    rebuilt = CNNArchitectureSpec.from_dict(spec.to_dict())
    graph = rebuilt.to_graph(name="cnn-demo")

    assert rebuilt.conv_channels == [16, 32]
    assert rebuilt.classifier_hidden_dims == [64]
    assert graph.metadata["family"] == "cnn"
    assert graph.nodes[1].label == "Conv 1 (16, k=3)"
    assert graph.nodes[-1].label == "Output (10)"



def test_build_cnn_network_from_spec() -> None:
    spec = CNNArchitectureSpec(
        input_channels=1,
        input_size=(28, 28),
        num_classes=10,
        blocks=[ConvBlockSpec(out_channels=8), ConvBlockSpec(out_channels=16)],
        classifier_hidden_dims=[32],
        activation="relu",
    )
    network = build_cnn_network(spec)
    logits = network(torch.randn(4, 1, 28, 28))

    assert logits.shape == (4, 10)



def test_synthetic_image_patterns_dataset_shapes() -> None:
    dataset = SyntheticImagePatternsDatasetFactory(
        train_size=64,
        validation_size=16,
        test_size=16,
        image_size=28,
        seed=7,
    ).build()

    assert dataset.train.inputs.shape == (64, 1, 28, 28)
    assert dataset.validation is not None
    assert dataset.validation.inputs.shape == (16, 1, 28, 28)
    assert dataset.metadata["dataset_name"] == "synthetic_image_patterns"
    assert dataset.metadata["num_classes"] == 10



def test_torch_cnn_classifier_training_step_and_artifacts() -> None:
    dataset = SyntheticImagePatternsDatasetFactory(train_size=32, validation_size=16, test_size=16, seed=7).build()
    model = TorchCNNClassifier(
        input_channels=1,
        input_size=[28, 28],
        num_classes=10,
        conv_channels=[8, 16],
        classifier_hidden_dims=[32],
        lr=0.001,
        device="cpu",
    )
    result = model.training_step({"inputs": dataset.train.inputs, "targets": dataset.train.targets})
    spec_dict, graph_dict = extract_architecture_artifacts(model, name="cnn-test")

    assert result["loss"] >= 0.0
    assert spec_dict is not None
    assert graph_dict is not None
    assert graph_dict["metadata"]["family"] == "cnn"



def test_experiment_executor_runs_fixed_cnn_end_to_end() -> None:
    config, experiment, registries = _build_experiment(
        {
            "name": "executor-cnn-test",
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
                    "conv_channels": [8, 16],
                    "classifier_hidden_dims": [32],
                    "lr": 0.001,
                },
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "trainer": {"epochs": 2},
            "runtime": {"seed": 7, "device": "cpu"},
        }
    )

    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)
    report = result.to_report_item()

    assert result.mode == "train"
    assert result.architecture_spec is not None
    assert result.architecture_graph is not None
    assert result.architecture_graph["metadata"]["family"] == "cnn"
    assert report["metadata"]["method_type"] == "baseline"
    assert report["final_val_accuracy"] is not None



def test_mnist_factory_requires_torchvision_when_missing() -> None:
    if importlib.util.find_spec("torchvision") is not None:
        pytest.skip("torchvision is installed in this environment")

    with pytest.raises(RuntimeError):
        MNISTDatasetFactory(download=False).build()
