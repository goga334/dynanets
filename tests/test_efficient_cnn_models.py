import torch

from dynanets.models.efficient_cnn import TorchCondenseStyleCNNClassifier, TorchSqueezeStyleCNNClassifier


def test_squeezenet_style_cnn_runs_training_step_and_exposes_spec() -> None:
    model = TorchSqueezeStyleCNNClassifier(
        input_channels=1,
        input_size=[28, 28],
        num_classes=10,
        stem_channels=16,
        fire_channels=[24, 32],
        lr=0.001,
        device="cpu",
    )
    batch = {
        "inputs": torch.randn(8, 1, 28, 28),
        "targets": torch.randint(0, 10, (8,), dtype=torch.long),
    }

    stats = model.training_step(batch)
    logits = model.evaluate(batch)
    graph = model.architecture_spec().to_graph(name="squeezenet_style")

    assert logits.shape == (8, 10)
    assert "loss" in stats
    assert model.structure_state()["metadata"]["architecture_family"] == "squeezenet_style"
    assert graph.metadata["family"] == "squeezenet_style"


def test_condensenet_style_cnn_runs_training_step_and_exposes_spec() -> None:
    model = TorchCondenseStyleCNNClassifier(
        input_channels=1,
        input_size=[28, 28],
        num_classes=10,
        stem_channels=16,
        grouped_channels=[24, 32],
        groups=[2, 4],
        lr=0.001,
        device="cpu",
    )
    batch = {
        "inputs": torch.randn(8, 1, 28, 28),
        "targets": torch.randint(0, 10, (8,), dtype=torch.long),
    }

    stats = model.training_step(batch)
    logits = model.evaluate(batch)
    graph = model.architecture_spec().to_graph(name="condensenet_style")

    assert logits.shape == (8, 10)
    assert "accuracy" in stats
    assert model.structure_state()["metadata"]["architecture_family"] == "condensenet_style"
    assert graph.metadata["family"] == "condensenet_style"
