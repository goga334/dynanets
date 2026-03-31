import torch

from dynanets.metrics.classification import AccuracyMetric



def test_accuracy_metric_aligns_target_device() -> None:
    metric = AccuracyMetric()
    predictions = torch.tensor([[0.1, 0.9], [0.9, 0.1]], device="cpu")
    targets = torch.tensor([1, 0], device="cpu")

    assert metric.compute(predictions, targets) == 1.0
