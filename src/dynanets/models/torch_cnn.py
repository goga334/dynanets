from __future__ import annotations

from typing import Any

import torch
from torch import nn

from dynanets.architecture import CNNArchitectureSpec, build_cnn_network, cnn_spec_from_params
from dynanets.models.base import NeuralModel
from dynanets.runtime import resolve_device


class TorchCNNClassifier(NeuralModel):
    def __init__(
        self,
        input_channels: int,
        input_size: int | list[int] | tuple[int, int] = (28, 28),
        num_classes: int = 10,
        conv_channels: list[int] | None = None,
        classifier_hidden_dims: list[int] | None = None,
        activation: str = "relu",
        use_batch_norm: bool = False,
        lr: float = 1e-3,
        device: str | None = None,
    ) -> None:
        self.spec = cnn_spec_from_params(
            input_channels=input_channels,
            input_size=input_size,
            num_classes=num_classes,
            conv_channels=conv_channels or [16, 32],
            classifier_hidden_dims=classifier_hidden_dims,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )
        self.device = resolve_device(device)
        self._lr = lr
        self.network = build_cnn_network(self.spec).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(self.device)
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(1)
        return self.network(inputs)

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.network.train()
        inputs = batch["inputs"]
        targets = batch["targets"].to(self.device)

        self.optimizer.zero_grad()
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, targets)
        loss.backward()
        self.optimizer.step()

        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
        return {
            "loss": float(loss.item()),
            "accuracy": float(accuracy),
        }

    def evaluate(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.network.eval()
        with torch.no_grad():
            return self.forward(batch["inputs"])

    def architecture_spec(self) -> CNNArchitectureSpec:
        return self.spec

    def init_params(self) -> dict[str, Any]:
        return {
            "input_channels": self.spec.input_channels,
            "input_size": list(self.spec.input_size),
            "num_classes": self.spec.num_classes,
            "conv_channels": self.spec.conv_channels,
            "classifier_hidden_dims": list(self.spec.classifier_hidden_dims),
            "activation": self.spec.activation,
            "use_batch_norm": self.spec.use_batch_norm,
            "lr": self._lr,
            "device": str(self.device),
        }
