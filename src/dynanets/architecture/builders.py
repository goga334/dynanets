from __future__ import annotations

import torch
from torch import nn

from dynanets.architecture.cnn import CNNArchitectureSpec
from dynanets.architecture.mlp import MLPArchitectureSpec


ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


def activation_module(name: str) -> nn.Module:
    try:
        return ACTIVATIONS[name.lower()]()
    except KeyError as exc:
        available = ", ".join(sorted(ACTIVATIONS))
        raise ValueError(f"Unsupported activation '{name}'. Available: {available}") from exc


class CNNClassifierNetwork(nn.Module):
    def __init__(self, features: nn.Sequential, classifier: nn.Sequential) -> None:
        super().__init__()
        self.features = features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = classifier

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.global_pool(features)
        flattened = torch.flatten(pooled, start_dim=1)
        return self.classifier(flattened)


def build_mlp_network(spec: MLPArchitectureSpec) -> nn.Sequential:
    spec.validate()
    modules: list[nn.Module] = []
    for layer in spec.dense_layers():
        modules.append(nn.Linear(layer.in_features, layer.out_features, bias=layer.bias))
        if layer.activation is not None:
            modules.append(activation_module(layer.activation))
    return nn.Sequential(*modules)


def build_cnn_network(spec: CNNArchitectureSpec) -> CNNClassifierNetwork:
    spec.validate()
    features: list[nn.Module] = []
    in_channels = spec.input_channels
    for block in spec.blocks:
        padding = block.kernel_size // 2
        features.append(nn.Conv2d(in_channels, block.out_channels, kernel_size=block.kernel_size, padding=padding))
        if spec.use_batch_norm:
            features.append(nn.BatchNorm2d(block.out_channels))
        features.append(activation_module(spec.activation))
        if block.pool == "max":
            features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif block.pool == "avg":
            features.append(nn.AvgPool2d(kernel_size=2, stride=2))
        in_channels = block.out_channels

    classifier: list[nn.Module] = []
    dims = [spec.final_feature_channels, *spec.classifier_hidden_dims, spec.num_classes]
    for index in range(len(dims) - 1):
        classifier.append(nn.Linear(dims[index], dims[index + 1]))
        if index < len(dims) - 2:
            classifier.append(activation_module(spec.activation))

    return CNNClassifierNetwork(nn.Sequential(*features), nn.Sequential(*classifier))
