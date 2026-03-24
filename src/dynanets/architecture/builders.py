from __future__ import annotations

from torch import nn

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



def build_mlp_network(spec: MLPArchitectureSpec) -> nn.Sequential:
    spec.validate()
    modules: list[nn.Module] = []
    for layer in spec.dense_layers():
        modules.append(nn.Linear(layer.in_features, layer.out_features, bias=layer.bias))
        if layer.activation is not None:
            modules.append(activation_module(layer.activation))
    return nn.Sequential(*modules)