from __future__ import annotations

from dynanets.architecture.mlp import MLPArchitectureSpec



def grow_hidden_layer(spec: MLPArchitectureSpec, *, layer_index: int = 0, amount: int) -> MLPArchitectureSpec:
    if amount <= 0:
        raise ValueError("amount must be positive")
    hidden_dims = list(spec.hidden_dims)
    if layer_index < 0 or layer_index >= len(hidden_dims):
        raise IndexError("layer_index is out of range")
    hidden_dims[layer_index] += amount
    grown = spec.with_hidden_dims(hidden_dims)
    grown.validate()
    return grown



def insert_hidden_layer(
    spec: MLPArchitectureSpec,
    *,
    layer_index: int,
    width: int,
) -> MLPArchitectureSpec:
    if width <= 0:
        raise ValueError("width must be positive")
    hidden_dims = list(spec.hidden_dims)
    if layer_index < 0 or layer_index > len(hidden_dims):
        raise IndexError("layer_index is out of range")
    hidden_dims.insert(layer_index, width)
    inserted = spec.with_hidden_dims(hidden_dims)
    inserted.validate()
    return inserted



def remove_hidden_layer(spec: MLPArchitectureSpec, *, layer_index: int) -> MLPArchitectureSpec:
    hidden_dims = list(spec.hidden_dims)
    if len(hidden_dims) == 1:
        raise ValueError("Cannot remove the only hidden layer")
    if layer_index < 0 or layer_index >= len(hidden_dims):
        raise IndexError("layer_index is out of range")
    hidden_dims.pop(layer_index)
    reduced = spec.with_hidden_dims(hidden_dims)
    reduced.validate()
    return reduced



def replace_hidden_activation(spec: MLPArchitectureSpec, *, activation: str) -> MLPArchitectureSpec:
    if not isinstance(activation, str) or not activation.strip():
        raise ValueError("activation must be a non-empty string")
    updated = MLPArchitectureSpec(
        input_dim=spec.input_dim,
        output_dim=spec.output_dim,
        hidden_dims=list(spec.hidden_dims),
        hidden_activation=activation,
        output_activation=spec.output_activation,
        bias=spec.bias,
        metadata=dict(spec.metadata),
    )
    updated.validate()
    return updated