from dynanets.architecture.builders import build_mlp_network
from dynanets.architecture.mlp import DenseLayerSpec, MLPArchitectureSpec, mlp_spec_from_params
from dynanets.architecture.mutations import (
    grow_hidden_layer,
    insert_hidden_layer,
    remove_hidden_layer,
    replace_hidden_activation,
    shrink_hidden_layer,
)

__all__ = [
    "DenseLayerSpec",
    "MLPArchitectureSpec",
    "build_mlp_network",
    "grow_hidden_layer",
    "insert_hidden_layer",
    "mlp_spec_from_params",
    "remove_hidden_layer",
    "replace_hidden_activation",
    "shrink_hidden_layer",
]