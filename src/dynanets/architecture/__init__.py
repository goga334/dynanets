from dynanets.architecture.artifacts import extract_architecture_artifacts
from dynanets.architecture.builders import build_cnn_network, build_mlp_network
from dynanets.architecture.cnn import CNNArchitectureSpec, ConvBlockSpec, cnn_spec_from_params
from dynanets.architecture.graph import ArchitectureEdge, ArchitectureGraph, ArchitectureNode
from dynanets.architecture.mlp import DenseLayerSpec, MLPArchitectureSpec, mlp_spec_from_params
from dynanets.architecture.mutations import (
    grow_hidden_layer,
    insert_hidden_layer,
    remove_hidden_layer,
    replace_hidden_activation,
    shrink_hidden_layer,
)

__all__ = [
    "ArchitectureEdge",
    "ArchitectureGraph",
    "ArchitectureNode",
    "CNNArchitectureSpec",
    "ConvBlockSpec",
    "DenseLayerSpec",
    "MLPArchitectureSpec",
    "build_cnn_network",
    "build_mlp_network",
    "cnn_spec_from_params",
    "extract_architecture_artifacts",
    "grow_hidden_layer",
    "insert_hidden_layer",
    "mlp_spec_from_params",
    "remove_hidden_layer",
    "replace_hidden_activation",
    "shrink_hidden_layer",
]
