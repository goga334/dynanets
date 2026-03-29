from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from dynanets.architecture.graph import ArchitectureEdge, ArchitectureGraph, ArchitectureNode


@dataclass(slots=True)
class DenseLayerSpec:
    in_features: int
    out_features: int
    activation: str | None = None
    bias: bool = True


@dataclass(slots=True)
class MLPArchitectureSpec:
    input_dim: int
    output_dim: int
    hidden_dims: list[int] = field(default_factory=list)
    hidden_activation: str = "relu"
    output_activation: str | None = None
    bias: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer width")
        if any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain only positive widths")
        if not isinstance(self.hidden_activation, str) or not self.hidden_activation.strip():
            raise ValueError("hidden_activation must be a non-empty string")

    @property
    def hidden_dim(self) -> int:
        return self.hidden_dims[0]

    def with_hidden_dims(self, hidden_dims: list[int]) -> "MLPArchitectureSpec":
        return MLPArchitectureSpec(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=list(hidden_dims),
            hidden_activation=self.hidden_activation,
            output_activation=self.output_activation,
            bias=self.bias,
            metadata=dict(self.metadata),
        )

    def dense_layers(self) -> list[DenseLayerSpec]:
        self.validate()
        dims = [self.input_dim, *self.hidden_dims, self.output_dim]
        layers: list[DenseLayerSpec] = []
        for index in range(len(dims) - 1):
            activation = self.hidden_activation if index < len(self.hidden_dims) else self.output_activation
            layers.append(
                DenseLayerSpec(
                    in_features=dims[index],
                    out_features=dims[index + 1],
                    activation=activation,
                    bias=self.bias,
                )
            )
        return layers

    def to_graph(self, name: str = "mlp") -> ArchitectureGraph:
        self.validate()
        nodes: list[ArchitectureNode] = [
            ArchitectureNode(
                id="input",
                label=f"Input ({self.input_dim})",
                op="input",
                params={"features": self.input_dim},
            )
        ]
        for index, width in enumerate(self.hidden_dims, start=1):
            nodes.append(
                ArchitectureNode(
                    id=f"hidden_{index}",
                    label=f"Hidden {index} ({width})",
                    op="linear",
                    params={"features": width, "activation": self.hidden_activation, "bias": self.bias},
                )
            )
        nodes.append(
            ArchitectureNode(
                id="output",
                label=f"Output ({self.output_dim})",
                op="linear",
                params={"features": self.output_dim, "activation": self.output_activation, "bias": self.bias},
            )
        )
        edges = [ArchitectureEdge(source=left.id, target=right.id) for left, right in zip(nodes, nodes[1:])]
        return ArchitectureGraph(
            name=name,
            nodes=nodes,
            edges=edges,
            metadata={"family": "mlp", **dict(self.metadata)},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MLPArchitectureSpec":
        spec = cls(
            input_dim=int(data["input_dim"]),
            output_dim=int(data["output_dim"]),
            hidden_dims=[int(value) for value in data["hidden_dims"]],
            hidden_activation=data.get("hidden_activation", "relu"),
            output_activation=data.get("output_activation"),
            bias=bool(data.get("bias", True)),
            metadata=dict(data.get("metadata", {})),
        )
        spec.validate()
        return spec


def mlp_spec_from_params(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int | None = None,
    hidden_dims: list[int] | None = None,
    activation: str = "relu",
    bias: bool = True,
    metadata: dict[str, Any] | None = None,
) -> MLPArchitectureSpec:
    if hidden_dims is None and hidden_dim is None:
        raise ValueError("Either hidden_dim or hidden_dims must be provided")
    resolved_hidden_dims = list(hidden_dims) if hidden_dims is not None else [int(hidden_dim)]
    spec = MLPArchitectureSpec(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=resolved_hidden_dims,
        hidden_activation=activation,
        bias=bias,
        metadata=dict(metadata or {}),
    )
    spec.validate()
    return spec
