from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from dynanets.architecture.graph import ArchitectureEdge, ArchitectureGraph, ArchitectureNode


@dataclass(slots=True)
class ConvBlockSpec:
    out_channels: int
    kernel_size: int = 3
    pool: str | None = "max"

    def validate(self) -> None:
        if self.out_channels <= 0:
            raise ValueError("ConvBlockSpec.out_channels must be positive")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("ConvBlockSpec.kernel_size must be a positive odd integer")
        if self.pool not in {None, "max", "avg"}:
            raise ValueError("ConvBlockSpec.pool must be one of None, 'max', or 'avg'")


@dataclass(slots=True)
class CNNArchitectureSpec:
    input_channels: int
    input_size: tuple[int, int]
    num_classes: int
    blocks: list[ConvBlockSpec] = field(default_factory=list)
    classifier_hidden_dims: list[int] = field(default_factory=list)
    activation: str = "relu"
    use_batch_norm: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if len(self.input_size) != 2 or any(value <= 0 for value in self.input_size):
            raise ValueError("input_size must be a pair of positive integers")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not self.blocks:
            raise ValueError("blocks must contain at least one convolution block")
        if any(width <= 0 for width in self.classifier_hidden_dims):
            raise ValueError("classifier_hidden_dims must contain only positive widths")
        if not isinstance(self.activation, str) or not self.activation.strip():
            raise ValueError("activation must be a non-empty string")
        for block in self.blocks:
            block.validate()

    @property
    def conv_channels(self) -> list[int]:
        return [block.out_channels for block in self.blocks]

    @property
    def final_feature_channels(self) -> int:
        return self.blocks[-1].out_channels

    def to_graph(self, name: str = "cnn") -> ArchitectureGraph:
        self.validate()
        nodes: list[ArchitectureNode] = [
            ArchitectureNode(
                id="input",
                label=f"Input ({self.input_channels}x{self.input_size[0]}x{self.input_size[1]})",
                op="input",
                params={"channels": self.input_channels, "size": list(self.input_size)},
            )
        ]
        edges: list[ArchitectureEdge] = []
        previous_id = "input"
        for index, block in enumerate(self.blocks, start=1):
            conv_id = f"conv_{index}"
            nodes.append(
                ArchitectureNode(
                    id=conv_id,
                    label=f"Conv {index} ({block.out_channels}, k={block.kernel_size})",
                    op="conv2d",
                    params={"out_channels": block.out_channels, "kernel_size": block.kernel_size},
                )
            )
            edges.append(ArchitectureEdge(source=previous_id, target=conv_id))
            previous_id = conv_id
            if block.pool is not None:
                pool_id = f"pool_{index}"
                nodes.append(
                    ArchitectureNode(
                        id=pool_id,
                        label=f"{block.pool.title()}Pool {index}",
                        op=f"{block.pool}pool2d",
                        params={"stride": 2},
                    )
                )
                edges.append(ArchitectureEdge(source=previous_id, target=pool_id))
                previous_id = pool_id
        nodes.append(
            ArchitectureNode(
                id="global_pool",
                label="GlobalPool",
                op="adaptiveavgpool2d",
                params={"output_size": [1, 1]},
            )
        )
        edges.append(ArchitectureEdge(source=previous_id, target="global_pool"))
        previous_id = "global_pool"
        for index, width in enumerate(self.classifier_hidden_dims, start=1):
            hidden_id = f"fc_hidden_{index}"
            nodes.append(
                ArchitectureNode(
                    id=hidden_id,
                    label=f"FC {index} ({width})",
                    op="linear",
                    params={"out_features": width, "activation": self.activation},
                )
            )
            edges.append(ArchitectureEdge(source=previous_id, target=hidden_id))
            previous_id = hidden_id
        nodes.append(
            ArchitectureNode(
                id="output",
                label=f"Output ({self.num_classes})",
                op="linear",
                params={"out_features": self.num_classes},
            )
        )
        edges.append(ArchitectureEdge(source=previous_id, target="output"))
        return ArchitectureGraph(
            name=name,
            nodes=nodes,
            edges=edges,
            metadata={"family": "cnn", **dict(self.metadata)},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CNNArchitectureSpec":
        input_size = data.get("input_size", [28, 28])
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        spec = cls(
            input_channels=int(data["input_channels"]),
            input_size=(int(input_size[0]), int(input_size[1])),
            num_classes=int(data["num_classes"]),
            blocks=[ConvBlockSpec(**dict(block)) for block in data.get("blocks", [])],
            classifier_hidden_dims=[int(value) for value in data.get("classifier_hidden_dims", [])],
            activation=str(data.get("activation", "relu")),
            use_batch_norm=bool(data.get("use_batch_norm", False)),
            metadata=dict(data.get("metadata", {})),
        )
        spec.validate()
        return spec


def cnn_spec_from_params(
    *,
    input_channels: int,
    input_size: int | list[int] | tuple[int, int] = (28, 28),
    num_classes: int,
    conv_channels: list[int] | None = None,
    classifier_hidden_dims: list[int] | None = None,
    activation: str = "relu",
    use_batch_norm: bool = False,
    metadata: dict[str, Any] | None = None,
) -> CNNArchitectureSpec:
    if conv_channels is None or not conv_channels:
        raise ValueError("conv_channels must contain at least one output channel value")
    resolved_input_size = input_size
    if isinstance(resolved_input_size, int):
        resolved_input_size = (resolved_input_size, resolved_input_size)
    spec = CNNArchitectureSpec(
        input_channels=int(input_channels),
        input_size=(int(resolved_input_size[0]), int(resolved_input_size[1])),
        num_classes=int(num_classes),
        blocks=[ConvBlockSpec(out_channels=int(value)) for value in conv_channels],
        classifier_hidden_dims=[int(value) for value in (classifier_hidden_dims or [])],
        activation=activation,
        use_batch_norm=use_batch_norm,
        metadata=dict(metadata or {}),
    )
    spec.validate()
    return spec
