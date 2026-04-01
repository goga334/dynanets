from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from dynanets.architecture.graph import ArchitectureEdge, ArchitectureGraph, ArchitectureNode


EfficientBlockKind = Literal["conv", "fire", "grouped"]


@dataclass(slots=True)
class EfficientBlockSpec:
    kind: EfficientBlockKind
    out_channels: int
    kernel_size: int = 3
    squeeze_channels: int | None = None
    groups: int = 1
    pool: str | None = None

    def validate(self) -> None:
        if self.kind not in {"conv", "fire", "grouped"}:
            raise ValueError("Unsupported efficient block kind")
        if self.out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if self.groups <= 0:
            raise ValueError("groups must be positive")
        if self.pool not in {None, "max", "avg"}:
            raise ValueError("pool must be one of None, 'max', or 'avg'")
        if self.kind == "fire" and (self.squeeze_channels is None or self.squeeze_channels <= 0):
            raise ValueError("fire blocks require positive squeeze_channels")


@dataclass(slots=True)
class EfficientCNNArchitectureSpec:
    family: str
    input_channels: int
    input_size: tuple[int, int]
    num_classes: int
    blocks: list[EfficientBlockSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.family:
            raise ValueError("family must be non-empty")
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if len(self.input_size) != 2 or any(value <= 0 for value in self.input_size):
            raise ValueError("input_size must contain two positive integers")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not self.blocks:
            raise ValueError("blocks must be non-empty")
        for block in self.blocks:
            block.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_graph(self, name: str = "efficient_cnn") -> ArchitectureGraph:
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
            node_id = f"block_{index}"
            if block.kind == "fire":
                label = f"Fire {index} (sq={block.squeeze_channels}, out={block.out_channels})"
            elif block.kind == "grouped":
                label = f"Grouped {index} (out={block.out_channels}, g={block.groups})"
            else:
                label = f"Conv {index} (out={block.out_channels}, k={block.kernel_size})"
            nodes.append(
                ArchitectureNode(
                    id=node_id,
                    label=label,
                    op=block.kind,
                    params={
                        "out_channels": block.out_channels,
                        "kernel_size": block.kernel_size,
                        "squeeze_channels": block.squeeze_channels,
                        "groups": block.groups,
                        "pool": block.pool,
                    },
                )
            )
            edges.append(ArchitectureEdge(source=previous_id, target=node_id))
            previous_id = node_id
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
                id="classifier",
                label=f"Classifier ({self.num_classes})",
                op="linear",
                params={"out_features": self.num_classes},
            )
        )
        edges.append(ArchitectureEdge(source=previous_id, target="classifier"))
        return ArchitectureGraph(
            name=name,
            nodes=nodes,
            edges=edges,
            metadata={"family": self.family, **dict(self.metadata)},
        )
