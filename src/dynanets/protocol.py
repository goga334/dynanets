from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ProtocolValidationError(ValueError):
    """Raised when a benchmark protocol manifest is invalid."""


@dataclass(slots=True)
class ProtocolExperiment:
    config: str
    role: str = "method"
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BenchmarkProtocol:
    name: str
    track: str
    description: str
    seeds: list[int]
    required_metrics: list[str]
    experiments: list[ProtocolExperiment]
    budgets: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name.strip():
            raise ProtocolValidationError("Protocol name must be non-empty")
        if not self.track.strip():
            raise ProtocolValidationError("Protocol track must be non-empty")
        if not self.experiments:
            raise ProtocolValidationError("Protocol must include at least one experiment")
        if not self.seeds:
            raise ProtocolValidationError("Protocol must include at least one seed")
        if any(not isinstance(seed, int) for seed in self.seeds):
            raise ProtocolValidationError("Protocol seeds must be integers")
        if not self.required_metrics:
            raise ProtocolValidationError("Protocol must include at least one required metric")

    def resolve_configs(self, base_dir: Path) -> list[ProtocolExperiment]:
        resolved: list[ProtocolExperiment] = []
        for item in self.experiments:
            path = Path(item.config)
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            resolved.append(ProtocolExperiment(config=str(path), role=item.role, tags=list(item.tags)))
        return resolved

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkProtocol":
        manifest_path = Path(path)
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        protocol = cls(
            name=str(data["name"]),
            track=str(data["track"]),
            description=str(data.get("description", "")),
            seeds=[int(seed) for seed in data.get("seeds", [])],
            required_metrics=[str(metric) for metric in data.get("required_metrics", [])],
            experiments=[
                ProtocolExperiment(
                    config=str(item["config"]),
                    role=str(item.get("role", "method")),
                    tags=[str(tag) for tag in item.get("tags", [])],
                )
                for item in data.get("experiments", [])
            ],
            budgets=dict(data.get("budgets", {})),
            metadata=dict(data.get("metadata", {})),
        )
        protocol.validate()
        return protocol


__all__ = ["BenchmarkProtocol", "ProtocolExperiment", "ProtocolValidationError"]
