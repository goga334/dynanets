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
class ProtocolAcceptance:
    min_seeds: int | None = None
    required_roles: list[str] = field(default_factory=list)
    required_experiments: list[str] = field(default_factory=list)
    minimum_methods: int | None = None
    minimum_baselines: int | None = None
    require_constraints: bool = False
    require_runtime_environment: bool = False
    require_stage_history: bool = False

    def validate(self) -> None:
        if self.min_seeds is not None and self.min_seeds < 1:
            raise ProtocolValidationError("Acceptance min_seeds must be positive when provided")
        if self.minimum_methods is not None and self.minimum_methods < 0:
            raise ProtocolValidationError("Acceptance minimum_methods cannot be negative")
        if self.minimum_baselines is not None and self.minimum_baselines < 0:
            raise ProtocolValidationError("Acceptance minimum_baselines cannot be negative")


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
    tier: str = "preview"
    acceptance: ProtocolAcceptance = field(default_factory=ProtocolAcceptance)
    leaderboard: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name.strip():
            raise ProtocolValidationError("Protocol name must be non-empty")
        if not self.track.strip():
            raise ProtocolValidationError("Protocol track must be non-empty")
        if not self.tier.strip():
            raise ProtocolValidationError("Protocol tier must be non-empty")
        if not self.experiments:
            raise ProtocolValidationError("Protocol must include at least one experiment")
        if not self.seeds:
            raise ProtocolValidationError("Protocol must include at least one seed")
        if any(not isinstance(seed, int) for seed in self.seeds):
            raise ProtocolValidationError("Protocol seeds must be integers")
        if not self.required_metrics:
            raise ProtocolValidationError("Protocol must include at least one required metric")
        self.acceptance.validate()

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
            tier=str(data.get("tier", "preview")),
            acceptance=ProtocolAcceptance(
                min_seeds=(int(data.get("acceptance", {}).get("min_seeds")) if data.get("acceptance", {}).get("min_seeds") is not None else None),
                required_roles=[str(role) for role in data.get("acceptance", {}).get("required_roles", [])],
                required_experiments=[str(item) for item in data.get("acceptance", {}).get("required_experiments", [])],
                minimum_methods=(int(data.get("acceptance", {}).get("minimum_methods")) if data.get("acceptance", {}).get("minimum_methods") is not None else None),
                minimum_baselines=(int(data.get("acceptance", {}).get("minimum_baselines")) if data.get("acceptance", {}).get("minimum_baselines") is not None else None),
                require_constraints=bool(data.get("acceptance", {}).get("require_constraints", False)),
                require_runtime_environment=bool(data.get("acceptance", {}).get("require_runtime_environment", False)),
                require_stage_history=bool(data.get("acceptance", {}).get("require_stage_history", False)),
            ),
            leaderboard=dict(data.get("leaderboard", {})),
        )
        protocol.validate()
        return protocol


__all__ = [
    "BenchmarkProtocol",
    "ProtocolAcceptance",
    "ProtocolExperiment",
    "ProtocolValidationError",
]
