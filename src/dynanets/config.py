from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ComponentConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    dataset: ComponentConfig
    model: ComponentConfig
    metrics: list[ComponentConfig] = field(default_factory=list)
    adaptation: ComponentConfig | None = None
    search: ComponentConfig | None = None
    trainer: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            name=data["name"],
            dataset=ComponentConfig(**data["dataset"]),
            model=ComponentConfig(**data["model"]),
            metrics=[ComponentConfig(**item) for item in data.get("metrics", [])],
            adaptation=ComponentConfig(**data["adaptation"]) if data.get("adaptation") else None,
            search=ComponentConfig(**data["search"]) if data.get("search") else None,
            trainer=data.get("trainer", {}),
            runtime=data.get("runtime", {}),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls.from_dict(data)
