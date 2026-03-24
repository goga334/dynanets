from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigValidationError(ValueError):
    """Raised when an experiment config is structurally invalid."""


@dataclass(slots=True)
class ComponentConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, section: str) -> "ComponentConfig":
        if not isinstance(data, dict):
            raise ConfigValidationError(f"Section '{section}' must be a mapping")
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ConfigValidationError(f"Section '{section}.name' must be a non-empty string")
        params = data.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ConfigValidationError(f"Section '{section}.params' must be a mapping")
        extra_keys = sorted(set(data) - {"name", "params"})
        if extra_keys:
            raise ConfigValidationError(
                f"Section '{section}' contains unsupported keys: {', '.join(extra_keys)}"
            )
        return cls(name=name, params=params)


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

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigValidationError("Field 'name' must be a non-empty string")
        if not isinstance(self.metrics, list):
            raise ConfigValidationError("Field 'metrics' must be a list")
        if not self.metrics:
            raise ConfigValidationError("At least one metric must be defined")
        if not isinstance(self.trainer, dict):
            raise ConfigValidationError("Field 'trainer' must be a mapping")
        if not isinstance(self.runtime, dict):
            raise ConfigValidationError("Field 'runtime' must be a mapping")

        epochs = self.trainer.get("epochs", 1)
        if not isinstance(epochs, int) or epochs <= 0:
            raise ConfigValidationError("Field 'trainer.epochs' must be a positive integer")

        seed = self.runtime.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise ConfigValidationError("Field 'runtime.seed' must be an integer when provided")

        if self.search is not None and self.adaptation is not None:
            raise ConfigValidationError(
                "Search and adaptation cannot currently be enabled in the same experiment config"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        if not isinstance(data, dict):
            raise ConfigValidationError("Experiment config must be a mapping")
        required = {"name", "dataset", "model"}
        missing = sorted(key for key in required if key not in data)
        if missing:
            raise ConfigValidationError(f"Missing required config sections: {', '.join(missing)}")

        metrics_data = data.get("metrics", [])
        if not isinstance(metrics_data, list):
            raise ConfigValidationError("Field 'metrics' must be a list")

        config = cls(
            name=data["name"],
            dataset=ComponentConfig.from_dict(data["dataset"], section="dataset"),
            model=ComponentConfig.from_dict(data["model"], section="model"),
            metrics=[ComponentConfig.from_dict(item, section=f"metrics[{index}]") for index, item in enumerate(metrics_data)],
            adaptation=ComponentConfig.from_dict(data["adaptation"], section="adaptation") if data.get("adaptation") else None,
            search=ComponentConfig.from_dict(data["search"], section="search") if data.get("search") else None,
            trainer=data.get("trainer", {}),
            runtime=data.get("runtime", {}),
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls.from_dict(data)