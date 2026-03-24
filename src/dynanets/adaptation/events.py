from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


AdaptationEventType = Literal[
    "grow_hidden",
    "prune_hidden",
    "net2wider",
    "insert_hidden_layer",
]


@dataclass(slots=True)
class AdaptationEvent:
    event_type: AdaptationEventType
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AppliedAdaptationEvent:
    epoch: int
    event_type: AdaptationEventType
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    applied: bool = True
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)