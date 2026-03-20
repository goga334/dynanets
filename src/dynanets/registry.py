from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self) -> None:
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Callable[..., T]) -> None:
        if name in self._items:
            raise ValueError(f"'{name}' is already registered")
        self._items[name] = factory

    def build(self, name: str, **params: object) -> T:
        try:
            factory = self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"Unknown component '{name}'. Available: {available}") from exc
        return factory(**params)

    def names(self) -> list[str]:
        return sorted(self._items)
