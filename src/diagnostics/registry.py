from __future__ import annotations

from typing import Callable, TypeVar

from .base import Diagnostic

REGISTRY: dict[str, type[Diagnostic]] = {}

T = TypeVar("T", bound=type[Diagnostic])


def register_diagnostic(name: str, *, category: str, thresholds: dict) -> Callable[[T], T]:
    def deco(cls: T) -> T:
        cls.name = name
        cls.category = category
        cls.thresholds = dict(thresholds)
        if name in REGISTRY:
            raise ValueError(f"diagnostic '{name}' already registered by {REGISTRY[name]}")
        REGISTRY[name] = cls
        return cls

    return deco


def get(name: str) -> type[Diagnostic]:
    if name not in REGISTRY:
        raise KeyError(f"diagnostic '{name}' not registered; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def names() -> list[str]:
    return sorted(REGISTRY)
