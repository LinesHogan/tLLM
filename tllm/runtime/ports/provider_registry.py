#!/usr/bin/env python3
"""Registry for runtime port providers."""

from __future__ import annotations

from typing import Dict, Generic, TypeVar

from tllm.ports.base import PortKind


ProviderT = TypeVar("ProviderT")


class ProviderRegistry(Generic[ProviderT]):
    def __init__(self) -> None:
        self._providers: Dict[PortKind, ProviderT] = {}

    def register(self, kind: PortKind, provider: ProviderT) -> None:
        self._providers[kind] = provider

    def get(self, kind: PortKind) -> ProviderT:
        return self._providers[kind]

    def has(self, kind: PortKind) -> bool:
        return kind in self._providers
