#!/usr/bin/env python3
"""Stable consumer-facing input bundle contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class BundleKey:
    engine_step_id: int
    phase: str
    request_id: str
    sample_idx: int


@dataclass(frozen=True)
class PortBundle:
    key: BundleKey
    entries: Mapping[str, Any] = field(default_factory=dict)
