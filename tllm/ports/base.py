#!/usr/bin/env python3
"""Core public port contracts for consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Sequence


class PortKind(str, Enum):
    RESIDUAL_STREAM = "residual_stream"
    KV_CACHE = "kv_cache"
    LOGITS = "logits"
    SAMPLER = "sampler"
    TOKEN_TARGET = "token_target"
    REQUEST_META = "request_meta"
    CPU_EXPORT = "cpu_export"


Window = Literal["background", "same_step", "next_step", "out_of_band_train"]


@dataclass(frozen=True)
class Locator:
    """Base logical locator for a public runtime port."""


@dataclass(frozen=True)
class PortRead:
    kind: PortKind
    locator: Locator | None = None
    role: str = ""


@dataclass(frozen=True)
class PortWrite:
    kind: PortKind
    locator: Locator | None = None
    mode: str = ""


@dataclass(frozen=True)
class ConsumerFlow:
    reads: Sequence[PortRead]
    writes: Sequence[PortWrite]
    window: Window
    bundle_key: tuple[str, ...] = field(default_factory=tuple)
    dispatch_every_n_steps: int = 1
    max_bundle_rows: int = 0
