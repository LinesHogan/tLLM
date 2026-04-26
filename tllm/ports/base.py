#!/usr/bin/env python3
"""Core public port contracts for consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)

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


class FlowWindow(StrEnum):
    BACKGROUND = "background"
    SAME_STEP = "same_step"
    NEXT_STEP = "next_step"
    OUT_OF_BAND = "out_of_band"


class FlowDelivery(StrEnum):
    BUNDLE = "bundle"
    DEVICE_LEASE = "device_lease"


class BufferOwnership(StrEnum):
    BORROWED = "borrowed"
    RUNTIME_LEASE = "runtime_lease"


class RowSelection(StrEnum):
    ALL = "none"
    FIRST_PER_PROMPT = "first_per_prompt"


Window = FlowWindow | Literal["background", "same_step", "next_step", "out_of_band"]
DeliveryMode = FlowDelivery | Literal["bundle", "device_lease"]
RowCompaction = RowSelection | Literal["none", "first_per_prompt"]


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
    delivery: DeliveryMode = "bundle"
    ownership: BufferOwnership = "borrowed"
    row_compaction: RowCompaction = "none"

    def __post_init__(self) -> None:
        try:
            window = FlowWindow(str(self.window))
        except ValueError as exc:
            raise ValueError(f"unsupported flow window `{self.window}`") from exc
        try:
            delivery = FlowDelivery(str(self.delivery))
        except ValueError as exc:
            raise ValueError(f"unsupported delivery mode `{self.delivery}`") from exc
        try:
            ownership = BufferOwnership(str(self.ownership))
        except ValueError as exc:
            raise ValueError(f"unsupported buffer ownership `{self.ownership}`") from exc
        try:
            row_compaction = RowSelection(str(self.row_compaction))
        except ValueError as exc:
            raise ValueError(f"unsupported row_compaction mode `{self.row_compaction}`") from exc
        bundle_key = tuple(self.bundle_key)
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "delivery", delivery)
        object.__setattr__(self, "ownership", ownership)
        object.__setattr__(self, "row_compaction", row_compaction)
        step_scope_key = ("engine_step_id", "phase")
        if delivery == FlowDelivery.DEVICE_LEASE:
            if ownership != BufferOwnership.RUNTIME_LEASE:
                raise ValueError("device_lease delivery requires ownership='runtime_lease'")
            if bundle_key != step_scope_key:
                raise ValueError("device_lease delivery currently requires bundle_key=('engine_step_id', 'phase')")
        if row_compaction == RowSelection.FIRST_PER_PROMPT and bundle_key != step_scope_key:
            raise ValueError("first_per_prompt row compaction currently requires bundle_key=('engine_step_id', 'phase')")
