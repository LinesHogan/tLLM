#!/usr/bin/env python3
"""Formal KV cache port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tllm.ports.base import Locator, PortKind, PortRead, PortWrite

KVPhase = Literal["decode"]
KVStepScope = Literal["current", "next"]


@dataclass(frozen=True)
class KVLocator(Locator):
    layer: int
    phase: KVPhase
    step_scope: KVStepScope

    def __post_init__(self) -> None:
        if not isinstance(self.layer, int):
            raise ValueError("KVLocator.layer must be an int")
        if self.phase != "decode":
            raise ValueError(f"unsupported KV phase: {self.phase}")
        if self.step_scope not in {"current", "next"}:
            raise ValueError(f"unsupported KV step_scope: {self.step_scope}")


class KVCache:
    KIND = PortKind.KV_CACHE
    READABLE = True
    WRITABLE = True
    SUPPORTED_PHASES = ("decode",)
    BACKING_VLLM_STRUCT = (
        "layer-local KV cache state used by vLLM decode execution; future "
        "writeback is intended for controlled next-step interventions."
    )

    @staticmethod
    def read(*, layer: int, phase: KVPhase = "decode", step_scope: KVStepScope = "current") -> PortRead:
        return PortRead(kind=KVCache.KIND, locator=KVLocator(layer=layer, phase=phase, step_scope=step_scope))

    @staticmethod
    def write(*, layer: int, phase: KVPhase = "decode", step_scope: KVStepScope = "next") -> PortWrite:
        return PortWrite(kind=KVCache.KIND, locator=KVLocator(layer=layer, phase=phase, step_scope=step_scope))
