#!/usr/bin/env python3
"""Formal logits port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tllm.ports.base import Locator, PortKind, PortRead

LogitsStepScope = Literal["current"]


@dataclass(frozen=True)
class LogitsLocator(Locator):
    step_scope: LogitsStepScope

    def __post_init__(self) -> None:
        if self.step_scope != "current":
            raise ValueError(f"unsupported logits step_scope: {self.step_scope}")


class Logits:
    KIND = PortKind.LOGITS
    READABLE = True
    WRITABLE = False
    SUPPORTED_PHASES = ("decode",)
    BACKING_VLLM_STRUCT = (
        "current-step logits immediately before sampling/logits post-processing "
        "in the vLLM decode path."
    )

    @staticmethod
    def read(*, step_scope: LogitsStepScope = "current") -> PortRead:
        return PortRead(kind=Logits.KIND, locator=LogitsLocator(step_scope=step_scope))
