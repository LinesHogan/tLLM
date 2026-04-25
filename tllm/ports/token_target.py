#!/usr/bin/env python3
"""Formal token target port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tllm.ports.base import Locator, PortKind, PortRead

TargetStepScope = Literal["current"]


@dataclass(frozen=True)
class TokenTargetLocator(Locator):
    step_scope: TargetStepScope

    def __post_init__(self) -> None:
        if self.step_scope != "current":
            raise ValueError(f"unsupported token target step_scope: {self.step_scope}")


class TokenTarget:
    KIND = PortKind.TOKEN_TARGET
    READABLE = True
    WRITABLE = False
    SUPPORTED_PHASES = ("decode", "prefill")
    BACKING_VLLM_STRUCT = (
        "teacher/target token ids aligned with the current training or guidance "
        "objective, when such targets are available to the runtime."
    )

    @staticmethod
    def read(*, step_scope: TargetStepScope = "current") -> PortRead:
        return PortRead(kind=TokenTarget.KIND, locator=TokenTargetLocator(step_scope=step_scope))
