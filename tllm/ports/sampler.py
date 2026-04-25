#!/usr/bin/env python3
"""Formal sampler-step port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from tllm.ports.base import Locator, PortKind, PortRead
from tllm.runtime.sampler_bridge.types import CandidateModifierRequest, CandidateModifierState

SamplerStepScope = Literal["current"]


@dataclass(frozen=True)
class SamplerLocator(Locator):
    step_scope: SamplerStepScope

    def __post_init__(self) -> None:
        if self.step_scope != "current":
            raise ValueError(f"unsupported sampler step_scope: {self.step_scope}")


class Sampler:
    KIND = PortKind.SAMPLER
    READABLE = True
    WRITABLE = False
    SUPPORTED_PHASES = ("decode",)
    BACKING_VLLM_STRUCT = (
        "same-step decode sampler view including aligned logits, request metadata, "
        "and source hidden rows immediately before final sampling."
    )

    @staticmethod
    def read(*, step_scope: SamplerStepScope = "current") -> PortRead:
        return PortRead(kind=Sampler.KIND, locator=SamplerLocator(step_scope=step_scope))


@runtime_checkable
class CandidateModifierProvider(Protocol):
    """Public sampler-port provider contract for candidate-level modifiers."""

    def is_active(self) -> bool:
        ...

    def prepare_candidate_state(
        self,
        request: CandidateModifierRequest,
    ) -> CandidateModifierState | None:
        ...
