#!/usr/bin/env python3
"""Provider protocol for same-step sampler intervention."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tllm.runtime.sampler_bridge.types import CandidateModifierState, SamplerStepView


@runtime_checkable
class SamplerModifierProvider(Protocol):
    def is_active(self) -> bool:
        ...

    def prepare_step(self, view: SamplerStepView) -> CandidateModifierState | None:
        ...
