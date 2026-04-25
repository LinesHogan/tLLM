#!/usr/bin/env python3
"""Formal request metadata port."""

from __future__ import annotations

from dataclasses import dataclass

from tllm.ports.base import Locator, PortKind, PortRead


@dataclass(frozen=True)
class RequestMetaLocator(Locator):
    """Logical locator for request identity metadata."""


class RequestMeta:
    KIND = PortKind.REQUEST_META
    READABLE = True
    WRITABLE = False
    SUPPORTED_PHASES = ("prefill", "decode")
    BACKING_VLLM_STRUCT = (
        "request identity and sampling metadata derived from runtime request "
        "mapping, including request_id / prompt_idx / sample_idx / phase / step."
    )

    @staticmethod
    def read() -> PortRead:
        return PortRead(kind=RequestMeta.KIND, locator=RequestMetaLocator())
