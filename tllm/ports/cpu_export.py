#!/usr/bin/env python3
"""Formal CPU export sink port."""

from __future__ import annotations

from dataclasses import dataclass

from tllm.ports.base import Locator, PortKind, PortWrite


@dataclass(frozen=True)
class CpuExportLocator(Locator):
    channel: str
    format: str
    schema: str | None = None

    def __post_init__(self) -> None:
        if not str(self.channel).strip():
            raise ValueError("CpuExportLocator.channel must be non-empty")
        if not str(self.format).strip():
            raise ValueError("CpuExportLocator.format must be non-empty")


class CpuExport:
    KIND = PortKind.CPU_EXPORT
    READABLE = False
    WRITABLE = True
    SUPPORTED_PHASES = ("prefill", "decode")
    BACKING_VLLM_STRUCT = (
        "runtime-owned asynchronous CPU sink for exporting model-side data "
        "without making database/file details part of the public port catalog."
    )

    @staticmethod
    def write(*, channel: str, format: str, schema: str | None = None) -> PortWrite:
        return PortWrite(
            kind=CpuExport.KIND,
            locator=CpuExportLocator(channel=channel, format=format, schema=schema),
        )
