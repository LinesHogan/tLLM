#!/usr/bin/env python3
"""Adapter selection for vLLM version-specific prepare-inputs handling."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

from tllm.runtime.vllm_patch.adapters.base import (
    BasePrepareInputsAdapter,
    PrepareInputsView,
)
from tllm.runtime.vllm_patch.adapters.v010 import V010PrepareInputsAdapter
from tllm.runtime.vllm_patch.adapters.v011_plus import V011PlusPrepareInputsAdapter
from tllm.runtime.vllm_patch.adapters.v072 import V072PrepareInputsAdapter


def _parse_major_minor(version: str) -> Tuple[int, int]:
    cleaned = version.strip().split("+", 1)[0]
    parts = cleaned.split(".")
    major = int(parts[0]) if len(parts) >= 1 and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
    return major, minor


def _current_vllm_version() -> str:
    try:
        import vllm

        version = getattr(vllm, "__version__", "")
        if isinstance(version, str) and version.strip():
            return version
    except Exception:
        pass

    try:
        from importlib.metadata import version as pkg_version

        return pkg_version("vllm")
    except Exception:
        return "0.0.0"


def select_prepare_inputs_adapter(version: Optional[str] = None) -> BasePrepareInputsAdapter:
    resolved = version or _current_vllm_version()
    major, minor = _parse_major_minor(resolved)
    if major == 0 and minor < 10:
        return V072PrepareInputsAdapter()
    if major == 0 and minor == 10:
        return V010PrepareInputsAdapter()
    return V011PlusPrepareInputsAdapter()


@lru_cache(maxsize=1)
def get_prepare_inputs_adapter() -> BasePrepareInputsAdapter:
    return select_prepare_inputs_adapter()


__all__ = [
    "BasePrepareInputsAdapter",
    "PrepareInputsView",
    "V010PrepareInputsAdapter",
    "V011PlusPrepareInputsAdapter",
    "V072PrepareInputsAdapter",
    "get_prepare_inputs_adapter",
    "select_prepare_inputs_adapter",
]
