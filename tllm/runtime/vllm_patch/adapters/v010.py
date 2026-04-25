#!/usr/bin/env python3
"""Prepare-inputs adapter for vLLM 0.10.x."""

from __future__ import annotations

from tllm.runtime.vllm_patch.adapters.base import BasePrepareInputsAdapter


class V010PrepareInputsAdapter(BasePrepareInputsAdapter):
    family_name = "0.10.x"
