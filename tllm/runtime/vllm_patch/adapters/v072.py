#!/usr/bin/env python3
"""Prepare-inputs adapter for vLLM 0.7.x."""

from __future__ import annotations

from tllm.runtime.vllm_patch.adapters.base import BasePrepareInputsAdapter


class V072PrepareInputsAdapter(BasePrepareInputsAdapter):
    family_name = "0.7.x"
