#!/usr/bin/env python3
"""Prepare-inputs adapter for vLLM 0.11+."""

from __future__ import annotations

from tllm.runtime.vllm_patch.adapters.base import BasePrepareInputsAdapter


class V011PlusPrepareInputsAdapter(BasePrepareInputsAdapter):
    family_name = "0.11+"
