#!/usr/bin/env python3
"""Generate wrappers that preserve request->prompt mapping during capture."""

from __future__ import annotations

from typing import Any


def run_generate_with_request_mapping(
    *,
    runtime: Any,
    llm: Any,
    prompts: Any,
    params: Any,
    request_prompt_indices: Any = None,
    request_sample_indices: Any = None,
):
    """Run one generate call while tracking request->prompt index mapping."""
    orig_add_request = llm.llm_engine.add_request
    counter = {"i": 0}
    runtime.reqid_to_promptidx = {}
    runtime.reqid_to_sampleidx = {}

    def _wrapped_add_request(request_id, prompt, p, *args, **kwargs):
        req_i = int(counter["i"])
        if request_prompt_indices is not None and req_i < len(request_prompt_indices):
            runtime.reqid_to_promptidx[str(request_id)] = int(request_prompt_indices[req_i])
        else:
            runtime.reqid_to_promptidx[str(request_id)] = req_i
        if request_sample_indices is not None and req_i < len(request_sample_indices):
            runtime.reqid_to_sampleidx[str(request_id)] = int(request_sample_indices[req_i])
        else:
            runtime.reqid_to_sampleidx[str(request_id)] = 0
        counter["i"] += 1
        return orig_add_request(request_id, prompt, p, *args, **kwargs)

    llm.llm_engine.add_request = _wrapped_add_request  # type: ignore[assignment]
    try:
        outputs = llm.generate(prompts, params)
    finally:
        llm.llm_engine.add_request = orig_add_request  # type: ignore[assignment]
    return outputs
