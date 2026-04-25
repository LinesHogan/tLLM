#!/usr/bin/env python3
"""Runtime helpers for building same-step sampler views."""

from __future__ import annotations

from typing import Any

import torch

from tllm.runtime.sampler_bridge.types import SamplerStepView


def build_sampler_step_view(
    *,
    runtime: Any,
    runner: Any,
    model: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
) -> SamplerStepView | None:
    decode_count = int(getattr(runtime, "decode_count", 0) or 0)
    if logits.ndim != 2 or decode_count <= 0 or int(logits.shape[0]) != decode_count:
        return None
    source_path = str(getattr(runtime, "source_resolved_path", "") or "").strip()
    if not source_path:
        return None
    source_hidden = getattr(runtime, "tap_decode_hidden", {}).get(source_path)
    if not isinstance(source_hidden, torch.Tensor):
        return None
    if int(source_hidden.shape[0]) < decode_count:
        return None
    request_ids: tuple[str, ...] = ()
    prompt_idxs: tuple[int, ...] = ()
    sample_idxs: tuple[int, ...] = ()
    prompt_idx_tensor = getattr(runtime, "decode_prompt_idx_tensor", None)
    sample_idx_tensor = getattr(runtime, "decode_sample_idx_tensor", None)
    if isinstance(prompt_idx_tensor, torch.Tensor) and int(prompt_idx_tensor.numel()) >= decode_count:
        prompt_idx_tensor = prompt_idx_tensor[:decode_count]
    else:
        prompt_idx_tensor = None
    if isinstance(sample_idx_tensor, torch.Tensor) and int(sample_idx_tensor.numel()) >= decode_count:
        sample_idx_tensor = sample_idx_tensor[:decode_count]
    else:
        sample_idx_tensor = None
    if prompt_idx_tensor is None:
        prompt_idxs = tuple(int(x) for x in list(getattr(runtime, "decode_prompt_idxs", []))[:decode_count])
    if sample_idx_tensor is None:
        sample_idxs = tuple(int(x) for x in list(getattr(runtime, "decode_sample_idxs", []))[:decode_count])
    if prompt_idx_tensor is None or sample_idx_tensor is None:
        request_ids = tuple(str(x) for x in list(getattr(runtime, "decode_request_ids", []))[:decode_count])
    if prompt_idx_tensor is None and len(prompt_idxs) != decode_count:
        return None
    if sample_idx_tensor is None and len(sample_idxs) != decode_count:
        return None
    if request_ids and len(request_ids) != decode_count:
        return None
    return SamplerStepView(
        engine_step_id=int(getattr(runtime, "event_step_id", 0) or 0),
        phase="decode",
        logits=logits,
        sampling_metadata=sampling_metadata,
        decode_count=decode_count,
        request_ids=request_ids,
        prompt_idxs=prompt_idxs,
        sample_idxs=sample_idxs,
        prompt_idx_tensor=prompt_idx_tensor,
        sample_idx_tensor=sample_idx_tensor,
        source_hidden=source_hidden[:decode_count],
        device=logits.device,
        model=model,
        runner=runner,
    )
