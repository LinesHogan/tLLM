#!/usr/bin/env python3
"""Optional dense performance backend for distiller sampler intervention."""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch
from torch.autograd.profiler import record_function

from tllm.runtime.sampler_bridge.types import SamplerModifierState

_ENABLE_DISTILLER_RECORD_FUNCTION = os.getenv("TLLM_ENABLE_DISTILLER_RECORD_FUNCTION", "") == "1"


def _maybe_record_function(name: str):
    return record_function(name) if _ENABLE_DISTILLER_RECORD_FUNCTION else nullcontext()


def modify_rows_dense(
    *,
    logits: torch.Tensor,
    state: SamplerModifierState,
    row_ids: torch.Tensor,
) -> torch.Tensor:
    with _maybe_record_function("distiller.modify_rows_dense"):
        pos_tensor = torch.searchsorted(
            state.affected_row_ids.to(device=row_ids.device, dtype=torch.long),
            row_ids.to(device=row_ids.device, dtype=torch.long),
        ).to(device=state.pred_hidden.device, dtype=torch.long)
        if state.precomputed_dense_logits is not None:
            dense = state.precomputed_dense_logits.index_select(0, pos_tensor)
        else:
            pred_hidden = state.pred_hidden.index_select(0, pos_tensor)
            dense = torch.matmul(pred_hidden.to(dtype=state.lm_head_weight.dtype), state.lm_head_weight.transpose(0, 1))
            if state.lm_head_bias is not None:
                dense = dense + state.lm_head_bias
        subset_logits = logits.index_select(0, row_ids).to(dtype=dense.dtype)
        modified = ((1.0 + float(state.beta)) * subset_logits) - (float(state.beta) * dense)
        return modified.to(device=logits.device, dtype=logits.dtype)
