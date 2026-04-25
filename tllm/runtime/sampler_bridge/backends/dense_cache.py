#!/usr/bin/env python3
"""Dense-cache helpers for post-filter candidate intervention."""

from __future__ import annotations

import torch

from tllm.runtime.sampler_bridge.types import CandidateModifierState


def gather_dense_candidate_logits(
    *,
    state: CandidateModifierState,
    row_ids: torch.Tensor,
    token_ids: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    assert state.precomputed_dense_logits is not None
    dense_row_ids = row_ids
    if state.pred_hidden_row_map is not None:
        dense_row_ids = state.pred_hidden_row_map.index_select(0, row_ids)
    dense = state.precomputed_dense_logits
    if dense_row_ids.device != dense.device:
        dense_row_ids = dense_row_ids.to(device=dense.device)
    if token_ids.device != dense.device:
        token_ids = token_ids.to(device=dense.device)
    gathered = dense[dense_row_ids, token_ids]
    return gathered.to(device=device, dtype=dtype) if gathered.device != device or gathered.dtype != dtype else gathered
