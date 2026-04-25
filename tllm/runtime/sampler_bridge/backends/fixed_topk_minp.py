#!/usr/bin/env python3
"""Experimental fixed-topK min-p candidate sampler.

This backend is intentionally opt-in because it is an approximation when min-p
would retain more candidates than the configured cap.
"""

from __future__ import annotations

from typing import Callable

import torch

from tllm.runtime.sampler_bridge.types import CandidateModifierState


def sample_fixed_topk_min_p_candidates(
    *,
    logits: torch.Tensor,
    min_p: torch.Tensor,
    state: CandidateModifierState,
    cap: int,
    record_candidate_stats: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    k = min(max(1, int(cap)), int(logits.shape[1]))
    values, token_ids = logits.topk(k, dim=-1)
    min_p_values = min_p.to(device=logits.device, dtype=torch.float32).clamp_min(torch.finfo(torch.float32).tiny)
    thresholds = values[:, :1].to(dtype=torch.float32) + min_p_values.log().unsqueeze(1)
    keep = values.to(dtype=torch.float32) >= thresholds
    keep[:, 0] = True
    if state.backend == "post_filter_dense_cache" and state.precomputed_dense_logits is not None:
        dense = state.precomputed_dense_logits
        dense_row_ids = torch.arange(int(logits.shape[0]), device=token_ids.device, dtype=torch.long)
        if state.pred_hidden_row_map is not None:
            dense_row_ids = state.pred_hidden_row_map.index_select(0, dense_row_ids)
        distiller_candidate_logits = dense[
            dense_row_ids.to(device=dense.device, dtype=torch.long).unsqueeze(1),
            token_ids.to(device=dense.device, dtype=torch.long),
        ]
        distiller_candidate_logits = distiller_candidate_logits.to(device=logits.device, dtype=logits.dtype)
    else:
        hidden_rows = torch.arange(int(logits.shape[0]), device=state.pred_hidden.device, dtype=torch.long)
        if state.pred_hidden_row_map is not None:
            hidden_rows = state.pred_hidden_row_map.to(device=state.pred_hidden.device, dtype=torch.long).index_select(0, hidden_rows)
        pred_rows = state.pred_hidden.index_select(0, hidden_rows)
        token_rows = state.lm_head_weight.index_select(
            0,
            token_ids.reshape(-1).to(device=state.lm_head_weight.device, dtype=torch.long),
        )
        token_rows = token_rows.view(int(logits.shape[0]), k, int(token_rows.shape[-1]))
        distiller_candidate_logits = (pred_rows.to(dtype=token_rows.dtype).unsqueeze(1) * token_rows).sum(dim=-1)
        if state.lm_head_bias is not None:
            bias = state.lm_head_bias.index_select(
                0,
                token_ids.reshape(-1).to(device=state.lm_head_bias.device, dtype=torch.long),
            )
            distiller_candidate_logits = distiller_candidate_logits + bias.view(int(logits.shape[0]), k)
        distiller_candidate_logits = distiller_candidate_logits.to(device=logits.device, dtype=logits.dtype)
    candidate_logits = ((1.0 + float(state.beta)) * values) - (float(state.beta) * distiller_candidate_logits)
    candidate_logits = candidate_logits.masked_fill(~keep, -float("inf"))
    noise = torch.empty_like(candidate_logits, dtype=torch.float32)
    noise.exponential_()
    pos = (candidate_logits.to(dtype=torch.float32) - noise.log_()).argmax(dim=-1)
    if record_candidate_stats is not None:
        record_candidate_stats(int(logits.shape[0]) * k, int(logits.shape[0]))
    return token_ids.gather(1, pos.unsqueeze(1)).view(-1).to(dtype=torch.long)
