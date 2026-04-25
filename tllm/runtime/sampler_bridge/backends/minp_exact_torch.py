#!/usr/bin/env python3
"""Torch reference backend for min-p exact candidate sampling."""

from __future__ import annotations

from typing import Any

import torch

from tllm.runtime.sampler_bridge.min_p import apply_min_p, min_p_keep_mask
from tllm.runtime.sampler_bridge.truth import project_candidate_logits, select_candidate_pairs
from tllm.runtime.sampler_bridge.types import CandidateModifierState


def sample_rows_exact_candidates(
    *,
    logits: torch.Tensor,
    sampling_metadata: Any,
    state: CandidateModifierState,
    row_ids: torch.Tensor,
    all_rows: bool | None = None,
) -> torch.Tensor | None:
    """Sample directly from post-filter candidates without rebuilding full-vocab probs."""
    from tllm.runtime.sampler_bridge import exact_backend

    top_k_hint = getattr(sampling_metadata, "top_k", None)
    top_p_hint = getattr(sampling_metadata, "top_p", None)
    pure_min_p = top_k_hint is None and top_p_hint is None
    subset_logits, subset_state, top_k, top_p, min_p, generators = exact_backend._select_subset_inputs(
        logits=logits,
        sampling_metadata=sampling_metadata,
        state=state,
        row_ids=row_ids,
        all_rows=all_rows,
        clone_logits=not pure_min_p,
    )
    if generators or min_p is None:
        return None

    filtered = exact_backend.apply_top_k_top_p(subset_logits, top_k, top_p)
    if top_k is None and top_p is None:
        candidate_row_ids, token_ids = min_p_keep_mask(filtered, min_p).nonzero(as_tuple=True)
    else:
        filtered = apply_min_p(filtered, min_p)
        candidate_row_ids, token_ids = select_candidate_pairs(filtered)
    if candidate_row_ids.numel() <= 0:
        return filtered.argmax(dim=-1).to(dtype=torch.long)
    llm_candidate_logits = filtered[candidate_row_ids, token_ids]
    exact_backend._record_candidate_stats(candidate_count=candidate_row_ids.numel(), row_count=filtered.shape[0])
    if subset_state.backend == "post_filter_dense_cache" and subset_state.precomputed_dense_logits is not None:
        distiller_candidate_logits = exact_backend._gather_dense_candidate_logits(
            state=subset_state,
            row_ids=candidate_row_ids,
            token_ids=token_ids,
            dtype=filtered.dtype,
            device=filtered.device,
        )
    else:
        distiller_candidate_logits = project_candidate_logits(
            pred_hidden=subset_state.pred_hidden,
            row_ids=candidate_row_ids,
            token_ids=token_ids,
            lm_head_weight=subset_state.lm_head_weight,
            lm_head_bias=subset_state.lm_head_bias,
            pred_hidden_row_map=subset_state.pred_hidden_row_map,
        )
    candidate_logits = ((1.0 + float(subset_state.beta)) * llm_candidate_logits) - (
        float(subset_state.beta) * distiller_candidate_logits.to(device=filtered.device, dtype=filtered.dtype)
    )
    noise = torch.empty_like(candidate_logits, dtype=torch.float32)
    noise.exponential_()
    scores = candidate_logits.to(dtype=torch.float32) - noise.log_()
    row_count = int(filtered.shape[0])
    row_scores = torch.full((row_count,), -float("inf"), device=filtered.device, dtype=torch.float32)
    row_scores.scatter_reduce_(0, candidate_row_ids, scores, reduce="amax", include_self=True)
    winner_mask = scores == row_scores.index_select(0, candidate_row_ids)
    sampled = torch.empty((row_count,), device=filtered.device, dtype=torch.long)
    sampled.scatter_(0, candidate_row_ids[winner_mask], token_ids[winner_mask].to(device=sampled.device, dtype=torch.long))
    return sampled
