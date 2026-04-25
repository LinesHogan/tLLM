#!/usr/bin/env python3
"""Pure-torch truth helpers for distiller sampler intervention."""

from __future__ import annotations

import torch


def select_candidate_pairs(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return row/token ids for finite candidate logits."""
    if logits.ndim != 2:
        raise ValueError(f"expected rank-2 logits tensor, got shape={tuple(logits.shape)}")
    return torch.isfinite(logits).nonzero(as_tuple=True)


def project_candidate_logits(
    *,
    pred_hidden: torch.Tensor,
    row_ids: torch.Tensor,
    token_ids: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    pred_hidden_row_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project predicted hidden states onto selected LM-head token rows."""
    if pred_hidden.ndim != 2:
        raise ValueError(f"expected rank-2 predicted hidden, got shape={tuple(pred_hidden.shape)}")
    if row_ids.ndim != 1 or token_ids.ndim != 1 or row_ids.shape != token_ids.shape:
        raise ValueError("row_ids and token_ids must be rank-1 tensors with matching shape")
    if row_ids.numel() == 0:
        return torch.empty((0,), device=pred_hidden.device, dtype=pred_hidden.dtype)
    hidden_row_ids = row_ids.to(device=pred_hidden.device, dtype=torch.long)
    if pred_hidden_row_map is not None:
        hidden_row_ids = pred_hidden_row_map.to(device=pred_hidden.device, dtype=torch.long).index_select(0, hidden_row_ids)
    pred_rows = pred_hidden.index_select(0, hidden_row_ids)
    token_rows = lm_head_weight.index_select(0, token_ids.to(device=lm_head_weight.device, dtype=torch.long))
    out = (pred_rows.to(dtype=token_rows.dtype) * token_rows).sum(dim=1)
    if lm_head_bias is not None:
        out = out + lm_head_bias.index_select(0, token_ids.to(device=lm_head_bias.device, dtype=torch.long))
    return out.to(device=pred_hidden.device)


def apply_candidate_intervention(
    *,
    logits: torch.Tensor,
    row_ids: torch.Tensor,
    token_ids: torch.Tensor,
    distiller_candidate_logits: torch.Tensor,
    beta: float,
    in_place: bool = False,
) -> torch.Tensor:
    """Apply the candidate-only distiller intervention formula."""
    if logits.ndim != 2:
        raise ValueError(f"expected rank-2 logits tensor, got shape={tuple(logits.shape)}")
    if row_ids.ndim != 1 or token_ids.ndim != 1 or distiller_candidate_logits.ndim != 1:
        raise ValueError("row_ids, token_ids, and distiller_candidate_logits must be rank-1")
    if not (row_ids.shape == token_ids.shape == distiller_candidate_logits.shape):
        raise ValueError("row_ids, token_ids, and distiller_candidate_logits must have matching shape")
    if float(beta) == 0.0 or row_ids.numel() == 0:
        return logits if in_place else logits.clone()
    out = logits if in_place else logits.clone()
    row_ids = row_ids.to(device=out.device, dtype=torch.long)
    token_ids = token_ids.to(device=out.device, dtype=torch.long)
    dist = distiller_candidate_logits.to(device=out.device, dtype=out.dtype)
    llm = out[row_ids, token_ids]
    out[row_ids, token_ids] = ((1.0 + float(beta)) * llm) - (float(beta) * dist)
    return out
