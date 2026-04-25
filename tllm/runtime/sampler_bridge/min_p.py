#!/usr/bin/env python3
"""Min-p filtering helper for post-filter sampler intervention."""

from __future__ import annotations

import torch


def apply_min_p(logits: torch.Tensor, min_p: torch.Tensor | None) -> torch.Tensor:
    if min_p is None:
        return logits
    if logits.ndim != 2:
        raise ValueError(f"expected rank-2 logits tensor, got shape={tuple(logits.shape)}")
    if min_p.ndim != 1 or int(min_p.numel()) != int(logits.shape[0]):
        raise ValueError("min_p must be a rank-1 tensor matching the logits row count")
    min_p_values = min_p.to(device=logits.device, dtype=torch.float32).clamp_min(torch.finfo(torch.float32).tiny)
    max_logits = logits.amax(dim=-1, keepdim=True).to(dtype=torch.float32)
    thresholds = max_logits + min_p_values.log().unsqueeze(1)
    keep = logits.to(dtype=torch.float32) >= thresholds
    best = logits.argmax(dim=-1, keepdim=True)
    keep.scatter_(1, best, True)
    logits.masked_fill_(~keep, -float("inf"))
    return logits


def min_p_keep_mask(logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"expected rank-2 logits tensor, got shape={tuple(logits.shape)}")
    if min_p.ndim != 1 or int(min_p.numel()) != int(logits.shape[0]):
        raise ValueError("min_p must be a rank-1 tensor matching the logits row count")
    min_p_values = min_p.to(device=logits.device, dtype=torch.float32).clamp_min(torch.finfo(torch.float32).tiny)
    max_logits = logits.amax(dim=-1, keepdim=True).to(dtype=torch.float32)
    thresholds = max_logits + min_p_values.log().unsqueeze(1)
    keep = logits.to(dtype=torch.float32) >= thresholds
    best = logits.argmax(dim=-1, keepdim=True)
    keep.scatter_(1, best, True)
    return keep
