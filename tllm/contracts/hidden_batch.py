#!/usr/bin/env python3
"""Shared payload contract emitted by hidden producers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch


@dataclass(frozen=True)
class HiddenBatch:
    """A minimal hidden payload independent from runtime timing semantics.

    HiddenBatch only holds tensor references. It does not clone `rows_hidden`.
    In the current decode producer, `rows_hidden` is typically a view onto a
    scratch buffer populated earlier via `torch.index_select(...)`.
    Consumers that need ownership beyond the dispatch window must clone
    explicitly.
    """

    step_id: int
    phase: Literal["decode", "prefill"]
    layer_path: str
    rows_hidden: torch.Tensor
    row_idx: torch.Tensor
    valid_mask: torch.Tensor
    prompt_idx: torch.Tensor
    sample_idx: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)
