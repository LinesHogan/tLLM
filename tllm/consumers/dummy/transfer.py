#!/usr/bin/env python3
"""Transfer helpers for the dummy hidden demo GPU->CPU handoff."""

from __future__ import annotations

import torch


def clone_hidden_to_cpu(x: torch.Tensor) -> torch.Tensor:
    """Detach and stage hidden rows on CPU without blocking the caller."""
    return x.detach().to(device="cpu", non_blocking=True)
