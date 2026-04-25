#!/usr/bin/env python3
"""Low-rank model used by ESamp train engine."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class LowRankGatedResidualModel(torch.nn.Module):
    """Low-rank gated residual projector with optional output layernorm."""

    def __init__(
        self,
        *,
        hidden_size: int,
        rank: int,
        use_output_layernorm: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank = max(1, min(int(rank), self.hidden_size))
        self.use_output_layernorm = bool(use_output_layernorm)

        self.a = torch.nn.Parameter(torch.empty((self.hidden_size, self.rank), device=device, dtype=dtype))
        self.g = torch.nn.Parameter(torch.empty((self.hidden_size, self.rank), device=device, dtype=dtype))
        self.b = torch.nn.Parameter(torch.empty((self.rank, self.hidden_size), device=device, dtype=dtype))
        self.gate_bias = torch.nn.Parameter(torch.ones((self.rank,), device=device, dtype=dtype))
        self.out_ln: Optional[torch.nn.LayerNorm]
        if self.use_output_layernorm:
            self.out_ln = torch.nn.LayerNorm(self.hidden_size, elementwise_affine=True, device=device, dtype=dtype)
        else:
            self.out_ln = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.a.normal_(mean=0.0, std=0.02)
            self.g.normal_(mean=0.0, std=0.02)
            self.b.normal_(mean=0.0, std=0.02)
            self.gate_bias.fill_(1.0)
            if self.out_ln is not None:
                self.out_ln.weight.fill_(1.0)
                self.out_ln.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = torch.matmul(x, self.a)
        gate = F.silu(torch.matmul(x, self.g) + self.gate_bias)
        delta = torch.matmul(up * gate, self.b)
        out = delta + x
        if self.out_ln is not None:
            out = self.out_ln(out)
        return out
