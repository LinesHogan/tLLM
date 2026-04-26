#!/usr/bin/env python3
"""ESamp-local model-bank forward backends."""

from __future__ import annotations

from dataclasses import dataclass
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)
from typing import Any, Protocol

import torch
import torch.nn.functional as F

class ModelBankForwardBackendName(StrEnum):
    TORCH = "torch"
    TRITON_GROUPED = "triton_grouped"

import triton
import triton.language as tl

class ESampModelBankForwardBackend(Protocol):
    name: str

    def forward(
        self,
        *,
        slot_ids: torch.Tensor,
        src: torch.Tensor,
        params: Any,
        use_output_layernorm: bool,
    ) -> torch.Tensor:
        ...


def normalize_model_bank_forward_backend(name: str | None) -> ModelBankForwardBackendName:
    key = str(name or "torch").strip().lower() or "torch"
    if key in {"torch", "reference", "default"}:
        return ModelBankForwardBackendName.TORCH
    if key in {"triton", "triton_grouped", "experimental_triton_grouped"}:
        return ModelBankForwardBackendName.TRITON_GROUPED
    raise ValueError(f"unsupported ESamp model-bank forward backend: {key}")


@dataclass(frozen=True)
class TorchModelBankForwardBackend:
    name: str = "torch"

    def forward(
        self,
        *,
        slot_ids: torch.Tensor,
        src: torch.Tensor,
        params: Any,
        use_output_layernorm: bool,
    ) -> torch.Tensor:
        a_rows = params.a.index_select(0, slot_ids)
        g_rows = params.g.index_select(0, slot_ids)
        b_rows = params.b.index_select(0, slot_ids)
        gb_rows = params.gate_bias.index_select(0, slot_ids)
        up = torch.bmm(src.unsqueeze(1), a_rows).squeeze(1)
        gate = F.silu(torch.bmm(src.unsqueeze(1), g_rows).squeeze(1) + gb_rows)
        out = torch.bmm((up * gate).unsqueeze(1), b_rows).squeeze(1) + src
        return _apply_output_layernorm(
            out=out,
            slot_ids=slot_ids,
            params=params,
            use_output_layernorm=use_output_layernorm,
        )


def _apply_output_layernorm(
    *,
    out: torch.Tensor,
    slot_ids: torch.Tensor,
    params: Any,
    use_output_layernorm: bool,
) -> torch.Tensor:
    if not use_output_layernorm:
        return out
    normalized = F.layer_norm(out, (int(out.shape[-1]),), weight=None, bias=None)
    return normalized * params.out_ln_weight.index_select(0, slot_ids) + params.out_ln_bias.index_select(0, slot_ids)


def _triton_available() -> bool:
    return triton is not None and tl is not None


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _triton_supported(slot_ids: torch.Tensor, src: torch.Tensor, params: Any) -> bool:
    if not _triton_available():
        return False
    if src.device.type != "cuda" or slot_ids.device.type != "cuda":
        return False
    if src.ndim != 2 or slot_ids.ndim != 1 or int(slot_ids.numel()) != int(src.shape[0]):
        return False
    if not src.is_contiguous():
        return False
    if not params.a.is_contiguous() or not params.g.is_contiguous() or not params.b.is_contiguous():
        return False
    if params.a.device != src.device or params.g.device != src.device or params.b.device != src.device:
        return False
    rank = int(params.a.shape[2])
    return 0 < rank <= 64


if tl is not None:

    @triton.jit
    def _up_gate_kernel(
        src_ptr,
        slot_ptr,
        a_ptr,
        g_ptr,
        gate_bias_ptr,
        up_gate_ptr,
        hidden_size: tl.constexpr,
        rank: tl.constexpr,
        block_h: tl.constexpr,
        block_r: tl.constexpr,
    ):
        row = tl.program_id(0)
        rank_block = tl.program_id(1)
        r_offsets = rank_block * block_r + tl.arange(0, block_r)
        r_mask = r_offsets < rank
        slot = tl.load(slot_ptr + row).to(tl.int64)
        acc_a = tl.zeros((block_r,), dtype=tl.float32)
        acc_g = tl.zeros((block_r,), dtype=tl.float32)
        for h0 in tl.range(0, hidden_size, block_h):
            h_offsets = h0 + tl.arange(0, block_h)
            h_mask = h_offsets < hidden_size
            src_vals = tl.load(src_ptr + row * hidden_size + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
            weight_offsets = slot * hidden_size * rank + h_offsets[:, None] * rank + r_offsets[None, :]
            mask = h_mask[:, None] & r_mask[None, :]
            a_vals = tl.load(a_ptr + weight_offsets, mask=mask, other=0.0).to(tl.float32)
            g_vals = tl.load(g_ptr + weight_offsets, mask=mask, other=0.0).to(tl.float32)
            acc_a += tl.sum(src_vals[:, None] * a_vals, axis=0)
            acc_g += tl.sum(src_vals[:, None] * g_vals, axis=0)
        bias = tl.load(gate_bias_ptr + slot * rank + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
        gate_pre = acc_g + bias
        gate = gate_pre / (1.0 + tl.exp(-gate_pre))
        tl.store(up_gate_ptr + row * rank + r_offsets, acc_a * gate, mask=r_mask)

    @triton.jit
    def _out_kernel(
        src_ptr,
        slot_ptr,
        b_ptr,
        up_gate_ptr,
        out_ptr,
        hidden_size: tl.constexpr,
        rank: tl.constexpr,
        block_h: tl.constexpr,
        block_r: tl.constexpr,
    ):
        row = tl.program_id(0)
        hidden_block = tl.program_id(1)
        h_offsets = hidden_block * block_h + tl.arange(0, block_h)
        h_mask = h_offsets < hidden_size
        r_offsets = tl.arange(0, block_r)
        r_mask = r_offsets < rank
        slot = tl.load(slot_ptr + row).to(tl.int64)
        up_gate = tl.load(up_gate_ptr + row * rank + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
        b_offsets = slot * rank * hidden_size + r_offsets[:, None] * hidden_size + h_offsets[None, :]
        b_vals = tl.load(b_ptr + b_offsets, mask=r_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
        acc = tl.sum(up_gate[:, None] * b_vals, axis=0)
        residual = tl.load(src_ptr + row * hidden_size + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * hidden_size + h_offsets, acc + residual, mask=h_mask)


@dataclass(frozen=True)
class TritonGroupedModelBankForwardBackend:
    name: str = "triton_grouped"

    def forward(
        self,
        *,
        slot_ids: torch.Tensor,
        src: torch.Tensor,
        params: Any,
        use_output_layernorm: bool,
    ) -> torch.Tensor:
        if not _triton_supported(slot_ids, src, params):
            return TorchModelBankForwardBackend().forward(
                slot_ids=slot_ids,
                src=src,
                params=params,
                use_output_layernorm=use_output_layernorm,
            )
        assert triton is not None
        rows = int(src.shape[0])
        hidden = int(src.shape[1])
        rank = int(params.a.shape[2])
        block_r = min(64, _next_power_of_2(rank))
        block_h_up = min(1024, _next_power_of_2(hidden))
        block_h_out = min(128, _next_power_of_2(hidden))
        up_gate = torch.empty((rows, rank), device=src.device, dtype=src.dtype)
        out = torch.empty_like(src)
        _up_gate_kernel[(rows, triton.cdiv(rank, block_r))](
            src,
            slot_ids,
            params.a,
            params.g,
            params.gate_bias,
            up_gate,
            hidden,
            rank,
            block_h_up,
            block_r,
        )
        _out_kernel[(rows, triton.cdiv(hidden, block_h_out))](
            src,
            slot_ids,
            params.b,
            up_gate,
            out,
            hidden,
            rank,
            block_h_out,
            block_r,
        )
        return _apply_output_layernorm(
            out=out,
            slot_ids=slot_ids,
            params=params,
            use_output_layernorm=use_output_layernorm,
        )


def select_model_bank_forward_backend(
    name: str | None,
    *,
    require_grad: bool,
    device: torch.device,
) -> ESampModelBankForwardBackend:
    backend = normalize_model_bank_forward_backend(name)
    if backend == "triton_grouped" and not require_grad and device.type == "cuda":
        return TritonGroupedModelBankForwardBackend()
    return TorchModelBankForwardBackend()
