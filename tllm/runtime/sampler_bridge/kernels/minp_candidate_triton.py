#!/usr/bin/env python3
"""Triton helpers for min-p candidate extraction."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - import availability depends on runtime env
    triton = None
    tl = None

_MAX_BLOCK_SIZE = 131072
_TILED_BLOCK_SIZE = 2048
_MAX_TILES = 256


def is_available() -> bool:
    return triton is not None and tl is not None


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def supports_keep_mask(logits: torch.Tensor, min_p: torch.Tensor) -> tuple[bool, str]:
    if not is_available():
        return False, "triton_unavailable"
    if logits.device.type != "cuda" or min_p.device.type != "cuda":
        return False, "non_cuda"
    if logits.ndim != 2 or min_p.ndim != 1 or int(min_p.numel()) != int(logits.shape[0]):
        return False, "bad_shape"
    block = _next_power_of_2(int(logits.shape[1]))
    if block > _MAX_BLOCK_SIZE and ((int(logits.shape[1]) + _TILED_BLOCK_SIZE - 1) // _TILED_BLOCK_SIZE) > _MAX_TILES:
        return False, "vocab_too_large"
    return True, ""


if tl is not None:

    @triton.jit
    def _minp_keep_mask_kernel(
        logits_ptr,
        min_p_ptr,
        out_ptr,
        vocab_size: tl.constexpr,
        stride_row: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < vocab_size
        vals = tl.load(logits_ptr + row * stride_row + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        row_max = tl.max(vals, axis=0)
        min_p = tl.load(min_p_ptr + row).to(tl.float32)
        min_p = tl.maximum(min_p, 1.1754943508222875e-38)
        threshold = row_max + tl.log(min_p)
        keep = (vals >= threshold) | (vals == row_max)
        tl.store(out_ptr + row * vocab_size + offsets, keep & mask, mask=mask)


    @triton.jit
    def _minp_partial_max_kernel(
        logits_ptr,
        partial_ptr,
        vocab_size: tl.constexpr,
        stride_row: tl.constexpr,
        num_tiles: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        offsets = tile * block_size + tl.arange(0, block_size)
        mask = offsets < vocab_size
        vals = tl.load(logits_ptr + row * stride_row + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        tile_max = tl.max(vals, axis=0)
        tl.store(partial_ptr + row * num_tiles + tile, tile_max)


    @triton.jit
    def _minp_keep_mask_tiled_kernel(
        logits_ptr,
        min_p_ptr,
        partial_ptr,
        out_ptr,
        vocab_size: tl.constexpr,
        stride_row: tl.constexpr,
        num_tiles: tl.constexpr,
        tile_block_size: tl.constexpr,
        reduce_block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        tile_offsets = tl.arange(0, reduce_block_size)
        tile_mask = tile_offsets < num_tiles
        partial_vals = tl.load(partial_ptr + row * num_tiles + tile_offsets, mask=tile_mask, other=-float("inf")).to(tl.float32)
        row_max = tl.max(partial_vals, axis=0)
        min_p = tl.load(min_p_ptr + row).to(tl.float32)
        min_p = tl.maximum(min_p, 1.1754943508222875e-38)
        threshold = row_max + tl.log(min_p)
        offsets = tile * tile_block_size + tl.arange(0, tile_block_size)
        mask = offsets < vocab_size
        vals = tl.load(logits_ptr + row * stride_row + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        keep = (vals >= threshold) | (vals == row_max)
        tl.store(out_ptr + row * vocab_size + offsets, keep & mask, mask=mask)


def keep_mask(logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    supported, reason = supports_keep_mask(logits, min_p)
    if not supported:
        raise NotImplementedError(reason)
    assert triton is not None
    rows = int(logits.shape[0])
    vocab = int(logits.shape[1])
    block = _next_power_of_2(vocab)
    out = torch.empty((rows, vocab), device=logits.device, dtype=torch.bool)
    if block <= _MAX_BLOCK_SIZE:
        _minp_keep_mask_kernel[(rows,)](
            logits,
            min_p,
            out,
            vocab,
            int(logits.stride(0)),
            block,
        )
        return out
    assert triton is not None
    num_tiles = triton.cdiv(vocab, _TILED_BLOCK_SIZE)
    reduce_block = _next_power_of_2(num_tiles)
    partial = torch.empty((rows, num_tiles), device=logits.device, dtype=torch.float32)
    _minp_partial_max_kernel[(rows, num_tiles)](
        logits,
        partial,
        vocab,
        int(logits.stride(0)),
        num_tiles,
        _TILED_BLOCK_SIZE,
    )
    _minp_keep_mask_tiled_kernel[(rows, num_tiles)](
        logits,
        min_p,
        partial,
        out,
        vocab,
        int(logits.stride(0)),
        num_tiles,
        _TILED_BLOCK_SIZE,
        reduce_block,
    )
    return out
