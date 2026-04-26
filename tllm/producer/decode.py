#!/usr/bin/env python3
"""Decode producer: localize decode rows and export capture-layer hidden."""

from __future__ import annotations

from typing import Any, Callable, List, Tuple

import torch

from tllm.common.state import (
    STATE,
    find_capture_layer,
    resolve_prompt_sample_for_req_id,
    set_or_register_buffer,
)

DECODE_ROW_IDX_BUFFER = "decode_row_idx"
DECODE_VALID_MASK_BUFFER = "decode_valid_mask"
DECODE_HIDDEN_ROWS_BUFFER = "decode_hidden_rows_buffer"


def compute_decode_localization(
    req_ids: List[str],
    is_decode_req: List[bool],
    logits_indices: torch.Tensor | None,
    num_actual_tokens: int,
    resolve_prompt_sample_fn: Callable[[str], Tuple[int, int]],
    max_decode_rows: int = 0,
) -> Tuple[torch.Tensor, List[int], List[int], List[int]]:
    """Pure decode-localization function (no model/runner dependency).

    Returns:
    - row_idx: packed hidden row index for each decode request in current step.
    - prompt_idxs: prompt index per localized row.
    - sample_idxs: sample index per localized row (n>1 aware).
    - decode_positions: request positions selected from current req order.
    """
    if num_actual_tokens <= 0 or logits_indices is None or logits_indices.numel() == 0:
        return torch.empty((0,), dtype=torch.long), [], [], []

    decode_positions = [
        i
        for i, is_decode in enumerate(is_decode_req)
        if is_decode and i < int(logits_indices.numel()) and i < len(req_ids)
    ]
    row_cap = int(max_decode_rows)
    if row_cap > 0:
        decode_positions = decode_positions[:row_cap]
    if not decode_positions:
        return torch.empty((0,), device=logits_indices.device, dtype=torch.long), [], [], []

    contiguous = decode_positions == list(range(int(decode_positions[0]), int(decode_positions[0]) + len(decode_positions)))
    if contiguous:
        row_idx = logits_indices.narrow(0, int(decode_positions[0]), len(decode_positions))
    else:
        decode_pos = torch.tensor(decode_positions, device=logits_indices.device, dtype=torch.long)
        row_idx = logits_indices.index_select(0, decode_pos)

    # Keep the explicit bounds diagnostic on CPU, but avoid forcing a host
    # sync on CUDA decode hot paths.
    if row_idx.device.type != "cuda":
        min_idx = int(row_idx.min().item())
        max_idx = int(row_idx.max().item())
        if min_idx < 0 or max_idx >= int(num_actual_tokens):
            raise RuntimeError(
                f"logits_indices out of range in prepare_inputs: min={min_idx} "
                f"max={max_idx} num_actual_tokens={int(num_actual_tokens)}"
            )

    prompt_idxs: List[int] = []
    sample_idxs: List[int] = []
    for req_pos in decode_positions:
        req_id = req_ids[req_pos]
        prompt_idx, sample_idx = resolve_prompt_sample_fn(req_id)
        prompt_idxs.append(int(prompt_idx))
        sample_idxs.append(int(sample_idx))

    return row_idx, prompt_idxs, sample_idxs, decode_positions


def ensure_decode_buffers(
    layer: torch.nn.Module,
    device: torch.device,
    rows: int,
    hidden_size: int,
    hidden_dtype: torch.dtype,
) -> None:
    """Ensure fixed decode-localization buffers exist on capture layer.

    These buffers are static across steps and reused, so graph replay does not
    allocate/free per-step tensors.
    """
    decode_row_idx = torch.zeros((rows,), device=device, dtype=torch.long)
    set_or_register_buffer(layer, DECODE_ROW_IDX_BUFFER, decode_row_idx)

    # Keep mask and decode buffer in activation dtype to avoid implicit casts.
    decode_valid_mask = torch.zeros((rows, 1), device=device, dtype=hidden_dtype)
    set_or_register_buffer(layer, DECODE_VALID_MASK_BUFFER, decode_valid_mask)

    decode_hidden_rows_buffer = torch.empty((rows, hidden_size), device=device, dtype=hidden_dtype)
    set_or_register_buffer(layer, DECODE_HIDDEN_ROWS_BUFFER, decode_hidden_rows_buffer)
    STATE.decode_hidden_rows = getattr(layer, DECODE_HIDDEN_ROWS_BUFFER)


def prepare_decode_localization(runner: Any) -> None:
    """Compute decode row indices from vLLM packed layout.

    Why this is needed:
    - vLLM packs tokens for all active requests into one dense hidden tensor.
    - `logits_indices[i]` points to the row used for request `i` sampling.
    - only requests in decode phase should be consumed by decode producer.

    Example:
    - req_ids        = [A, B, C, D]
    - is_decode_req  = [F, T, T, F]
    - logits_indices = [4, 7, 11, 15]
    -> decode_positions = [1, 2]
    -> decode rows      = [7, 11]

    The rows above are written into fixed GPU buffers so capture-layer forward can
    run a graph-safe gather/mask sequence.
    """
    step = STATE.step

    layer, _ = find_capture_layer(runner.model)
    decode_row_idx = getattr(layer, DECODE_ROW_IDX_BUFFER)
    decode_valid_mask = getattr(layer, DECODE_VALID_MASK_BUFFER)

    step.decode_count = 0
    step.decode_prompt_idxs = []
    step.decode_sample_idxs = []

    with torch.no_grad():
        decode_row_idx.zero_()
        decode_valid_mask.zero_()

    row_idx, prompt_idxs, sample_idxs, _decode_positions = compute_decode_localization(
        req_ids=step.req_ids,
        is_decode_req=step.is_decode_req,
        logits_indices=step.logits_indices,
        num_actual_tokens=step.num_actual_tokens,
        resolve_prompt_sample_fn=resolve_prompt_sample_for_req_id,
    )
    if row_idx.numel() == 0:
        return

    if row_idx.device != decode_row_idx.device:
        row_idx = row_idx.to(decode_row_idx.device)

    k = int(row_idx.numel())
    if k > int(decode_row_idx.numel()):
        raise RuntimeError(
            f"decode rows exceed graph scratch rows: decode_rows={k} "
            f"capacity={int(decode_row_idx.numel())}; increase graph_scratch_rows"
        )

    with torch.no_grad():
        decode_row_idx[:k].copy_(row_idx)
        decode_valid_mask[:k].fill_(1.0)

    step.decode_count = k
    step.decode_prompt_idxs = [int(x) for x in prompt_idxs]
    step.decode_sample_idxs = [int(x) for x in sample_idxs]


def gather_decode_hidden_from_scratch(layer: torch.nn.Module, scratch: torch.Tensor) -> torch.Tensor:
    """Run decode gather from scratch with precomputed row_idx/mask buffers."""
    decode_row_idx = getattr(layer, DECODE_ROW_IDX_BUFFER)
    decode_valid_mask = getattr(layer, DECODE_VALID_MASK_BUFFER)
    decode_hidden_rows_buffer = getattr(layer, DECODE_HIDDEN_ROWS_BUFFER)

    # Note: scratch.index_select creates a new tensor, then copy_ writes into
    # decode_hidden_rows_buffer. In CUDA Graph, write directly into fixed memory.
    torch.index_select(scratch, 0, decode_row_idx, out=decode_hidden_rows_buffer)
    decode_hidden_rows_buffer.mul_(decode_valid_mask)
    return decode_hidden_rows_buffer


def export_decode_capture() -> None:
    """Export localized decode hidden rows to capture storage (CPU)."""
    if not STATE.capture_active:
        return

    decode_hidden_rows = STATE.decode_hidden_rows
    if decode_hidden_rows is None:
        return

    step = STATE.step
    requested = int(step.decode_count)

    if requested < 0:
        raise RuntimeError(f"Invalid decode_count={requested}")
    if requested > len(step.decode_prompt_idxs):
        raise RuntimeError(
            "Decode export metadata mismatch: "
            f"count={requested} prompt_idxs={len(step.decode_prompt_idxs)}"
        )
    if requested > len(step.decode_sample_idxs):
        raise RuntimeError(
            "Decode export metadata mismatch: "
            f"count={requested} sample_idxs={len(step.decode_sample_idxs)}"
        )
    if requested > int(decode_hidden_rows.shape[0]):
        raise RuntimeError(
            "Decode export buffer mismatch: "
            f"count={requested} decode_hidden_rows={int(decode_hidden_rows.shape[0])}"
        )

    if requested <= 0:
        return

    selected = decode_hidden_rows[:requested]
    # if not bool(torch.isfinite(selected).all().item()):
    #     raise RuntimeError("Found non-finite hidden in localized decode rows")

    for local_i in range(requested):
        prompt_idx = int(step.decode_prompt_idxs[local_i])
        if prompt_idx < 0:
            continue
        if prompt_idx not in STATE.captured_decode:
            STATE.captured_decode[prompt_idx] = []
        STATE.captured_decode[prompt_idx].append(
            selected[local_i].detach().to(device="cpu")
        )
