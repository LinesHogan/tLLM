#!/usr/bin/env python3
"""Prefill producer: eager-first prefill row localization and export."""

from __future__ import annotations

from typing import Any

import torch

from tllm.common.state import (
    STATE,
    resolve_prompt_sample_for_req_id,
)


def prepare_prefill_localization(_runner: Any) -> None:
    """Plan prefill rows in current packed step.

    We follow vLLM mixed prefill/decode semantics per request:
      scheduled = num_scheduled_tokens[r]
      computed  = num_computed_tokens[r]
      prompt    = num_prompt_tokens[r]
      prefill_len = clamp(prompt - computed, 0, scheduled)

    Row layout in packed hidden is request-contiguous in current req order.
    If request r starts at row_base and has `scheduled` rows, then:
      prefill rows for r are [row_base, row_base + prefill_len).

    This producer is eager-first. It is intentionally decoupled from graph-safe
    decode path so prefill experiments can iterate without touching decode logic.
    """
    step = STATE.step
    STATE.prefill_hidden_rows = None
    step.prefill_count = 0
    step.prefill_row_indices = []
    step.prefill_prompt_idxs = []
    step.prefill_sample_idxs = []
    step.prefill_token_offsets = []

    if not STATE.config.enable_prefill_producer:
        return

    num_scheduled_tokens_np = step.num_scheduled_tokens_np
    if num_scheduled_tokens_np is None:
        return

    row_base = 0
    for req_pos, req_id in enumerate(step.req_ids):
        if req_pos >= len(num_scheduled_tokens_np):
            break

        scheduled = int(num_scheduled_tokens_np[req_pos])
        if scheduled <= 0:
            continue

        req_idx = step.req_id_to_index.get(req_id)
        if req_idx is None:
            row_base += scheduled
            continue

        prompt_len = int(step.num_prompt_tokens[req_idx])
        computed = int(step.num_computed_tokens[req_idx])
        prefill_len = max(0, min(scheduled, prompt_len - computed))

        if prefill_len > 0:
            prompt_idx, sample_idx = resolve_prompt_sample_for_req_id(req_id)
            for i in range(prefill_len):
                step.prefill_row_indices.append(row_base + i)
                step.prefill_prompt_idxs.append(prompt_idx)
                step.prefill_sample_idxs.append(sample_idx)
                step.prefill_token_offsets.append(computed + i)

        row_base += scheduled

    step.prefill_count = len(step.prefill_row_indices)


def stash_prefill_hidden_from_layer_output(layer_out: torch.Tensor) -> None:
    """Stash prefill-selected hidden rows from current capture-layer output.

    This function is called from the capture-layer forward wrapper. For prefill we
    use dynamic row-index tensors (eager-first), not static graph buffers in decoding.
    """
    STATE.prefill_hidden_rows = None

    if not STATE.config.enable_prefill_producer:
        return

    step = STATE.step
    k = int(step.prefill_count)
    if k <= 0:
        return

    if layer_out.shape[0] <= 0:
        return

    row_idx = torch.tensor(step.prefill_row_indices, device=layer_out.device, dtype=torch.long)
    if row_idx.numel() == 0:
        return

    # min_idx = int(row_idx.min().item())
    # max_idx = int(row_idx.max().item())
    # if min_idx < 0 or max_idx >= int(layer_out.shape[0]):
    #     raise RuntimeError(
    #         f"prefill row index out of bounds: min={min_idx} max={max_idx} layer_rows={int(layer_out.shape[0])}"
    #     )

    selected = layer_out.index_select(0, row_idx)
    if selected.shape[0] != k:
        raise RuntimeError(
            f"prefill stash count mismatch: selected={int(selected.shape[0])} planned={k}"
        )
    STATE.prefill_hidden_rows = selected


def export_prefill_capture() -> None:
    """Export prefill hidden rows to capture storage (CPU)."""
    if not STATE.capture_active:
        return
    if not STATE.config.enable_prefill_producer:
        return

    selected = STATE.prefill_hidden_rows
    if selected is None:
        return

    step = STATE.step
    k = int(step.prefill_count)
    if k <= 0:
        return

    if k > len(step.prefill_prompt_idxs):
        raise RuntimeError(
            "Prefill export metadata mismatch: "
            f"count={k} prompt_idxs={len(step.prefill_prompt_idxs)}"
        )

    if k > int(selected.shape[0]):
        raise RuntimeError(
            "Prefill export buffer mismatch: "
            f"count={k} selected_rows={int(selected.shape[0])}"
        )

    # if not bool(torch.isfinite(selected[:k]).all().item()):
    #     raise RuntimeError("Found non-finite hidden in localized prefill rows")

    for i in range(k):
        prompt_idx = int(step.prefill_prompt_idxs[i])
        if prompt_idx < 0:
            continue
        if prompt_idx not in STATE.captured_prefill:
            STATE.captured_prefill[prompt_idx] = []
        STATE.captured_prefill[prompt_idx].append(
            selected[i].detach().to(device="cpu")
        )

    STATE.prefill_hidden_rows = None
