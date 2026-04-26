#!/usr/bin/env python3
"""Internal helper for residual capture buffer initialization."""

from __future__ import annotations

from typing import Protocol

import torch


class RuntimeTapBufferState(Protocol):
    tap_layers: dict[str, torch.nn.Module]
    tap_scratch: dict[str, torch.Tensor]
    tap_decode_hidden: dict[str, torch.Tensor]
    tap_decode_hidden_compact: dict[str, torch.Tensor]


def initialize_runtime_tap_buffers(
    *,
    runtime: RuntimeTapBufferState,
    resolved_layers: dict[str, torch.nn.Module],
    device: torch.device,
    rows: int,
    hidden_size: int,
    hidden_dtype: torch.dtype,
    compact_rows: int = 0,
) -> None:
    runtime.tap_layers = {}
    runtime.tap_scratch = {}
    runtime.tap_decode_hidden = {}
    runtime.tap_decode_hidden_compact = {}

    for tap_i, (resolved_path, layer) in enumerate(resolved_layers.items()):
        scratch_name = f"tllm_residual_capture_scratch_{tap_i}"
        decode_name = f"tllm_residual_capture_rows_{tap_i}"
        compact_name = f"tllm_residual_capture_compact_rows_{tap_i}"

        scratch = torch.empty((rows, hidden_size), device=device, dtype=hidden_dtype)
        decode = torch.empty((rows, hidden_size), device=device, dtype=hidden_dtype)
        if hasattr(layer, scratch_name):
            setattr(layer, scratch_name, scratch)
        else:
            layer.register_buffer(scratch_name, scratch, persistent=False)
        if hasattr(layer, decode_name):
            setattr(layer, decode_name, decode)
        else:
            layer.register_buffer(decode_name, decode, persistent=False)

        runtime.tap_layers[resolved_path] = layer
        runtime.tap_scratch[resolved_path] = getattr(layer, scratch_name)
        runtime.tap_decode_hidden[resolved_path] = getattr(layer, decode_name)
        if int(compact_rows) > 0:
            compact = torch.empty((int(compact_rows), hidden_size), device=device, dtype=hidden_dtype)
            if hasattr(layer, compact_name):
                setattr(layer, compact_name, compact)
            else:
                layer.register_buffer(compact_name, compact, persistent=False)
            runtime.tap_decode_hidden_compact[resolved_path] = getattr(layer, compact_name)
