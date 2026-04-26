#!/usr/bin/env python3
"""Internal helper for installing residual capture forward wrappers."""

from __future__ import annotations

from typing import Protocol

import torch

from tllm.contracts.runtime_context import RunnerLike
from tllm.runtime import active_targets
from tllm.runtime import hidden_event_bridge as _hidden_bridge
from tllm.runtime.vllm_patch import sampler_patch as _sampler_patch


class ResidualCaptureRuntimeLike(Protocol):
    decode_row_idx: torch.Tensor | None
    decode_valid_mask: torch.Tensor | None
    decode_compact_row_idx: torch.Tensor | None
    decode_compact_count: int
    tap_decode_hidden: dict[str, torch.Tensor]
    tap_decode_hidden_compact: dict[str, torch.Tensor]
    launch_consumer_from_hooks: bool
    dispatch_plan: object | None
    consumer: object | None
    source_resolved_path: str
    target_resolved_path: str


class ResidualCaptureCoreLike(Protocol):
    RUNTIME: ResidualCaptureRuntimeLike


def install_layer_forward_taps(
    *,
    core: ResidualCaptureCoreLike,
    runner: RunnerLike,
    resolved_layers: dict[str, torch.nn.Module],
) -> None:
    for resolved_path, layer in resolved_layers.items():
        orig_forward = layer.forward

        def _forward_with_tap(*args, __orig=orig_forward, __path=resolved_path, **kwargs):
            out = __orig(*args, **kwargs)
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(tensor, torch.Tensor):
                return out
            if not active_targets.runtime_has_active_targets(core.RUNTIME):
                return out

            decode_row_idx = core.RUNTIME.decode_row_idx
            decode_valid_mask = core.RUNTIME.decode_valid_mask
            if decode_row_idx is None or decode_valid_mask is None:
                return out

            capture_full = bool(getattr(core.RUNTIME, "capture_full_residual_rows", True))
            if capture_full:
                decode_buf = core.RUNTIME.tap_decode_hidden[__path]
                if int(tensor.shape[-1]) != int(decode_buf.shape[1]):
                    raise RuntimeError(
                        "tap hidden width mismatch: "
                        f"tensor_hidden={int(tensor.shape[-1])} decode_buf_hidden={int(decode_buf.shape[1])}"
                    )

                torch.index_select(tensor, 0, decode_row_idx, out=decode_buf)
            compact_row_idx = getattr(core.RUNTIME, "decode_compact_row_idx", None)
            compact_buf = getattr(core.RUNTIME, "tap_decode_hidden_compact", {}).get(__path)
            if isinstance(compact_row_idx, torch.Tensor) and isinstance(compact_buf, torch.Tensor):
                compact_count = max(0, int(getattr(core.RUNTIME, "decode_compact_count", 0) or 0))
                if compact_count > 0:
                    if int(tensor.shape[-1]) != int(compact_buf.shape[1]):
                        raise RuntimeError(
                            "compact tap hidden width mismatch: "
                            f"tensor_hidden={int(tensor.shape[-1])} compact_buf_hidden={int(compact_buf.shape[1])}"
                        )
                    if compact_count > int(compact_row_idx.numel()) or compact_count > int(compact_buf.shape[0]):
                        raise RuntimeError(
                            "compact tap active row count exceeds compact capture capacity: "
                            f"active={compact_count} row_idx_capacity={int(compact_row_idx.numel())} "
                            f"compact_buf_capacity={int(compact_buf.shape[0])}"
                        )
                    torch.index_select(tensor, 0, compact_row_idx[:compact_count], out=compact_buf[:compact_count])
            if capture_full:
                _sampler_patch.maybe_capture_source_precompute(
                    runtime=core.RUNTIME,
                    runner=runner,
                    layer_path=__path,
                )

            _hidden_bridge.dispatch_layer_lifecycle_events(
                core=core,
                runner=runner,
                layer_path=__path,
                capture_enabled=bool(core.RUNTIME.launch_consumer_from_hooks),
            )
            return out

        layer.forward = _forward_with_tap  # type: ignore[method-assign]
