#!/usr/bin/env python3
"""Internal bridge for HiddenBatch event dispatch."""

from __future__ import annotations

from typing import Protocol

import torch

from tllm.contracts.hidden_batch import HiddenBatch
from tllm.contracts.runtime_context import RunnerLike
from tllm.runtime.decode_runtime_metadata import active_request_prompt_sample_metadata
from tllm.runtime.ports import residual_bindings as _residual_bindings
from tllm.runtime.vllm_patch import common_hooks as _common_hooks


class ResidualRuntimeLike(Protocol):
    tap_decode_hidden: dict[str, torch.Tensor]
    decode_row_idx: torch.Tensor | None
    decode_valid_mask: torch.Tensor | None
    decode_count: int
    decode_prompt_idxs: list[int]
    decode_sample_idxs: list[int]
    decode_prompt_idx_tensor: torch.Tensor | None
    decode_sample_idx_tensor: torch.Tensor | None
    source_resolved_path: str
    target_resolved_path: str
    event_step_id: int


class EventCoreLike(Protocol):
    RUNTIME: ResidualRuntimeLike


def build_runtime_hidden_batch(*, core: EventCoreLike, layer_path: str) -> HiddenBatch | None:
    decode_buf = core.RUNTIME.tap_decode_hidden.get(layer_path)
    decode_row_idx = core.RUNTIME.decode_row_idx
    decode_valid_mask = core.RUNTIME.decode_valid_mask
    if not isinstance(decode_buf, torch.Tensor):
        return None
    if decode_row_idx is None or decode_valid_mask is None:
        return None

    active = int(core.RUNTIME.decode_count)
    if active <= 0:
        return None
    if active > int(decode_row_idx.numel()):
        active = int(decode_row_idx.numel())

    _, prompt_idx, sample_idx = active_request_prompt_sample_metadata(core.RUNTIME, active)
    prompt_tensor = getattr(core.RUNTIME, "decode_prompt_idx_tensor", None)
    sample_tensor = getattr(core.RUNTIME, "decode_sample_idx_tensor", None)
    if isinstance(prompt_tensor, torch.Tensor) and int(prompt_tensor.numel()) >= active:
        prompt_idx_tensor = prompt_tensor[:active].to(device=decode_buf.device, dtype=torch.long)
    else:
        prompt_idx_tensor = torch.tensor(prompt_idx, device=decode_buf.device, dtype=torch.long)
    if isinstance(sample_tensor, torch.Tensor) and int(sample_tensor.numel()) >= active:
        sample_idx_tensor = sample_tensor[:active].to(device=decode_buf.device, dtype=torch.long)
    else:
        sample_idx_tensor = torch.tensor(sample_idx, device=decode_buf.device, dtype=torch.long)

    return HiddenBatch(
        step_id=int(core.RUNTIME.event_step_id),
        phase="decode",
        layer_path=str(layer_path),
        rows_hidden=decode_buf[:active],
        row_idx=decode_row_idx[:active],
        valid_mask=decode_valid_mask[:active, 0],
        prompt_idx=prompt_idx_tensor,
        sample_idx=sample_idx_tensor,
        metadata={
            "prompt_idxs": list(prompt_idx),
            "sample_idxs": list(sample_idx),
            "full_rows_hidden": decode_buf,
        },
    )


def dispatch_deferred_layer_batches(*, core: EventCoreLike, runner: RunnerLike) -> int:
    dispatched = 0
    seen: set[str] = set()
    source_path, target_path = _residual_bindings.default_resolved_paths(core.RUNTIME)
    for layer_path in (source_path, target_path):
        layer_key = str(layer_path).strip()
        if not layer_key or layer_key in seen:
            continue
        seen.add(layer_key)
        plan = getattr(core.RUNTIME, "dispatch_plan", None)
        if plan is not None:
            targets = plan.select(
                event_name="layer.post",
                phase="decode",
                layer_path=layer_key,
                capture_enabled=False,
            )
            if not targets:
                continue
            batch = build_runtime_hidden_batch(core=core, layer_path=layer_key)
            if batch is None:
                continue
            dispatched += _common_hooks.dispatch_runtime_event(
                runtime=core.RUNTIME,
                runner=runner,
                event_name="layer.post",
                phase="decode",
                layer_path=layer_key,
                capture_enabled=False,
                batch=batch,
            )
            continue
        dispatched += _common_hooks.dispatch_runtime_event(
            runtime=core.RUNTIME,
            runner=runner,
            event_name="layer.post",
            phase="decode",
            layer_path=layer_key,
            capture_enabled=False,
            batch_factory=lambda layer_key=layer_key: build_runtime_hidden_batch(core=core, layer_path=layer_key),
        )
    return dispatched


def dispatch_layer_lifecycle_events(
    *,
    core: EventCoreLike,
    runner: RunnerLike,
    layer_path: str,
    capture_enabled: bool,
) -> None:
    _common_hooks.dispatch_runtime_event(
        runtime=core.RUNTIME,
        runner=runner,
        event_name="layer.pre",
        phase="decode",
        layer_path=layer_path,
        capture_enabled=bool(capture_enabled),
    )
    _common_hooks.dispatch_runtime_event(
        runtime=core.RUNTIME,
        runner=runner,
        event_name="layer.post",
        phase="decode",
        layer_path=layer_path,
        capture_enabled=bool(capture_enabled),
        batch_factory=lambda: build_runtime_hidden_batch(core=core, layer_path=layer_path),
    )
    _common_hooks.dispatch_runtime_event(
        runtime=core.RUNTIME,
        runner=runner,
        event_name="block.end",
        phase="decode",
        layer_path=layer_path,
        capture_enabled=bool(capture_enabled),
    )
    _, target_path = _residual_bindings.default_resolved_paths(core.RUNTIME)
    if layer_path == target_path:
        _common_hooks.dispatch_runtime_event(
            runtime=core.RUNTIME,
            runner=runner,
            event_name="stack.end",
            phase="decode",
            layer_path=layer_path,
            capture_enabled=bool(capture_enabled),
        )
