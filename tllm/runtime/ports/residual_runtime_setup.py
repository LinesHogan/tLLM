#!/usr/bin/env python3
"""Generic residual runtime setup extracted from side-train hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import torch

from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.ports import residual_bindings as _residual_bindings
from tllm.runtime.ports import residual_capture_buffers as _residual_capture_buffers
from tllm.runtime.ports import residual_capture_hooks as _residual_capture_hooks
from tllm.runtime.ports.residual_bindings import ResidualPathBinding


@dataclass(frozen=True)
class ResidualRuntimeSetup:
    resolved_layers: Dict[str, torch.nn.Module]
    runtime_bindings: Dict[str, ResidualPathBinding]
    hook_spec: tuple[tuple[str, ...], str, str]
    rows: int
    compact_rows: int
    capture_full_rows: bool
    hidden_size: int
    hidden_dtype: torch.dtype
    source_resolved: str
    target_resolved: str


class ConfigLike(Protocol):
    tap_layer_paths: list[str]
    graph_scratch_rows: int


class RuntimeLike(Protocol):
    config: ConfigLike
    residual_raw_paths: dict[ResidualLocator, str]
    dispatch_plan: DispatchPlan | None
    decode_row_idx: torch.Tensor | None
    decode_valid_mask: torch.Tensor | None
    decode_prompt_idx_buf: torch.Tensor | None
    decode_sample_idx_buf: torch.Tensor | None
    decode_compact_row_idx: torch.Tensor | None
    decode_compact_count: int
    decode_compact_row_ids: tuple[int, ...]
    decode_count: int
    tap_layers: dict[str, torch.nn.Module]
    tap_scratch: dict[str, torch.Tensor]
    tap_decode_hidden: dict[str, torch.Tensor]
    tap_decode_hidden_compact: dict[str, torch.Tensor]
    capture_full_residual_rows: bool
    residual_bindings: dict[str, ResidualPathBinding]
    source_resolved_path: str
    target_resolved_path: str
    launch_consumer_from_hooks: bool


class CoreLike(Protocol):
    RUNTIME: RuntimeLike

    def _resolve_module_by_path_with_fallback(self, model: torch.nn.Module, path: str) -> tuple[torch.nn.Module, str]:
        ...

    def _infer_hidden_dtype(self, layer: torch.nn.Module) -> torch.dtype:
        ...

    def _runner_uses_compilation_or_cudagraph(self, runner: object) -> bool:
        ...


def resolve_runtime_setup(*, core: CoreLike, runner: object) -> ResidualRuntimeSetup:
    model = getattr(runner, "model")
    cfg = core.RUNTIME.config
    plan = getattr(core.RUNTIME, "dispatch_plan", None)
    required = set(plan.required_residual_layers()) if plan is not None and hasattr(plan, "required_residual_layers") else set()

    raw_paths_by_locator = _residual_bindings.raw_paths_from_runtime(core.RUNTIME)
    tap_paths = _residual_bindings.build_raw_tap_paths(
        raw_paths_by_locator=raw_paths_by_locator,
        required=required,
    )
    if not tap_paths:
        fallback: list[str] = []
        for path in cfg.tap_layer_paths:
            if path not in fallback:
                fallback.append(path)
        tap_paths = fallback

    resolved_layers: Dict[str, torch.nn.Module] = {}
    resolve_alias: Dict[str, str] = {}
    for raw_path in tap_paths:
        layer, resolved = core._resolve_module_by_path_with_fallback(model, raw_path)
        resolved_layers[resolved] = layer
        resolve_alias[raw_path] = resolved

    runtime_bindings = _residual_bindings.build_resolved_bindings(
        raw_paths_by_locator=raw_paths_by_locator,
        resolve_alias=resolve_alias,
        required=required,
    )
    source_resolved = _residual_bindings.resolved_path_for_locator(
        runtime_bindings,
        _residual_bindings.default_source_locator(),
    ) or ""
    target_resolved = _residual_bindings.resolved_path_for_locator(
        runtime_bindings,
        _residual_bindings.default_target_locator(),
    ) or ""

    hook_spec = (
        tuple(sorted(resolved_layers.keys())),
        source_resolved,
        target_resolved,
    )

    rows = int(cfg.graph_scratch_rows or 0)
    if rows <= 0:
        rows = int(getattr(runner, "max_num_reqs", 0) or 0)
    if rows <= 0:
        raise RuntimeError("Please set graph_scratch_rows > 0 (or ensure runner.max_num_reqs > 0)")
    compact_rows = 0
    compact_lane_enabled = bool(getattr(cfg, "compact_capture_lane", False))
    if compact_lane_enabled and plan is not None and hasattr(plan, "has_residual_row_compaction") and plan.has_residual_row_compaction("first_per_prompt"):
        compact_rows = int(plan.max_residual_compact_rows("first_per_prompt")) if hasattr(plan, "max_residual_compact_rows") else 0
        if compact_rows <= 0:
            compact_rows = rows
    capture_full_rows = True
    if plan is not None and hasattr(plan, "requires_full_residual_capture"):
        capture_full_rows = bool(plan.requires_full_residual_capture())
    if bool(getattr(cfg, "enable_distiller_intervention", False)):
        capture_full_rows = True
    hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
    if hidden_size <= 0:
        raise RuntimeError("Cannot infer hidden_size from model.config.hidden_size")

    any_layer = next(iter(resolved_layers.values()))
    hidden_dtype = core._infer_hidden_dtype(any_layer)

    return ResidualRuntimeSetup(
        resolved_layers=resolved_layers,
        runtime_bindings=runtime_bindings,
        hook_spec=hook_spec,
        rows=rows,
        compact_rows=compact_rows,
        capture_full_rows=bool(capture_full_rows),
        hidden_size=hidden_size,
        hidden_dtype=hidden_dtype,
        source_resolved=source_resolved,
        target_resolved=target_resolved,
    )


def apply_runtime_setup(*, core: CoreLike, runner: object, setup: ResidualRuntimeSetup) -> None:
    core.RUNTIME.decode_row_idx = torch.zeros((setup.rows,), device=getattr(runner, "device"), dtype=torch.long)
    core.RUNTIME.decode_valid_mask = torch.zeros((setup.rows, 1), device=getattr(runner, "device"), dtype=setup.hidden_dtype)
    core.RUNTIME.decode_prompt_idx_buf = torch.full(
        (setup.rows,),
        fill_value=-1,
        device=getattr(runner, "device"),
        dtype=torch.long,
    )
    core.RUNTIME.decode_sample_idx_buf = torch.full(
        (setup.rows,),
        fill_value=-1,
        device=getattr(runner, "device"),
        dtype=torch.long,
    )
    core.RUNTIME.decode_count = 0
    if int(setup.compact_rows) > 0:
        core.RUNTIME.decode_compact_row_idx = torch.zeros((int(setup.compact_rows),), device=getattr(runner, "device"), dtype=torch.long)
    else:
        core.RUNTIME.decode_compact_row_idx = None
    core.RUNTIME.decode_compact_count = 0
    core.RUNTIME.decode_compact_row_ids = ()
    core.RUNTIME.capture_full_residual_rows = bool(setup.compact_rows <= 0 or setup.capture_full_rows)

    _residual_capture_buffers.initialize_runtime_tap_buffers(
        runtime=core.RUNTIME,
        resolved_layers=setup.resolved_layers,
        device=getattr(runner, "device"),
        rows=setup.rows,
        hidden_size=setup.hidden_size,
        hidden_dtype=setup.hidden_dtype,
        compact_rows=int(setup.compact_rows),
    )
    core.RUNTIME.residual_bindings = setup.runtime_bindings
    core.RUNTIME.source_resolved_path = setup.source_resolved
    core.RUNTIME.target_resolved_path = setup.target_resolved
    core.RUNTIME.launch_consumer_from_hooks = not core._runner_uses_compilation_or_cudagraph(runner)

    _residual_capture_hooks.install_layer_forward_taps(
        core=core,
        runner=runner,
        resolved_layers=setup.resolved_layers,
    )
