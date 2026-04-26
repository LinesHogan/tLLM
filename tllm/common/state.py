#!/usr/bin/env python3
"""Shared runtime state and helpers for vLLM v1 hidden localization."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import torch

from tllm.common.path_resolution import (
    candidate_capture_paths as _candidate_capture_paths,
    resolve_layers_container as _resolve_layers_container,
    resolve_object_by_path,
)
from tllm.common.runtime_step_state import (
    snapshot_step_common as _snapshot_step_common,
)


def ensure_v1_env() -> None:
    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


@dataclass
class RuntimeConfig:
    graph_scratch_rows: int = 0
    enable_decode_producer: bool = True
    enable_prefill_producer: bool = False
    capture_layer_path: str = "model.model.layers[0]"
    capture_layer_index: Optional[int] = None


class CaptureTapConsumer(Protocol):
    def ensure_resources(
        self,
        *,
        layer: torch.nn.Module,
        device: torch.device,
        rows: int,
        hidden_size: int,
        hidden_dtype: torch.dtype,
    ) -> None:
        ...

    def launch(
        self,
        *,
        layer: torch.nn.Module,
        decode_hidden_rows: torch.Tensor,
        source_device: torch.device,
    ) -> None:
        ...

    def synchronize(self) -> None:
        ...

    def set_enabled(self, enabled: bool) -> None:
        ...


@dataclass
class StepState:
    # Common per-step snapshot from _prepare_inputs.
    req_ids: List[str] = field(default_factory=list)
    req_id_to_index: Dict[str, int] = field(default_factory=dict)
    is_decode_req: List[bool] = field(default_factory=list)
    logits_indices: Optional[torch.Tensor] = None
    num_actual_tokens: int = 0
    num_prompt_tokens: Any = None
    num_computed_tokens: Any = None
    num_scheduled_tokens_np: Any = None

    # Decode producer outputs.
    decode_count: int = 0
    decode_prompt_idxs: List[int] = field(default_factory=list)
    decode_sample_idxs: List[int] = field(default_factory=list)

    # Prefill producer outputs.
    prefill_count: int = 0
    prefill_row_indices: List[int] = field(default_factory=list)
    prefill_prompt_idxs: List[int] = field(default_factory=list)
    prefill_sample_idxs: List[int] = field(default_factory=list)
    prefill_token_offsets: List[int] = field(default_factory=list)

    def reset_localization_outputs(self) -> None:
        self.decode_count = 0
        self.decode_prompt_idxs = []
        self.decode_sample_idxs = []

        self.prefill_count = 0
        self.prefill_row_indices = []
        self.prefill_prompt_idxs = []
        self.prefill_sample_idxs = []
        self.prefill_token_offsets = []


@dataclass
class RuntimeState:
    config: RuntimeConfig = field(default_factory=RuntimeConfig)
    step: StepState = field(default_factory=StepState)

    capture_active: bool = False
    reqid_to_promptidx: Dict[str, int] = field(default_factory=dict)

    captured_decode: Dict[int, List[torch.Tensor]] = field(default_factory=dict)
    captured_prefill: Dict[int, List[torch.Tensor]] = field(default_factory=dict)

    decode_hidden_rows: Optional[torch.Tensor] = None
    prefill_hidden_rows: Optional[torch.Tensor] = None

    tap_consumer: Optional[CaptureTapConsumer] = None

    def configure(
        self,
        graph_scratch_rows: int,
        enable_prefill_producer: bool = False,
        capture_layer_path: Optional[str] = None,
        capture_layer_index: Optional[int] = None,
    ) -> None:
        self.config.graph_scratch_rows = max(0, int(graph_scratch_rows))
        self.config.enable_prefill_producer = bool(enable_prefill_producer)

        if capture_layer_index is not None:
            self.config.capture_layer_index = int(capture_layer_index)
            self.config.capture_layer_path = ""
        elif capture_layer_path is not None:
            path = str(capture_layer_path).strip()
            if not path:
                raise RuntimeError("capture_layer_path must be non-empty when provided.")
            self.config.capture_layer_index = None
            self.config.capture_layer_path = path
        elif not self.config.capture_layer_path and self.config.capture_layer_index is None:
            self.config.capture_layer_path = "model.model.layers[0]"

        self.capture_active = False
        self.step.reset_localization_outputs()

    def reset_capture(self) -> None:
        self.capture_active = False
        self.reqid_to_promptidx = {}
        self.captured_decode = {}
        self.captured_prefill = {}

    def reset_runtime_tensors(self) -> None:
        self.decode_hidden_rows = None
        self.prefill_hidden_rows = None
        self.step.reset_localization_outputs()


STATE = RuntimeState()


def pick_common_attn_metadata(attn_metadata: Any, fallback: Any) -> Any:
    """Pick vLLM common attention metadata object across minor API variants."""
    if isinstance(attn_metadata, dict):
        for meta in attn_metadata.values():
            return meta
    if attn_metadata is not None and (
        hasattr(attn_metadata, "query_start_loc") or hasattr(attn_metadata, "num_actual_tokens")
    ):
        return attn_metadata
    if fallback is not None:
        return fallback
    return None

def resolve_prompt_sample_for_req_id(req_id: str) -> tuple[int, int]:
    """Resolve prompt/sample index from runtime request id.

    For vLLM parallel sampling with n>1:
      child_req_id = f"{sample_idx}_{parent_req_id}".
    """
    prompt_idx = STATE.reqid_to_promptidx.get(req_id)
    if prompt_idx is not None:
        return int(prompt_idx), 0

    if "_" in req_id:
        maybe_sample_idx, parent_req_id = req_id.split("_", 1)
        if maybe_sample_idx.isdigit():
            parent_prompt_idx = STATE.reqid_to_promptidx.get(parent_req_id)
            if parent_prompt_idx is not None:
                return int(parent_prompt_idx), int(maybe_sample_idx)

    return -1, -1


def find_capture_layer(model: Any) -> tuple[torch.nn.Module, str]:
    """Resolve capture layer module from runtime config."""
    cfg = STATE.config
    if cfg.capture_layer_index is not None:
        layers, layers_prefix = _resolve_layers_container(model)
        idx = int(cfg.capture_layer_index)
        if idx < 0:
            idx += len(layers)
        if idx < 0 or idx >= len(layers):
            raise RuntimeError(
                f"capture_layer_index={cfg.capture_layer_index} out of range for {layers_prefix} "
                f"(num_layers={len(layers)})."
            )
        layer = layers[idx]
        resolved_path = f"{layers_prefix}[{idx}]"
    else:
        path = cfg.capture_layer_path or "model.model.layers[0]"
        resolve_errors: List[str] = []
        layer = None
        resolved_path = ""
        for candidate in _candidate_capture_paths(path):
            try:
                obj = resolve_object_by_path(model, candidate)
            except RuntimeError as e:
                resolve_errors.append(f"{candidate}: {e}")
                continue
            layer = obj
            resolved_path = candidate
            break
        if layer is None:
            details = "; ".join(resolve_errors) if resolve_errors else "no candidate path"
            raise RuntimeError(
                f"Capture layer path `{path}` is invalid for model root "
                f"`{type(model).__name__}`. Tried candidates: {details}"
            )

    if not isinstance(layer, torch.nn.Module):
        raise RuntimeError(
            f"Resolved capture target `{resolved_path}` is not a torch.nn.Module "
            f"(got {type(layer).__name__})."
        )
    return layer, resolved_path

def set_or_register_buffer(module: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    if hasattr(module, name):
        setattr(module, name, tensor)
    else:
        module.register_buffer(name, tensor, persistent=False)


def snapshot_step_common_from_prepare_inputs(
    runner: Any,
    attn_metadata: Any,
    logits_indices: torch.Tensor,
    spec_decode_common: Any,
    num_scheduled_tokens_np: Any,
) -> None:
    """Snapshot per-step common metadata needed by all producers.

    We intentionally read this in `_prepare_inputs`, because vLLM finalizes
    packed ordering there.
    """
    common = pick_common_attn_metadata(attn_metadata, spec_decode_common)
    _snapshot_step_common(
        step=STATE.step,
        runner=runner,
        common_attn_metadata=common,
        logits_indices=logits_indices,
        num_scheduled_tokens_np=num_scheduled_tokens_np,
    )
