#!/usr/bin/env python3
"""Minimal prefill-capture support for the teacher-forcing MSE repro."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import torch

from tllm.common.state import (
    STATE,
    ensure_v1_env,
    find_capture_layer,
    snapshot_step_common_from_prepare_inputs,
)
from tllm.producer.prefill import (
    export_prefill_capture,
    prepare_prefill_localization,
    stash_prefill_hidden_from_layer_output,
)
from tllm.runtime.token_map import build_token_maps_from_outputs
from tllm.runtime.vllm_patch import capture_runner as _capture_runner
from tllm.runtime.vllm_patch.adapters import get_prepare_inputs_adapter
from tllm.util import tools as _tool_helpers

ensure_v1_env()

from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # noqa: E402

_ORIG_LOAD_MODEL = GPUModelRunner.load_model
_ORIG_PREPARE_INPUTS = GPUModelRunner._prepare_inputs
_ORIG_EXECUTE_MODEL = GPUModelRunner.execute_model
_PATCH_INSTALLED = False
_PREPARE_INPUTS_ADAPTER = get_prepare_inputs_adapter()

MODEL_HOOK_FLAG = "prefill_capture_hook_installed"
MODEL_HOOK_LAYER_PATH_ATTR = "prefill_capture_hook_layer_path"


@dataclass
class _RequestMappingState:
    reqid_to_promptidx: Dict[str, int] = field(default_factory=dict)
    reqid_to_sampleidx: Dict[str, int] = field(default_factory=dict)

    def resolve_prompt_sample_for_req_id(self, req_id: str) -> tuple[int, int]:
        prompt_idx = self.reqid_to_promptidx.get(req_id)
        if prompt_idx is not None:
            return int(prompt_idx), int(self.reqid_to_sampleidx.get(req_id, 0))
        if "_" in req_id:
            maybe_sample_idx, parent_req_id = req_id.split("_", 1)
            if maybe_sample_idx.isdigit():
                parent_prompt_idx = self.reqid_to_promptidx.get(parent_req_id)
                if parent_prompt_idx is not None:
                    return int(parent_prompt_idx), int(maybe_sample_idx)
        return -1, -1


def configure_capture_runtime(
    *,
    graph_scratch_rows: int,
    capture_layer_path: str | None,
    capture_layer_index: int | None,
) -> None:
    STATE.configure(
        graph_scratch_rows=graph_scratch_rows,
        enable_prefill_producer=True,
        capture_layer_path=capture_layer_path,
        capture_layer_index=capture_layer_index,
    )


def reset_capture_runtime() -> None:
    STATE.reset_capture()
    STATE.reset_runtime_tensors()


def _ensure_capture_layer_hook(runner: Any) -> None:
    model = runner.model
    layer, layer_path = find_capture_layer(model)
    if getattr(model, MODEL_HOOK_FLAG, False):
        existing_path = getattr(model, MODEL_HOOK_LAYER_PATH_ATTR, None)
        if existing_path is not None and existing_path != layer_path:
            raise RuntimeError(
                "Model already has prefill capture hook on a different layer: "
                f"existing={existing_path} requested={layer_path}. "
                "Create a new LLM instance for a different capture layer."
            )
        return

    orig_forward = layer.forward

    def _forward_with_prefill_capture(*args: object, **kwargs: object) -> object:
        out = orig_forward(*args, **kwargs)
        tensor = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(tensor, torch.Tensor):
            stash_prefill_hidden_from_layer_output(tensor)
        return out

    layer.forward = _forward_with_prefill_capture  # type: ignore[method-assign]
    setattr(model, MODEL_HOOK_FLAG, True)
    setattr(model, MODEL_HOOK_LAYER_PATH_ATTR, layer_path)


def _wrapped_load_model(self: Any, *args: object, **kwargs: object) -> object:
    out = _ORIG_LOAD_MODEL(self, *args, **kwargs)
    _ensure_capture_layer_hook(self)
    return out


def _wrapped_prepare_inputs(self: Any, scheduler_output: object) -> object:
    _ensure_capture_layer_hook(self)
    out = _ORIG_PREPARE_INPUTS(self, scheduler_output)
    if not isinstance(out, tuple) or len(out) < 2:
        return out

    view = _PREPARE_INPUTS_ADAPTER.unpack_prepare_inputs_output(
        runner=self,
        scheduler_output=scheduler_output,
        out=out,
    )
    snapshot_step_common_from_prepare_inputs(
        runner=self,
        attn_metadata=view.attn_metadata,
        logits_indices=view.logits_indices,
        spec_decode_common=view.spec_decode_common,
        num_scheduled_tokens_np=view.num_scheduled_tokens_np,
    )
    prepare_prefill_localization(self)
    return out


def _wrapped_execute_model(self: Any, *args: object, **kwargs: object) -> object:
    _ensure_capture_layer_hook(self)
    out = _ORIG_EXECUTE_MODEL(self, *args, **kwargs)
    export_prefill_capture()
    return out


def install_runner_patch() -> None:
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return
    GPUModelRunner.load_model = _wrapped_load_model
    GPUModelRunner._prepare_inputs = _wrapped_prepare_inputs
    GPUModelRunner.execute_model = _wrapped_execute_model
    _PATCH_INSTALLED = True


def make_llm(*args: object, **kwargs: object) -> object:
    install_runner_patch()
    return _tool_helpers.make_llm(*args, **kwargs)


def run_generate_with_sample_tokens(
    *,
    llm: object,
    prompts: Sequence[str],
    params: Sequence[object],
) -> Dict[int, Dict[int, List[int]]]:
    mapping = _RequestMappingState()
    outputs = _capture_runner.run_generate_with_request_mapping(
        runtime=mapping,
        llm=llm,
        prompts=prompts,
        params=params,
    )
    _, token_ids_by_prompt_sample = build_token_maps_from_outputs(
        outputs=outputs,
        prompt_count=len(prompts),
        resolve_prompt_sample_fn=mapping.resolve_prompt_sample_for_req_id,
    )
    return token_ids_by_prompt_sample


def run_prefill_capture(
    *,
    llm: object,
    prompts: Sequence[object],
    params: Sequence[object],
) -> Dict[int, List[torch.Tensor]]:
    reset_capture_runtime()
    STATE.captured_prefill = {i: [] for i in range(len(prompts))}
    STATE.capture_active = True
    try:
        _capture_runner.run_generate_with_request_mapping(
            runtime=STATE,
            llm=llm,
            prompts=prompts,
            params=params,
            request_prompt_indices=list(range(len(prompts))),
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return STATE.captured_prefill
    finally:
        STATE.capture_active = False


build_greedy_params = _tool_helpers.build_greedy_params
read_prompts = _tool_helpers.read_prompts
shutdown_llm_instance = _tool_helpers.shutdown_llm_instance


__all__ = [
    "build_greedy_params",
    "configure_capture_runtime",
    "make_llm",
    "read_prompts",
    "reset_capture_runtime",
    "run_generate_with_sample_tokens",
    "run_prefill_capture",
    "shutdown_llm_instance",
]
