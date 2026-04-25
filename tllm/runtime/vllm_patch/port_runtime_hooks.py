#!/usr/bin/env python3
"""Neutral vLLM hook orchestration for port-based runtime dispatch."""

from __future__ import annotations

from typing import Dict, List, Mapping, Protocol, Sequence

import torch

from tllm.contracts.runtime_context import RunnerLike
from tllm.ports.base import ConsumerFlow
from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime import active_targets
from tllm.runtime import hidden_event_bridge as _hidden_bridge
from tllm.runtime.ports import residual_bundle_dispatch as _residual_bundle_dispatch
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.ports import residual_bindings as _residual_bindings
from tllm.runtime.ports import residual_runtime_setup as _residual_runtime_setup
from tllm.runtime.vllm_patch.adapters import (
    PrepareInputsView,
    get_prepare_inputs_adapter,
)
from tllm.runtime.vllm_patch import sampler_patch as _sampler_patch
from tllm.runtime.consumer_compat import apply_feedback as apply_consumer_feedback
from tllm.runtime.vllm_patch import common_hooks as _common_hooks

RUNTIME_EVENT_POINTS = (
    "load_model.pre",
    "load_model.post",
    "prepare_inputs.pre",
    "prepare_inputs.post",
    "layer.pre",
    "layer.post",
    "block.end",
    "stack.end",
    "execute_model.pre",
    "execute_model.post",
)

_COMPUTE_LOGITS_HOOK_FLAG = "_tllm_compute_logits_wrapped"
_INPUT_BATCH_MIN_P_PATCHED = False

def _prepare_inputs_adapter():
    return get_prepare_inputs_adapter()


class DecodeInputBatchLike(Protocol):
    req_ids: Sequence[object | None]
    num_reqs: int
    req_id_to_index: Mapping[object, int]
    num_prompt_tokens: object
    num_computed_tokens_cpu: object


class DecodeRunnerLike(RunnerLike, Protocol):
    input_batch: DecodeInputBatchLike
    _scheduler_output: object | None


class PortRuntimeConfigLike(Protocol):
    source_layer_path: str
    target_layer_path: str
    tap_layer_paths: Sequence[str]


class PortRuntimeStateLike(Protocol):
    config: PortRuntimeConfigLike
    decode_row_idx: torch.Tensor | None
    decode_valid_mask: torch.Tensor | None
    decode_count: int
    decode_prompt_idxs: list[int]
    decode_sample_idxs: list[int]
    decode_request_ids: list[str]
    decode_prompt_idx_buf: torch.Tensor | None
    decode_sample_idx_buf: torch.Tensor | None
    tap_decode_hidden: dict[str, torch.Tensor]
    residual_raw_paths: dict[ResidualLocator, str]
    residual_bindings: dict[str, ResidualPathBinding]
    source_resolved_path: str
    target_resolved_path: str
    event_step_id: int
    dispatch_plan: DispatchPlan | None


class DecodeHookCoreLike(Protocol):
    RUNTIME: PortRuntimeStateLike

    def pick_common_attn_metadata(self, attn_metadata: object, spec_decode_common: object) -> object:
        ...

    def compute_decode_localization(
        self,
        *,
        req_ids: list[object],
        is_decode_req: list[bool],
        logits_indices: torch.Tensor,
        num_actual_tokens: int,
        resolve_prompt_sample_fn: object,
    ) -> tuple[torch.Tensor, list[int], list[int], list[int]]:
        ...

    def _resolve_prompt_sample_for_req_id(self, req_id: object) -> tuple[int, int]:
        ...


class PortDispatchCoreLike(Protocol):
    RUNTIME: PortRuntimeStateLike


def prepare_decode_localization(
    *,
    core: DecodeHookCoreLike,
    runner: DecodeRunnerLike,
    out: tuple,
    prepare_inputs_view: PrepareInputsView | None = None,
) -> None:
    if len(out) < 2:
        return
    view = prepare_inputs_view
    if view is None:
        view = _prepare_inputs_adapter().unpack_prepare_inputs_output(
            runner=runner,
            scheduler_output=getattr(runner, "_scheduler_output", None),
            out=out,
        )
    if view.logits_indices is None:
        return
    attn_metadata = view.attn_metadata
    logits_indices = view.logits_indices
    spec_decode_common = view.spec_decode_common

    decode_row_idx = core.RUNTIME.decode_row_idx
    decode_valid_mask = core.RUNTIME.decode_valid_mask
    decode_prompt_idx_buf = getattr(core.RUNTIME, "decode_prompt_idx_buf", None)
    decode_sample_idx_buf = getattr(core.RUNTIME, "decode_sample_idx_buf", None)
    if decode_row_idx is None or decode_valid_mask is None or decode_prompt_idx_buf is None or decode_sample_idx_buf is None:
        raise RuntimeError(
            "Active runtime requires decode scratch buffers decode_row_idx/decode_valid_mask/decode_prompt_idx_buf/decode_sample_idx_buf"
        )

    with torch.no_grad():
        decode_valid_mask.zero_()
        decode_prompt_idx_buf.fill_(-1)
        decode_sample_idx_buf.fill_(-1)

    common = core.pick_common_attn_metadata(attn_metadata, spec_decode_common)
    num_actual_tokens = int(getattr(common, "num_actual_tokens", 0) or 0) if common else 0
    if num_actual_tokens <= 0:
        core.RUNTIME.decode_count = 0
        core.RUNTIME.decode_prompt_idxs = []
        core.RUNTIME.decode_sample_idxs = []
        core.RUNTIME.decode_request_ids = []
        core.RUNTIME.decode_prompt_idx_tensor = None
        core.RUNTIME.decode_sample_idx_tensor = None
        return

    plan = getattr(core.RUNTIME, "dispatch_plan", None)
    row_cap = 0
    if plan is not None and hasattr(plan, "max_residual_bundle_rows"):
        row_cap = int(plan.max_residual_bundle_rows())

    raw_req_ids = runner.input_batch.req_ids[: runner.input_batch.num_reqs]
    req_id_to_index = dict(runner.input_batch.req_id_to_index)
    num_prompt_tokens = runner.input_batch.num_prompt_tokens
    num_computed_tokens = runner.input_batch.num_computed_tokens_cpu
    req_ids: List[object] = []
    is_decode_req: List[bool] = []
    capped_decode_count = 0
    for req_id in raw_req_ids:
        if req_id is None:
            continue
        req_ids.append(req_id)
        req_idx = req_id_to_index.get(req_id)
        if req_idx is None:
            is_decode_req.append(False)
            continue
        is_decode = int(num_computed_tokens[req_idx]) >= int(num_prompt_tokens[req_idx])
        is_decode_req.append(is_decode)
        if is_decode and row_cap > 0:
            capped_decode_count += 1
            if capped_decode_count >= row_cap:
                break

    row_idx, prompt_idxs, sample_idxs, decode_positions = core.compute_decode_localization(
        req_ids=req_ids,
        is_decode_req=is_decode_req,
        logits_indices=logits_indices,
        num_actual_tokens=num_actual_tokens,
        resolve_prompt_sample_fn=core._resolve_prompt_sample_for_req_id,
        max_decode_rows=row_cap,
    )
    if row_idx.numel() == 0:
        core.RUNTIME.decode_count = 0
        core.RUNTIME.decode_prompt_idxs = []
        core.RUNTIME.decode_sample_idxs = []
        core.RUNTIME.decode_request_ids = []
        core.RUNTIME.decode_prompt_idx_tensor = None
        core.RUNTIME.decode_sample_idx_tensor = None
        return

    k = int(row_idx.numel())
    if k > int(decode_row_idx.numel()):
        raise RuntimeError(
            f"decode rows exceed configured scratch rows: decode_rows={k} capacity={int(decode_row_idx.numel())}. "
            "Increase graph_scratch_rows."
        )

    if row_idx.device != decode_row_idx.device:
        row_idx = row_idx.to(decode_row_idx.device)

    with torch.no_grad():
        decode_row_idx[:k].copy_(row_idx)
        decode_valid_mask[:k].fill_(1.0)
        decode_prompt_idx_buf[:k].copy_(torch.as_tensor(prompt_idxs, device=decode_row_idx.device, dtype=torch.long))
        decode_sample_idx_buf[:k].copy_(torch.as_tensor(sample_idxs, device=decode_row_idx.device, dtype=torch.long))

    core.RUNTIME.decode_count = k
    core.RUNTIME.decode_prompt_idxs = [int(x) for x in prompt_idxs]
    core.RUNTIME.decode_sample_idxs = [int(x) for x in sample_idxs]
    core.RUNTIME.decode_request_ids = [str(req_ids[pos]) for pos in decode_positions[:k]]
    core.RUNTIME.decode_prompt_idx_tensor = decode_prompt_idx_buf[:k]
    core.RUNTIME.decode_sample_idx_tensor = decode_sample_idx_buf[:k]


def build_tap_path_list(*, core: PortDispatchCoreLike) -> List[str]:
    cfg = core.RUNTIME.config
    plan = getattr(core.RUNTIME, "dispatch_plan", None)
    required = set(plan.required_residual_layers()) if plan is not None and hasattr(plan, "required_residual_layers") else set()
    raw_paths_by_locator = _residual_bindings.raw_paths_from_runtime(core.RUNTIME)
    paths = _residual_bindings.build_raw_tap_paths(
        raw_paths_by_locator=raw_paths_by_locator,
        required=required,
    )
    if paths:
        return paths

    fallback: List[str] = []
    for p in cfg.tap_layer_paths:
        if p not in fallback:
            fallback.append(p)
    if cfg.source_layer_path not in fallback:
        fallback.append(cfg.source_layer_path)
    if cfg.target_layer_path not in fallback:
        fallback.append(cfg.target_layer_path)
    return fallback


def build_decode_port_frames(*, core: PortDispatchCoreLike, layer_path: str) -> List[PortFrame]:
    return _residual_bundle_dispatch.build_decode_port_frames(core=core, layer_path=layer_path)


def build_step_scope_port_bundle(*, core: PortDispatchCoreLike, flow: ConsumerFlow) -> PortBundle | None:
    return _residual_bundle_dispatch.build_step_scope_port_bundle(core=core, flow=flow)


def dispatch_decode_port_bundles(*, core: PortDispatchCoreLike, runner: RunnerLike) -> int:
    return _residual_bundle_dispatch.dispatch_decode_port_bundles(core=core, runner=runner)


def maybe_launch_post_logits_decode_work(*, core: Any, runner: Any) -> None:
    runtime = core.RUNTIME
    step_id = int(getattr(runtime, "event_step_id", -1))
    if step_id < 0:
        return
    if not bool(getattr(runtime, "distiller_port_enabled", True)):
        return
    runtime.distiller_port_publish_attempt_count = int(getattr(runtime, "distiller_port_publish_attempt_count", 0)) + 1
    if int(getattr(runtime, "decode_post_logits_launched_step_id", -1)) == step_id:
        return
    decode_count = int(getattr(runtime, "decode_count", 0) or 0)
    if decode_count <= 0:
        return

    source_path = str(getattr(runtime, "source_resolved_path", "") or "").strip()
    if source_path:
        runtime.distiller_port_publish_step_id = step_id
        runtime.distiller_port_publish_hit_count = int(getattr(runtime, "distiller_port_publish_hit_count", 0)) + 1
        _sampler_patch.maybe_schedule_sampler_precompute(
            runtime=runtime,
            runner=runner,
            layer_path=source_path,
        )
    runtime.decode_post_logits_launched_step_id = step_id


def setup_runtime_hooks_if_active(*, core: Any, runner: Any) -> None:
    model = runner.model
    if not active_targets.runtime_has_active_targets(core.RUNTIME):
        core.RUNTIME.launch_consumer_from_hooks = False
        return

    setup = _residual_runtime_setup.resolve_runtime_setup(core=core, runner=runner)
    resolved_layers = setup.resolved_layers
    target_resolved = setup.target_resolved
    hook_spec = setup.hook_spec

    if getattr(model, core.MODEL_HOOK_FLAG, False):
        old_spec = getattr(model, core.MODEL_HOOK_SPEC_ATTR, None)
        if old_spec != hook_spec:
            raise RuntimeError(
                "Side-train hook already installed with different layer spec. "
                f"existing={old_spec} requested={hook_spec}. Create a new LLM instance."
            )
        core.RUNTIME.launch_consumer_from_hooks = not core._runner_uses_compilation_or_cudagraph(runner)
        return

    _residual_runtime_setup.apply_runtime_setup(core=core, runner=runner, setup=setup)
    consumer = getattr(core.RUNTIME, "consumer", None)
    ensure_consumer_resources = getattr(consumer, "_ensure_runtime_resources", None)
    if callable(ensure_consumer_resources):
        ensure_consumer_resources(
            _common_hooks.build_runtime_context(runner=runner, event_name="load_model.post"),
            rows_hidden=None,
        )
    _sampler_patch.ensure_sampler_precompute_buffers(runtime=core.RUNTIME, runner=runner)

    if hasattr(model, "compute_logits") and not getattr(model, _COMPUTE_LOGITS_HOOK_FLAG, False):
        orig_compute_logits = model.compute_logits

        def _wrapped_compute_logits(*args, **kwargs):
            out = orig_compute_logits(*args, **kwargs)
            if active_targets.runtime_has_active_targets(core.RUNTIME):
                maybe_launch_post_logits_decode_work(core=core, runner=runner)
            return out

        model.compute_logits = _wrapped_compute_logits
        setattr(model, _COMPUTE_LOGITS_HOOK_FLAG, True)

    setattr(model, core.MODEL_HOOK_FLAG, True)
    setattr(model, core.MODEL_HOOK_SPEC_ATTR, hook_spec)


def install_input_batch_min_p_patch() -> None:
    global _INPUT_BATCH_MIN_P_PATCHED
    try:
        from vllm.v1.worker.gpu_input_batch import InputBatch
    except Exception:
        return
    if getattr(InputBatch, "_tllm_min_p_patched", False):
        _INPUT_BATCH_MIN_P_PATCHED = True
        return
    orig_init = InputBatch.__init__
    orig_add_request = InputBatch.add_request
    orig_remove_request = InputBatch.remove_request
    orig_swap_states = InputBatch.swap_states
    orig_condense = InputBatch.condense
    orig_make_sampling_metadata = InputBatch._make_sampling_metadata

    def _init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        max_num_reqs = int(getattr(self, "max_num_reqs", 0) or len(getattr(self, "temperature_cpu", [])))
        self.min_p = torch.empty((max_num_reqs,), dtype=torch.float32, device=self.device)
        self.min_p_cpu_tensor = torch.empty(
            (max_num_reqs,),
            dtype=torch.float32,
            device="cpu",
            pin_memory=bool(getattr(self, "pin_memory", False)),
        )
        self.min_p_cpu = self.min_p_cpu_tensor.numpy()
        self.min_p_reqs = set()

    def _add_request(self, request):
        req_index = orig_add_request(self, request)
        sampling_params = getattr(request, "sampling_params", None)
        min_p = float(getattr(sampling_params, "min_p", 0.0) or 0.0) if sampling_params is not None else 0.0
        self.min_p_cpu[req_index] = min_p
        if min_p > 0.0:
            self.min_p_reqs.add(getattr(request, "req_id", getattr(request, "request_id", "")))
        return req_index

    def _remove_request(self, req_id: str):
        self.min_p_reqs.discard(req_id)
        return orig_remove_request(self, req_id)

    def _swap_states(self, i1: int, i2: int):
        out = orig_swap_states(self, i1, i2)
        self.min_p_cpu[i1], self.min_p_cpu[i2] = self.min_p_cpu[i2], self.min_p_cpu[i1]
        return out

    def _condense(self):
        out = orig_condense(self)
        num_reqs = int(getattr(self, "num_reqs", 0))
        active_req_ids = set((getattr(self, "_req_ids", []) or [])[:num_reqs])
        self.min_p_reqs.intersection_update(active_req_ids)
        return out

    def _make_sampling_metadata(self, *args, **kwargs):
        metadata = orig_make_sampling_metadata(self, *args, **kwargs)
        if not hasattr(self, "min_p_reqs") or not hasattr(self, "min_p_cpu_tensor"):
            setattr(metadata, "min_p", None)
            return metadata
        num_reqs = self.num_reqs
        if num_reqs <= 0 or not bool((self.min_p_cpu[:num_reqs] > 0.0).any()):
            setattr(metadata, "min_p", None)
        else:
            copy_slice = __import__("vllm.v1.utils", fromlist=["copy_slice"]).copy_slice
            copy_slice(self.min_p_cpu_tensor, self.min_p, num_reqs)
            setattr(metadata, "min_p", self.min_p[:num_reqs])
        return metadata

    InputBatch.__init__ = _init
    InputBatch.add_request = _add_request
    InputBatch.remove_request = _remove_request
    InputBatch.swap_states = _swap_states
    InputBatch.condense = _condense
    InputBatch._make_sampling_metadata = _make_sampling_metadata
    setattr(InputBatch, "_tllm_min_p_patched", True)
    _INPUT_BATCH_MIN_P_PATCHED = True


def wrapped_load_model(*, core: Any, runner: Any, args: tuple, kwargs: dict) -> Any:
    if not active_targets.runtime_has_active_targets(core.RUNTIME):
        return core._ORIG_LOAD_MODEL(runner, *args, **kwargs)
    _common_hooks.dispatch_runtime_event(
        runtime=core.RUNTIME,
        runner=runner,
        event_name="load_model.pre",
        phase=None,
        layer_path=None,
        capture_enabled=False,
    )
    out = core._ORIG_LOAD_MODEL(runner, *args, **kwargs)
    _sampler_patch.bind_runner_sampler(runtime=core.RUNTIME, runner=runner)
    setup_runtime_hooks_if_active(core=core, runner=runner)
    _common_hooks.dispatch_runtime_event(
        runtime=core.RUNTIME,
        runner=runner,
        event_name="load_model.post",
        phase=None,
        layer_path=None,
        capture_enabled=bool(core.RUNTIME.launch_consumer_from_hooks),
    )
    return out


def wrapped_prepare_inputs(*, core: Any, runner: Any, scheduler_output: Any) -> Any:
    if not active_targets.runtime_has_active_targets(core.RUNTIME):
        return core._ORIG_PREPARE_INPUTS(runner, scheduler_output)
    _common_hooks.dispatch_runtime_event(
        runtime=core.RUNTIME,
        runner=runner,
        event_name="prepare_inputs.pre",
        phase="decode",
        layer_path=None,
        capture_enabled=bool(core.RUNTIME.launch_consumer_from_hooks),
    )
    model_hook_flag = getattr(core, "MODEL_HOOK_FLAG", "_tllm_residual_hooks_installed")
    if getattr(getattr(runner, "model", None), model_hook_flag, False):
        core.RUNTIME.launch_consumer_from_hooks = not core._runner_uses_compilation_or_cudagraph(runner)
    else:
        setup_runtime_hooks_if_active(core=core, runner=runner)
    out = core._ORIG_PREPARE_INPUTS(runner, scheduler_output)
    core.RUNTIME.event_step_id += 1
    if isinstance(out, tuple):
        view = _prepare_inputs_adapter().unpack_prepare_inputs_output(
            runner=runner,
            scheduler_output=scheduler_output,
            out=out,
        )
        prepare_decode_localization(core=core, runner=runner, out=out, prepare_inputs_view=view)
        _sampler_patch.maybe_prepare_sampler_decode_step(runtime=core.RUNTIME, runner=runner)
        _common_hooks.dispatch_runtime_event(
            runtime=core.RUNTIME,
            runner=runner,
            event_name="prepare_inputs.post",
            phase="decode",
            layer_path=None,
            capture_enabled=bool(core.RUNTIME.launch_consumer_from_hooks),
        )
    else:
        core.RUNTIME.decode_count = 0
        core.RUNTIME.decode_prompt_idxs = []
        core.RUNTIME.decode_sample_idxs = []
        core.RUNTIME.decode_request_ids = []
        core.RUNTIME.decode_prompt_idx_tensor = None
        core.RUNTIME.decode_sample_idx_tensor = None
        _sampler_patch.maybe_prepare_sampler_decode_step(runtime=core.RUNTIME, runner=runner)
    return out


def wrapped_execute_model(*, core: Any, runner: Any, args: tuple, kwargs: dict) -> Any:
    if not active_targets.runtime_has_active_targets(core.RUNTIME):
        return core._ORIG_EXECUTE_MODEL(runner, *args, **kwargs)
    _common_hooks.dispatch_runtime_event(
        runtime=core.RUNTIME,
        runner=runner,
        event_name="execute_model.pre",
        phase="decode",
        layer_path=None,
        capture_enabled=bool(core.RUNTIME.launch_consumer_from_hooks),
    )
    model_hook_flag = getattr(core, "MODEL_HOOK_FLAG", "_tllm_residual_hooks_installed")
    if getattr(getattr(runner, "model", None), model_hook_flag, False):
        core.RUNTIME.launch_consumer_from_hooks = not core._runner_uses_compilation_or_cudagraph(runner)
    else:
        setup_runtime_hooks_if_active(core=core, runner=runner)
    out = core._ORIG_EXECUTE_MODEL(runner, *args, **kwargs)
    if not bool(getattr(getattr(runner, "model", None), _COMPUTE_LOGITS_HOOK_FLAG, False)):
        maybe_launch_post_logits_decode_work(core=core, runner=runner)
    dispatch_decode_port_bundles(core=core, runner=runner)
    if not core.RUNTIME.launch_consumer_from_hooks:
        _hidden_bridge.dispatch_deferred_layer_batches(core=core, runner=runner)
    _common_hooks.dispatch_runtime_event(
        runtime=core.RUNTIME,
        runner=runner,
        event_name="execute_model.post",
        phase="decode",
        layer_path=None,
        capture_enabled=bool(core.RUNTIME.launch_consumer_from_hooks),
    )
    return out


def install_runner_patch(*, core: Any) -> None:
    if core._PATCH_INSTALLED:
        return
    install_input_batch_min_p_patch()
    _sampler_patch.install_sampler_patch(core=core)
    core.GPUModelRunner.load_model = core._wrapped_load_model
    core.GPUModelRunner._prepare_inputs = core._wrapped_prepare_inputs
    core.GPUModelRunner.execute_model = core._wrapped_execute_model
    core._PATCH_INSTALLED = True
