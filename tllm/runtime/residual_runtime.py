#!/usr/bin/env python3
"""Generic residual-stream runtime host for consumer-driven workflows."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch

from tllm.common.state import ensure_v1_env, pick_common_attn_metadata, resolve_object_by_path
from tllm.consumers.base import BaseConsumer
from tllm.consumers.esamp.config import AdaptationStreamMode, normalize_adaptation_stream_mode
from tllm.consumers.esamp.initializers.svd import SVDModelBankInitializerConfig
from tllm.consumers.esamp.model_bank_backend import ModelBankForwardBackendName, normalize_model_bank_forward_backend
from tllm.consumers.esamp.engine import ESampStats
from tllm.contracts.runtime_context import RunnerLike
from tllm.ports.residual_stream import ResidualLocator
from tllm.producer.decode import compute_decode_localization
from tllm.runtime.sampler_bridge.types import SamplerBackend, normalize_sampler_backend
from tllm.runtime.consumer_registry import ConsumerRegistry
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.consumer_compat import synchronize as synchronize_consumer
from tllm.runtime.ports.residual_bindings import (
    ResidualPathBinding,
    default_raw_paths_from_config,
)
from tllm.runtime.vllm_patch import port_runtime_hooks as _hooks

ensure_v1_env()

GPUModelRunner = None
_ORIG_LOAD_MODEL = None
_ORIG_PREPARE_INPUTS = None
_ORIG_EXECUTE_MODEL = None
_PATCH_INSTALLED = False

MODEL_HOOK_FLAG = "esamp_hook_installed"
MODEL_HOOK_SPEC_ATTR = "esamp_hook_spec"

RuntimeConsumer = BaseConsumer


@dataclass
class DistillerTimingStats:
    port_publish_attempt_count: int = 0
    port_publish_hit_count: int = 0
    precompute_ms_total: float = 0.0
    precompute_count: int = 0
    wait_ms_total: float = 0.0
    wait_count: int = 0
    schedule_attempt_count: int = 0
    schedule_hit_count: int = 0
    fallback_ms_total: float = 0.0
    fallback_count: int = 0
    candidate_sample_count: int = 0
    candidate_token_count: int = 0
    candidate_row_count: int = 0
    candidate_max_count: int = 0
    candidate_kernel_triton_count: int = 0
    candidate_kernel_torch_count: int = 0
    candidate_kernel_fallback_count: int = 0

    @property
    def precompute_ms_avg(self) -> float:
        return float(self.precompute_ms_total / self.precompute_count) if self.precompute_count > 0 else 0.0

    @property
    def wait_ms_avg(self) -> float:
        return float(self.wait_ms_total / self.wait_count) if self.wait_count > 0 else 0.0

    @property
    def fallback_ms_avg(self) -> float:
        return float(self.fallback_ms_total / self.fallback_count) if self.fallback_count > 0 else 0.0


@dataclass
class GraphDebugStats:
    capture_state: str = "uncaptured"
    capture_attempt_count: int = 0
    skip_not_enabled_count: int = 0
    skip_missing_optimizer_state_count: int = 0
    skip_wrong_device_count: int = 0
    replay_attempt_count: int = 0
    replay_hit_count: int = 0
    replay_stage_miss_count: int = 0
    kernel_fallback_count: int = 0
    disable_reason: str = ""


@dataclass
class PathHotspotStats:
    cpu_ms_total: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class SamplerPrecomputeBuffers:
    pred_hidden: torch.Tensor
    valid_mask: torch.Tensor
    source_hidden: torch.Tensor
    prompt_idx: torch.Tensor
    all_row_ids: torch.Tensor
    dense_logits: torch.Tensor | None = None

    def covers(self, *, rows: int, hidden: int, device: torch.device, dtype: torch.dtype) -> bool:
        return (
            tuple(self.pred_hidden.shape) == (rows, hidden)
            and self.pred_hidden.device == device
            and self.pred_hidden.dtype == dtype
            and int(self.valid_mask.numel()) == rows
            and self.valid_mask.device == device
            and tuple(self.source_hidden.shape) == (rows, hidden)
            and self.source_hidden.device == device
            and self.source_hidden.dtype == dtype
            and int(self.prompt_idx.numel()) == rows
            and self.prompt_idx.device == device
            and int(self.all_row_ids.numel()) == rows
            and self.all_row_ids.device == device
        )

    def dense_covers(self, *, rows: int, vocab: int, device: torch.device) -> bool:
        return (
            self.dense_logits is not None
            and tuple(self.dense_logits.shape) == (rows, vocab)
            and self.dense_logits.device == device
            and self.dense_logits.dtype == torch.float32
        )


@dataclass
class SamplerPrecomputeCache:
    step_id: int
    row_ids: torch.Tensor
    pred_hidden: torch.Tensor
    dense_logits: torch.Tensor | None = None
    pred_hidden_row_map: torch.Tensor | None = None
    all_rows: bool = False


@dataclass
class SamplerPrecomputeState:
    stream: torch.cuda.Stream | None = None
    event: torch.cuda.Event | None = None
    buffers: SamplerPrecomputeBuffers | None = None
    cache: SamplerPrecomputeCache | None = None
    source_enabled: bool = False
    source_capture_step_id: int = -1
    port_enabled: bool = True
    port_capture_step_id: int = -1
    port_publish_step_id: int = -1
    port_consume_step_id: int = -1
    precomputed_step_id: int = -1
    all_rows: bool = False
    timing_enabled: bool = False
    precompute_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)
    wait_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)
    fallback_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)
    port_publish_attempt_count: int = 0
    port_publish_hit_count: int = 0
    schedule_attempt_count: int = 0
    schedule_hit_count: int = 0
    candidate_sample_count: int = 0
    candidate_token_count: int = 0
    candidate_row_count: int = 0
    candidate_max_count: int = 0
    candidate_kernel_triton_count: int = 0
    candidate_kernel_torch_count: int = 0
    candidate_kernel_fallback_count: int = 0

    def reset_step(self) -> None:
        self.source_enabled = False
        self.source_capture_step_id = -1
        self.port_capture_step_id = -1
        self.port_publish_step_id = -1
        self.port_consume_step_id = -1
        self.precomputed_step_id = -1
        self.cache = None
        self.all_rows = False

    def reset_decode_step(self, step_id: int) -> None:
        self.source_enabled = False
        self.source_capture_step_id = -1
        self.port_capture_step_id = -1
        self.port_publish_step_id = -1
        self.port_consume_step_id = -1
        self.precomputed_step_id = int(step_id)
        self.cache = None
        self.all_rows = False
        if self.buffers is not None:
            self.buffers.valid_mask.zero_()

    def ensure_buffers(
        self,
        *,
        source_hidden: torch.Tensor,
        dense_vocab: int | None = None,
    ) -> SamplerPrecomputeBuffers:
        rows = int(source_hidden.shape[0])
        hidden = int(source_hidden.shape[1])
        device = source_hidden.device
        dtype = source_hidden.dtype
        if self.buffers is None or not self.buffers.covers(rows=rows, hidden=hidden, device=device, dtype=dtype):
            self.buffers = SamplerPrecomputeBuffers(
                pred_hidden=torch.empty((rows, hidden), device=device, dtype=dtype),
                valid_mask=torch.zeros((rows,), device=device, dtype=torch.bool),
                source_hidden=torch.empty((rows, hidden), device=device, dtype=dtype),
                prompt_idx=torch.empty((rows,), device=device, dtype=torch.long),
                all_row_ids=torch.arange(rows, device=device, dtype=torch.long),
            )
        if dense_vocab is not None and not self.buffers.dense_covers(rows=rows, vocab=int(dense_vocab), device=device):
            self.buffers.dense_logits = torch.empty((rows, int(dense_vocab)), device=device, dtype=torch.float32)
        return self.buffers

    def cache_for_step(self, step_id: int) -> SamplerPrecomputeCache | None:
        if self.cache is None or int(self.cache.step_id) != int(step_id):
            return None
        return self.cache

    def store_cache(
        self,
        *,
        step_id: int,
        row_ids: torch.Tensor,
        pred_hidden: torch.Tensor,
        dense_logits: torch.Tensor | None = None,
        pred_hidden_row_map: torch.Tensor | None = None,
        all_rows: bool = False,
    ) -> SamplerPrecomputeCache:
        self.precomputed_step_id = int(step_id)
        self.cache = SamplerPrecomputeCache(
            step_id=int(step_id),
            row_ids=row_ids,
            pred_hidden=pred_hidden,
            dense_logits=dense_logits,
            pred_hidden_row_map=pred_hidden_row_map,
            all_rows=bool(all_rows),
        )
        self.all_rows = bool(all_rows)
        return self.cache

    def captured_rows_for_step(
        self,
        *,
        step_id: int,
        decode_count: int,
    ) -> SamplerPrecomputeCache | None:
        buffers = self.buffers
        count = int(decode_count)
        if (
            buffers is None
            or count <= 0
            or int(self.source_capture_step_id) != int(step_id)
            or int(buffers.pred_hidden.shape[0]) < count
            or int(buffers.valid_mask.numel()) < count
        ):
            return None
        dense_logits = buffers.dense_logits[:count] if buffers.dense_logits is not None and int(buffers.dense_logits.shape[0]) >= count else None
        if self.all_rows:
            row_ids = buffers.all_row_ids[:count]
            return SamplerPrecomputeCache(
                step_id=int(step_id),
                row_ids=row_ids,
                pred_hidden=buffers.pred_hidden[:count],
                dense_logits=dense_logits,
                all_rows=True,
            )
        row_ids = buffers.valid_mask[:count].nonzero(as_tuple=False).view(-1)
        if int(row_ids.numel()) <= 0:
            return None
        selected_dense = dense_logits.index_select(0, row_ids) if dense_logits is not None else None
        return SamplerPrecomputeCache(
            step_id=int(step_id),
            row_ids=row_ids,
            pred_hidden=buffers.pred_hidden[:count].index_select(0, row_ids),
            dense_logits=selected_dense,
            all_rows=False,
        )


@dataclass
class ResidualRuntimeConfig:
    graph_scratch_rows: int = 0
    tap_layer_paths: list[str] = field(default_factory=list)
    source_layer_path: str = "model.model.layers[0].input_layernorm"
    target_layer_path: str = "model.model.layers[-1].input_layernorm"
    enable_esamp_training: bool = True
    distiller_hidden_dim: int = 128
    distiller_lr: float = 1e-3
    per_request_models: bool = False
    per_request_model_bank: bool = False
    model_bank_slots: int = 0
    model_bank_flush_interval: int = 1
    model_bank_rank: int = 64
    model_bank_use_output_layernorm: bool = True
    model_bank_initializer: SVDModelBankInitializerConfig | None = None
    model_bank_train_cudagraph: bool = False
    model_bank_forward_backend: ModelBankForwardBackendName = "torch"
    adaptation_pipeline_slots: int = 4
    adaptation_stream_mode: AdaptationStreamMode = "dual"
    adaptation_stream_priority: int = 0
    trace_per_request_losses: bool = False
    trace_interval: int = 1
    trace_max_points: int = 0
    enable_distiller_intervention: bool = False
    distiller_beta: float = 0.0
    distiller_sampler_backend: SamplerBackend = "post_filter_exact"
    compact_capture_lane: bool = False


@dataclass
class ResidualRuntimeState:
    config: ResidualRuntimeConfig = field(default_factory=ResidualRuntimeConfig)

    decode_row_idx: Optional[torch.Tensor] = None
    decode_valid_mask: Optional[torch.Tensor] = None
    decode_count: int = 0
    decode_prompt_idxs: list[int] = field(default_factory=list)
    decode_sample_idxs: list[int] = field(default_factory=list)
    decode_request_ids: list[str] = field(default_factory=list)
    decode_prompt_idx_buf: torch.Tensor | None = None
    decode_sample_idx_buf: torch.Tensor | None = None
    decode_prompt_idx_tensor: torch.Tensor | None = None
    decode_sample_idx_tensor: torch.Tensor | None = None
    decode_compact_row_idx: Optional[torch.Tensor] = None
    decode_compact_count: int = 0
    decode_compact_row_ids: tuple[int, ...] = ()

    tap_layers: Dict[str, torch.nn.Module] = field(default_factory=dict)
    tap_scratch: Dict[str, torch.Tensor] = field(default_factory=dict)
    tap_decode_hidden: Dict[str, torch.Tensor] = field(default_factory=dict)
    tap_decode_hidden_compact: Dict[str, torch.Tensor] = field(default_factory=dict)
    capture_full_residual_rows: bool = True
    residual_raw_paths: Dict[ResidualLocator, str] = field(default_factory=dict)
    residual_bindings: Dict[str, ResidualPathBinding] = field(default_factory=dict)

    source_resolved_path: str = ""
    target_resolved_path: str = ""
    launch_consumer_from_hooks: bool = True
    reqid_to_promptidx: Dict[str, int] = field(default_factory=dict)
    reqid_to_sampleidx: Dict[str, int] = field(default_factory=dict)
    event_step_id: int = 0
    decode_post_logits_launched_step_id: int = -1
    sampler_precompute: SamplerPrecomputeState = field(default_factory=SamplerPrecomputeState)
    path_hotspot_enabled: bool = False
    path_hotspot_cpu_ms: dict[str, float] = field(default_factory=dict)
    path_hotspot_counts: dict[str, int] = field(default_factory=dict)

    consumer: Optional[RuntimeConsumer] = None
    consumer_registry: Optional[ConsumerRegistry] = None
    dispatch_plan: Optional[DispatchPlan] = None

    def reset_step(self) -> None:
        self.decode_count = 0
        self.decode_prompt_idxs = []
        self.decode_sample_idxs = []
        self.decode_request_ids = []
        self.decode_prompt_idx_buf = self.decode_prompt_idx_buf
        self.decode_sample_idx_buf = self.decode_sample_idx_buf
        self.decode_prompt_idx_tensor = None
        self.decode_sample_idx_tensor = None
        self.decode_compact_count = 0
        self.decode_compact_row_ids = ()
        self.decode_post_logits_launched_step_id = -1
        self.sampler_precompute.reset_step()


RUNTIME = ResidualRuntimeState()


def _ensure_runner_binding() -> None:
    global GPUModelRunner, _ORIG_LOAD_MODEL, _ORIG_PREPARE_INPUTS, _ORIG_EXECUTE_MODEL
    if (
        GPUModelRunner is not None
        and _ORIG_LOAD_MODEL is not None
        and _ORIG_PREPARE_INPUTS is not None
        and _ORIG_EXECUTE_MODEL is not None
    ):
        return

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as _GPUModelRunner  # noqa: E402

    GPUModelRunner = _GPUModelRunner
    _ORIG_LOAD_MODEL = GPUModelRunner.load_model
    _ORIG_PREPARE_INPUTS = GPUModelRunner._prepare_inputs
    _ORIG_EXECUTE_MODEL = GPUModelRunner.execute_model


def _ensure_dispatch_registry() -> ConsumerRegistry:
    if RUNTIME.consumer_registry is None:
        RUNTIME.consumer_registry = ConsumerRegistry()
    return RUNTIME.consumer_registry


def _refresh_dispatch_plan() -> None:
    registry = _ensure_dispatch_registry()
    consumers = registry.consumers()
    if isinstance(RUNTIME.consumer, BaseConsumer):
        consumers = consumers + [RUNTIME.consumer]
    if not consumers:
        RUNTIME.dispatch_plan = None
        return
    plan = DispatchPlan.build(consumers)
    RUNTIME.dispatch_plan = None if plan.is_empty() else plan


def register_dispatch_consumer(consumer: BaseConsumer) -> None:
    _ensure_dispatch_registry().register(consumer)
    _refresh_dispatch_plan()


def replace_dispatch_consumers(consumers: Sequence[BaseConsumer]) -> None:
    _ensure_dispatch_registry().replace_all(consumers)
    _refresh_dispatch_plan()


def clear_dispatch_consumers() -> None:
    _ensure_dispatch_registry().clear()
    _refresh_dispatch_plan()


def set_runtime_consumer(consumer: BaseConsumer | None) -> None:
    RUNTIME.consumer = consumer
    _refresh_dispatch_plan()


def runner_uses_compilation_or_cudagraph(runner: RunnerLike) -> bool:
    use_cuda_graph = getattr(runner, "use_cuda_graph", None)
    if use_cuda_graph is True:
        return True

    cfg = getattr(runner, "compilation_config", None)
    if cfg is None:
        return False

    level_raw = getattr(cfg, "level", 0)
    mode_raw = getattr(cfg, "cudagraph_mode", 0)
    try:
        level = int(level_raw)
    except Exception:
        level = 0
    try:
        cg_mode = int(mode_raw)
    except Exception:
        cg_mode = 0
    return (level > 0) or (cg_mode > 0)


def infer_hidden_dtype(layer: torch.nn.Module) -> torch.dtype:
    for p in layer.parameters():
        return p.dtype
    for b in layer.buffers():
        return b.dtype
    return torch.float32


def resolve_module_by_path_with_fallback(model: torch.nn.Module, path: str) -> tuple[torch.nn.Module, str]:
    from tllm.common.path_resolution import candidate_capture_paths

    errors: list[str] = []
    for candidate in candidate_capture_paths(path):
        try:
            obj = resolve_object_by_path(model, candidate)
        except RuntimeError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        if not isinstance(obj, torch.nn.Module):
            errors.append(f"{candidate}: resolved to non-module {type(obj).__name__}")
            continue
        return obj, candidate

    details = "; ".join(errors) if errors else "no candidate"
    raise RuntimeError(
        f"Cannot resolve layer path `{path}` from model root `{type(model).__name__}`. Tried: {details}"
    )


_runner_uses_compilation_or_cudagraph = runner_uses_compilation_or_cudagraph
_infer_hidden_dtype = infer_hidden_dtype
_resolve_module_by_path_with_fallback = resolve_module_by_path_with_fallback


def _resolve_prompt_sample_for_req_id(req_id: str) -> tuple[int, int]:
    prompt_idx = RUNTIME.reqid_to_promptidx.get(req_id)
    if prompt_idx is not None:
        return int(prompt_idx), int(RUNTIME.reqid_to_sampleidx.get(req_id, 0))

    if "_" in req_id:
        maybe_sample_idx, parent_req_id = req_id.split("_", 1)
        if maybe_sample_idx.isdigit():
            parent_prompt_idx = RUNTIME.reqid_to_promptidx.get(parent_req_id)
            if parent_prompt_idx is not None:
                return int(parent_prompt_idx), int(maybe_sample_idx)
    return -1, -1


def _is_torch_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None:
        fn = getattr(compiler, "is_compiling", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                pass

    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None:
        fn = getattr(dynamo, "is_compiling", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                pass

    return False


def configure_runtime(
    *,
    graph_scratch_rows: int,
    tap_layer_paths: Sequence[str],
    source_layer_path: str,
    target_layer_path: str,
    enable_esamp_training: bool,
    distiller_hidden_dim: int,
    distiller_lr: float,
    per_request_models: bool = False,
    per_request_model_bank: bool = False,
    model_bank_slots: int = 0,
    model_bank_flush_interval: int = 1,
    model_bank_rank: int = 64,
    model_bank_use_output_layernorm: bool = True,
    model_bank_initializer: SVDModelBankInitializerConfig | None = None,
    model_bank_train_cudagraph: bool = False,
    model_bank_forward_backend: str = "torch",
    adaptation_pipeline_slots: int = 4,
    adaptation_stream_mode: str = "dual",
    adaptation_stream_priority: int = 0,
    trace_per_request_losses: bool = False,
    trace_interval: int = 1,
    trace_max_points: int = 0,
    enable_distiller_intervention: bool = False,
    distiller_beta: float = 0.0,
    distiller_sampler_backend: SamplerBackend = "post_filter_exact",
    compact_capture_lane: bool = False,
) -> None:
    cfg = RUNTIME.config
    cfg.graph_scratch_rows = max(0, int(graph_scratch_rows))
    cfg.tap_layer_paths = [str(path).strip() for path in tap_layer_paths if str(path).strip()]
    cfg.source_layer_path = str(source_layer_path).strip()
    cfg.target_layer_path = str(target_layer_path).strip()
    cfg.enable_esamp_training = bool(enable_esamp_training)
    cfg.distiller_hidden_dim = max(1, int(distiller_hidden_dim))
    cfg.distiller_lr = float(distiller_lr)
    cfg.per_request_models = bool(per_request_models)
    cfg.per_request_model_bank = bool(per_request_model_bank)
    cfg.model_bank_slots = max(0, int(model_bank_slots))
    cfg.model_bank_flush_interval = max(1, int(model_bank_flush_interval))
    cfg.model_bank_rank = max(1, int(model_bank_rank))
    cfg.model_bank_use_output_layernorm = bool(model_bank_use_output_layernorm)
    cfg.model_bank_initializer = model_bank_initializer
    cfg.model_bank_train_cudagraph = bool(model_bank_train_cudagraph)
    cfg.model_bank_forward_backend = normalize_model_bank_forward_backend(model_bank_forward_backend)
    cfg.adaptation_pipeline_slots = max(1, int(adaptation_pipeline_slots))
    cfg.adaptation_stream_mode = normalize_adaptation_stream_mode(adaptation_stream_mode)
    cfg.adaptation_stream_priority = int(adaptation_stream_priority)
    cfg.trace_per_request_losses = bool(trace_per_request_losses)
    cfg.trace_interval = max(1, int(trace_interval))
    cfg.trace_max_points = max(0, int(trace_max_points))
    cfg.enable_distiller_intervention = bool(enable_distiller_intervention)
    cfg.distiller_beta = float(distiller_beta)
    cfg.distiller_sampler_backend = normalize_sampler_backend(str(distiller_sampler_backend).strip() or "post_filter_exact")
    cfg.compact_capture_lane = bool(compact_capture_lane)

    if not cfg.source_layer_path:
        raise RuntimeError("source_layer_path must be non-empty")
    if not cfg.target_layer_path:
        raise RuntimeError("target_layer_path must be non-empty")

    RUNTIME.residual_raw_paths = default_raw_paths_from_config(cfg)
    precompute = RUNTIME.sampler_precompute
    precompute.timing_enabled = os.getenv("TLLM_TRACE_DISTILLER_TIMING", "") == "1"
    precompute.precompute_event_pairs = []
    precompute.wait_event_pairs = []
    precompute.fallback_event_pairs = []
    precompute.port_publish_attempt_count = 0
    precompute.port_publish_hit_count = 0
    precompute.schedule_attempt_count = 0
    precompute.schedule_hit_count = 0
    RUNTIME.path_hotspot_enabled = os.getenv("TLLM_TRACE_PATH_HOTSPOTS", "") == "1"
    RUNTIME.path_hotspot_cpu_ms = {}
    RUNTIME.path_hotspot_counts = {}
    _refresh_dispatch_plan()
    RUNTIME.reset_step()


def set_esamp_training_enabled(enabled: bool) -> None:
    RUNTIME.config.enable_esamp_training = bool(enabled)
    if RUNTIME.consumer is not None:
        RUNTIME.consumer.set_enabled(enabled)


def synchronize_esamp() -> None:
    if RUNTIME.consumer is not None:
        synchronize_consumer(RUNTIME.consumer)
    registry = RUNTIME.consumer_registry
    if registry is None:
        return
    for consumer in registry.consumers():
        synchronize_consumer(consumer)


def read_and_reset_esamp_stats(sync: bool = True) -> ESampStats:
    if RUNTIME.consumer is None:
        return ESampStats(loss_avg=0.0, loss_count=0)
    return RUNTIME.consumer.read_and_reset_stats(sync=sync)


def read_and_reset_esamp_per_request_stats(sync: bool = True) -> Dict[int, ESampStats]:
    if RUNTIME.consumer is None:
        return {}
    return RUNTIME.consumer.read_and_reset_per_request_stats(sync=sync)


def read_and_reset_distiller_timing_stats(sync: bool = True) -> DistillerTimingStats:
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    out = DistillerTimingStats()

    def _accumulate(pairs: list[tuple[object, object]]) -> tuple[float, int]:
        total = 0.0
        count = 0
        for start, end in pairs:
            try:
                total += float(start.elapsed_time(end))
            except RuntimeError:
                continue
            count += 1
        return total, count

    precompute = RUNTIME.sampler_precompute
    out.precompute_ms_total, out.precompute_count = _accumulate(precompute.precompute_event_pairs)
    out.wait_ms_total, out.wait_count = _accumulate(precompute.wait_event_pairs)
    out.fallback_ms_total, out.fallback_count = _accumulate(precompute.fallback_event_pairs)
    out.port_publish_attempt_count = int(precompute.port_publish_attempt_count)
    out.port_publish_hit_count = int(precompute.port_publish_hit_count)
    out.schedule_attempt_count = int(precompute.schedule_attempt_count)
    out.schedule_hit_count = int(precompute.schedule_hit_count)
    out.candidate_sample_count = int(precompute.candidate_sample_count)
    out.candidate_token_count = int(precompute.candidate_token_count)
    out.candidate_row_count = int(precompute.candidate_row_count)
    out.candidate_max_count = int(precompute.candidate_max_count)
    out.candidate_kernel_triton_count = int(precompute.candidate_kernel_triton_count)
    out.candidate_kernel_torch_count = int(precompute.candidate_kernel_torch_count)
    out.candidate_kernel_fallback_count = int(precompute.candidate_kernel_fallback_count)
    precompute.precompute_event_pairs = []
    precompute.wait_event_pairs = []
    precompute.fallback_event_pairs = []
    precompute.port_publish_attempt_count = 0
    precompute.port_publish_hit_count = 0
    precompute.schedule_attempt_count = 0
    precompute.schedule_hit_count = 0
    precompute.candidate_sample_count = 0
    precompute.candidate_token_count = 0
    precompute.candidate_row_count = 0
    precompute.candidate_max_count = 0
    precompute.candidate_kernel_triton_count = 0
    precompute.candidate_kernel_torch_count = 0
    precompute.candidate_kernel_fallback_count = 0
    return out


def record_path_hotspot_cpu(name: str, ms: float) -> None:
    if not bool(getattr(RUNTIME, "path_hotspot_enabled", False)):
        return
    key = str(name).strip()
    if not key:
        return
    RUNTIME.path_hotspot_cpu_ms[key] = float(RUNTIME.path_hotspot_cpu_ms.get(key, 0.0)) + float(ms)
    RUNTIME.path_hotspot_counts[key] = int(RUNTIME.path_hotspot_counts.get(key, 0)) + 1


def read_and_reset_path_hotspot_stats(sync: bool = True) -> PathHotspotStats:
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    out = PathHotspotStats(
        cpu_ms_total={str(k): float(v) for k, v in RUNTIME.path_hotspot_cpu_ms.items()},
        counts={str(k): int(v) for k, v in RUNTIME.path_hotspot_counts.items()},
    )
    RUNTIME.path_hotspot_cpu_ms = {}
    RUNTIME.path_hotspot_counts = {}
    return out


def read_graph_debug_stats(*, mode: str = "model_bank") -> GraphDebugStats:
    consumer = getattr(RUNTIME, "consumer", None)
    engine = getattr(consumer, "_engine", None)
    if engine is None:
        return GraphDebugStats()
    if str(mode) == "model_bank":
        slot_graphs = list(getattr(engine.state, "model_bank_slot_graphs", {}).values())
        active_slot_graphs = [graph for graph in slot_graphs if graph.capture_attempt_count or graph.replay_attempt_count or graph.capture_state != "uncaptured"]
        if active_slot_graphs:
            disabled = [graph for graph in active_slot_graphs if graph.capture_state == "disabled"]
            captured = [graph for graph in active_slot_graphs if graph.capture_state == "captured"]
            state = "disabled" if disabled else ("captured" if captured else "uncaptured")
            return GraphDebugStats(
                capture_state=state,
                capture_attempt_count=sum(int(graph.capture_attempt_count) for graph in active_slot_graphs),
                skip_not_enabled_count=sum(int(graph.skip_not_enabled_count) for graph in active_slot_graphs),
                skip_missing_optimizer_state_count=sum(int(graph.skip_missing_optimizer_state_count) for graph in active_slot_graphs),
                skip_wrong_device_count=sum(int(graph.skip_wrong_device_count) for graph in active_slot_graphs),
                replay_attempt_count=sum(int(graph.replay_attempt_count) for graph in active_slot_graphs),
                replay_hit_count=sum(int(graph.replay_hit_count) for graph in active_slot_graphs),
                replay_stage_miss_count=sum(int(graph.replay_stage_miss_count) for graph in active_slot_graphs),
                kernel_fallback_count=sum(int(graph.kernel_fallback_count) for graph in active_slot_graphs),
                disable_reason="; ".join(str(graph.disable_reason) for graph in disabled if str(graph.disable_reason)),
            )
    graph = engine.state.graphs.get(str(mode))
    if graph is None:
        return GraphDebugStats()
    return GraphDebugStats(
        capture_state=str(graph.capture_state),
        capture_attempt_count=int(graph.capture_attempt_count),
        skip_not_enabled_count=int(graph.skip_not_enabled_count),
        skip_missing_optimizer_state_count=int(graph.skip_missing_optimizer_state_count),
        skip_wrong_device_count=int(graph.skip_wrong_device_count),
        replay_attempt_count=int(graph.replay_attempt_count),
        replay_hit_count=int(graph.replay_hit_count),
        replay_stage_miss_count=int(graph.replay_stage_miss_count),
        kernel_fallback_count=int(graph.kernel_fallback_count),
        disable_reason=str(graph.disable_reason),
    )


def _prepare_decode_localization(runner: RunnerLike, out: tuple[object, ...]) -> None:
    _hooks.prepare_decode_localization(core=sys.modules[__name__], runner=runner, out=out)


def _build_tap_path_list() -> list[str]:
    return _hooks.build_tap_path_list(core=sys.modules[__name__])


def _setup_runtime_hooks_if_active(runner: RunnerLike) -> None:
    _hooks.setup_runtime_hooks_if_active(core=sys.modules[__name__], runner=runner)


def _wrapped_load_model(self: object, *args: object, **kwargs: object) -> object:
    return _hooks.wrapped_load_model(core=sys.modules[__name__], runner=self, args=args, kwargs=kwargs)


def _wrapped_prepare_inputs(self: object, scheduler_output: object) -> object:
    return _hooks.wrapped_prepare_inputs(core=sys.modules[__name__], runner=self, scheduler_output=scheduler_output)


def _wrapped_execute_model(self: object, *args: object, **kwargs: object) -> object:
    return _hooks.wrapped_execute_model(core=sys.modules[__name__], runner=self, args=args, kwargs=kwargs)


def install_runner_patch() -> None:
    _ensure_runner_binding()
    _hooks.install_runner_patch(core=sys.modules[__name__])


def make_llm(*args: object, **kwargs: object) -> object:
    install_runner_patch()
    from tllm.util import tools as _tool_helpers

    return _tool_helpers.make_plain_llm(*args, **kwargs)


__all__ = [
    "DistillerTimingStats",
    "GraphDebugStats",
    "PathHotspotStats",
    "MODEL_HOOK_FLAG",
    "MODEL_HOOK_SPEC_ATTR",
    "RUNTIME",
    "ResidualRuntimeConfig",
    "ResidualRuntimeState",
    "RuntimeConsumer",
    "clear_dispatch_consumers",
    "compute_decode_localization",
    "configure_runtime",
    "infer_hidden_dtype",
    "make_llm",
    "pick_common_attn_metadata",
    "read_and_reset_distiller_timing_stats",
    "read_and_reset_path_hotspot_stats",
    "read_graph_debug_stats",
    "read_and_reset_esamp_per_request_stats",
    "read_and_reset_esamp_stats",
    "register_dispatch_consumer",
    "replace_dispatch_consumers",
    "resolve_module_by_path_with_fallback",
    "record_path_hotspot_cpu",
    "runner_uses_compilation_or_cudagraph",
    "set_runtime_consumer",
    "set_esamp_training_enabled",
    "synchronize_esamp",
]
