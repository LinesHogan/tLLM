#!/usr/bin/env python3
"""ESamp train engine with centralized state and shared graph orchestration."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F

from tllm.consumers.esamp.config import ESampConsumerConfig
from tllm.consumers.esamp.config import normalize_adaptation_stream_mode
from tllm.consumers.esamp.initializers.svd import (
    SVDModelBankInitializer,
    SVDModelBankInitializerConfig,
    build_model_bank_initializer,
)
from tllm.consumers.esamp.model_bank_backend import (
    ModelBankForwardBackendName,
    normalize_model_bank_forward_backend,
    select_model_bank_forward_backend,
)
from tllm.consumers.esamp.model import LowRankGatedResidualModel

_Mode = Literal["shared", "model_bank"]
_AdaptationStreamMode = Literal["dual", "single", "serial"]
_ResidualModel = LowRankGatedResidualModel
_CaptureState = Literal["uncaptured", "captured", "disabled"]


class _ReplayGraphMissing:
    pass


_MISSING_REPLAY_GRAPH = _ReplayGraphMissing()
_ReplayGraphRef = torch.cuda.CUDAGraph | _ReplayGraphMissing
_DEFAULT_ADAPTATION_STREAM_PRIORITY = 0


@dataclass(slots=True, frozen=True)
class ESampStats:
    loss_avg: float
    loss_count: int
    trace_losses: tuple[float, ...] = ()


def _group_rows(prompt_idxs: Sequence[int], active_rows: int) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for row_i, prompt_idx in enumerate(prompt_idxs[: max(0, int(active_rows))]):
        if int(prompt_idx) >= 0:
            groups.setdefault(int(prompt_idx), []).append(int(row_i))
    return groups


def _copy_rows(dst: torch.Tensor, src: torch.Tensor) -> int:
    if dst.ndim != 2 or src.ndim != 2 or int(dst.shape[1]) != int(src.shape[1]):
        raise ValueError("ESamp engine expects rank-2 hidden batches with matching width")
    copied = min(int(dst.shape[0]), int(src.shape[0]))
    if copied > 0:
        dst[:copied].copy_(src[:copied])
    return copied


def _ensure_events(events: list[torch.cuda.Event], size: int) -> list[torch.cuda.Event]:
    return events if len(events) == size else [torch.cuda.Event(blocking=False) for _ in range(size)]


def _copy_parameters(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    with torch.no_grad():
        for p_dst, p_src in zip(dst.parameters(), src.parameters()):
            p_dst.copy_(p_src)


def _make_esamp_stream(device: torch.device, *, priority: int) -> torch.cuda.Stream:
    return torch.cuda.Stream(device=device, priority=int(priority))


def _sampling_lookup_capacity(*, slots: int, rows: int) -> int:
    return max(4096, int(slots), int(rows) * 8)


def _make_optimizer(
    params: Iterable[torch.nn.Parameter],
    *,
    lr: float,
    capturable: bool,
) -> torch.optim.AdamW:
    kwargs = {"lr": lr, "foreach": False}
    if capturable:
        kwargs["capturable"] = True
    return torch.optim.AdamW(params, **kwargs)


def _make_model_bank_optimizer(
    params: Iterable[torch.nn.Parameter],
    *,
    lr: float,
    capturable: bool,
) -> torch.optim.Optimizer:
    if capturable:
        return torch.optim.SGD(params, lr=lr)
    return _make_optimizer(params, lr=lr, capturable=False)


def _optimizer_requires_state_for_capture(optimizer: torch.optim.Optimizer) -> bool:
    return isinstance(optimizer, torch.optim.AdamW)


def _record_trace_sample(entry: "_PerRequestEntry", loss: torch.Tensor, *, interval: int, max_points: int) -> None:
    entry.trace_update_count += 1
    if max_points == 0 or (entry.trace_update_count - 1) % max(1, interval) != 0:
        return
    if len(entry.trace_losses) >= max_points:
        return
    entry.trace_losses.append(float(loss.detach().item()))


@contextmanager
def _grad_context() -> Iterator[None]:
    with torch.inference_mode(False), torch.enable_grad():
        yield


@dataclass(slots=True)
class _PerRequestEntry:
    loss_sum: torch.Tensor
    loss_count: torch.Tensor
    trace_losses: list[float] = field(default_factory=list)
    trace_update_count: int = 0


@dataclass(slots=True)
class _PerRequestTrainer:
    train_model: _ResidualModel
    forward_model: _ResidualModel
    optimizer: torch.optim.AdamW
    stats: _PerRequestEntry


@dataclass(slots=True)
class _GraphState:
    rows: int = 0
    slots: int = 0
    active_rows: int = 0
    external_inputs: bool = False
    capture_state: _CaptureState = "uncaptured"
    replay_graph: _ReplayGraphRef = field(default_factory=lambda: _MISSING_REPLAY_GRAPH)
    disable_reason: str = ""
    capture_attempt_count: int = 0
    skip_not_enabled_count: int = 0
    skip_missing_optimizer_state_count: int = 0
    skip_wrong_device_count: int = 0
    replay_attempt_count: int = 0
    replay_hit_count: int = 0
    replay_stage_miss_count: int = 0
    kernel_fallback_count: int = 0
    buffers: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(slots=True)
class _ModelBankParams:
    a: torch.nn.Parameter
    g: torch.nn.Parameter
    b: torch.nn.Parameter
    gate_bias: torch.nn.Parameter
    out_ln_weight: torch.nn.Parameter
    out_ln_bias: torch.nn.Parameter

    def as_dict(self) -> dict[str, torch.Tensor | torch.nn.Parameter]:
        return {
            "a": self.a,
            "g": self.g,
            "b": self.b,
            "gate_bias": self.gate_bias,
            "out_ln_weight": self.out_ln_weight,
            "out_ln_bias": self.out_ln_bias,
        }


@dataclass(slots=True)
class _SharedTrainResources:
    forward_model: _ResidualModel
    train_model: _ResidualModel
    optimizer: torch.optim.AdamW


@dataclass(slots=True)
class _PipelineBuffers:
    src: torch.Tensor
    tgt: torch.Tensor


@dataclass(slots=True)
class _StatsBuffers:
    loss_sum: torch.Tensor
    loss_count: torch.Tensor


@dataclass(slots=True)
class _EngineState:
    hidden_dim: int = 1
    lr: float = 0.0
    enabled: bool = True
    per_request_models: bool = False
    per_request_model_bank: bool = False
    model_bank_slots: int = 0
    model_bank_flush_interval: int = 1
    model_bank_rank: int = 64
    model_bank_use_output_layernorm: bool = True
    model_bank_initializer: SVDModelBankInitializer | None = None
    model_bank_train_cudagraph: bool = False
    model_bank_forward_backend: ModelBankForwardBackendName = "torch"
    trace_per_request_losses: bool = False
    trace_interval: int = 1
    trace_max_points: int = 0
    model_bank_rank_effective: int = 0
    pipeline_slots: int = 4
    adaptation_stream_mode: _AdaptationStreamMode = "dual"
    adaptation_stream_priority: int = _DEFAULT_ADAPTATION_STREAM_PRIORITY

    device: torch.device = field(default_factory=lambda: torch.device("cuda"))
    hidden_dtype: torch.dtype = torch.float32
    hidden_size: int = 0
    forward_stream: torch.cuda.Stream | None = None
    train_stream: torch.cuda.Stream | None = None

    shared: _SharedTrainResources | None = None

    per_request: dict[int, _PerRequestEntry] = field(default_factory=dict)
    per_request_trainers: dict[int, _PerRequestTrainer] = field(default_factory=dict)
    prompt_to_slot: dict[int, int] = field(default_factory=dict)
    slot_to_prompt: dict[int, int] = field(default_factory=dict)
    sampling_lookup_device: torch.device | None = None
    sampling_lookup_size: int = -1
    sampling_lookup_keys: torch.Tensor | None = None
    sampling_lookup_values: torch.Tensor | None = None
    sampling_lookup_dense: torch.Tensor | None = None
    sampling_lookup_capacity: int = 0
    model_bank_slot_tensor_cache: dict[tuple[int, ...], torch.Tensor] = field(default_factory=dict)
    model_bank: _ModelBankParams | None = None
    model_bank_optimizer: torch.optim.Optimizer | None = None
    model_bank_next_slot: int = 0

    current_step_slot: int | None = None
    ready_step_slot: int | None = None
    next_pipeline_slot: int = 0
    pending_train_queue: deque[tuple[int, int, tuple[int, ...]]] = field(default_factory=deque)
    src_ready_events: list[torch.cuda.Event] = field(default_factory=list)
    tgt_ready_events: list[torch.cuda.Event] = field(default_factory=list)
    src_staged_events: list[torch.cuda.Event] = field(default_factory=list)
    tgt_done_events: list[torch.cuda.Event] = field(default_factory=list)
    slot_train_done_events: list[torch.cuda.Event] = field(default_factory=list)

    pipeline: _PipelineBuffers | None = None
    stats: _StatsBuffers | None = None
    graphs: dict[_Mode, _GraphState] = field(
        default_factory=lambda: {"shared": _GraphState(), "model_bank": _GraphState()}
    )
    model_bank_slot_graphs: dict[int, _GraphState] = field(default_factory=dict)


class _ESampEngineCore:
    """Centralized ESamp engine core."""

    def __init__(
        self,
        hidden_dim: int,
        lr: float,
        enabled: bool = True,
        per_request_models: bool = False,
        per_request_model_bank: bool = False,
        model_bank_slots: int = 0,
        model_bank_flush_interval: int = 1,
        model_bank_rank: int = 64,
        model_bank_use_output_layernorm: bool = True,
        model_bank_initializer: SVDModelBankInitializer | None = None,
        model_bank_train_cudagraph: bool = False,
        model_bank_forward_backend: str = "torch",
        adaptation_pipeline_slots: int = 4,
        adaptation_stream_mode: str = "dual",
        adaptation_stream_priority: int = _DEFAULT_ADAPTATION_STREAM_PRIORITY,
        trace_per_request_losses: bool = False,
        trace_interval: int = 1,
        trace_max_points: int = 0,
    ) -> None:
        self.state = _EngineState(
            hidden_dim=max(1, int(hidden_dim)),
            lr=float(lr),
            enabled=bool(enabled),
            per_request_models=bool(per_request_models),
            per_request_model_bank=bool(per_request_model_bank),
            model_bank_slots=max(0, int(model_bank_slots)),
            model_bank_flush_interval=max(1, int(model_bank_flush_interval)),
            model_bank_rank=max(1, int(model_bank_rank)),
            model_bank_use_output_layernorm=bool(model_bank_use_output_layernorm),
            model_bank_initializer=model_bank_initializer,
            model_bank_train_cudagraph=bool(model_bank_train_cudagraph),
            model_bank_forward_backend=normalize_model_bank_forward_backend(model_bank_forward_backend),
            pipeline_slots=max(1, int(adaptation_pipeline_slots)),
            adaptation_stream_mode=normalize_adaptation_stream_mode(adaptation_stream_mode),
            adaptation_stream_priority=int(adaptation_stream_priority),
            trace_per_request_losses=bool(trace_per_request_losses),
            trace_interval=max(1, int(trace_interval)),
            trace_max_points=max(0, int(trace_max_points)),
        )

    def _config_resource_signature(self, config: ESampConsumerConfig) -> tuple[object, ...]:
        return (
            max(1, int(config.distiller_hidden_dim)),
            float(config.distiller_lr),
            bool(config.per_request_models),
            bool(config.per_request_model_bank),
            max(0, int(config.model_bank_slots)),
            max(1, int(config.model_bank_rank)),
            bool(config.model_bank_use_output_layernorm),
            config.model_bank_initializer,
            bool(config.model_bank_train_cudagraph),
            normalize_model_bank_forward_backend(getattr(config, "model_bank_forward_backend", "torch")),
            max(1, int(getattr(config, "adaptation_pipeline_slots", 4))),
            normalize_adaptation_stream_mode(getattr(config, "adaptation_stream_mode", "dual")),
            int(getattr(config, "adaptation_stream_priority", _DEFAULT_ADAPTATION_STREAM_PRIORITY)),
        )

    def _resource_signature(self) -> tuple[object, ...]:
        s = self.state
        initializer = None if s.model_bank_initializer is None else s.model_bank_initializer.config
        return (
            int(s.hidden_dim),
            float(s.lr),
            bool(s.per_request_models),
            bool(s.per_request_model_bank),
            int(s.model_bank_slots),
            int(s.model_bank_rank),
            bool(s.model_bank_use_output_layernorm),
            initializer,
            bool(s.model_bank_train_cudagraph),
            normalize_model_bank_forward_backend(s.model_bank_forward_backend),
            int(s.pipeline_slots),
            normalize_adaptation_stream_mode(s.adaptation_stream_mode),
            int(s.adaptation_stream_priority),
        )

    def _invalidate_training_resources(self) -> None:
        s = self.state
        s.per_request.clear()
        s.per_request_trainers.clear()
        s.sampling_lookup_device = None
        s.sampling_lookup_size = -1
        s.sampling_lookup_keys = None
        s.sampling_lookup_values = None
        s.sampling_lookup_dense = None
        s.sampling_lookup_capacity = 0
        s.shared = None
        s.model_bank = None
        s.model_bank_optimizer = None
        s.model_bank_rank_effective = 0
        self._clear_in_flight_step_state()
        self._reset_runtime_cache()

    def _clear_in_flight_step_state(self) -> None:
        s = self.state
        s.pending_train_queue.clear()
        s.current_step_slot = None
        s.ready_step_slot = None

    def configure(self, config: ESampConsumerConfig) -> None:
        s = self.state
        reset_resources = self._resource_signature() != self._config_resource_signature(config)
        reset_runtime_cache = reset_resources or int(s.model_bank_flush_interval) != max(1, int(config.model_bank_flush_interval))
        s.hidden_dim = max(1, int(config.distiller_hidden_dim))
        s.lr = float(config.distiller_lr)
        s.enabled = bool(config.enable_esamp_training)
        s.per_request_models = bool(config.per_request_models)
        s.per_request_model_bank = bool(config.per_request_model_bank)
        s.model_bank_slots = max(0, int(config.model_bank_slots))
        s.model_bank_flush_interval = max(1, int(config.model_bank_flush_interval))
        s.model_bank_rank = max(1, int(config.model_bank_rank))
        s.model_bank_use_output_layernorm = bool(config.model_bank_use_output_layernorm)
        if config.model_bank_initializer is None:
            s.model_bank_initializer = None
        elif (
            s.model_bank_initializer is None
            or type(s.model_bank_initializer) is not SVDModelBankInitializer
            or s.model_bank_initializer.config != config.model_bank_initializer
        ):
            s.model_bank_initializer = build_model_bank_initializer(config.model_bank_initializer)
        s.model_bank_train_cudagraph = bool(config.model_bank_train_cudagraph)
        s.model_bank_forward_backend = normalize_model_bank_forward_backend(getattr(config, "model_bank_forward_backend", "torch"))
        s.pipeline_slots = max(1, int(getattr(config, "adaptation_pipeline_slots", 4)))
        s.adaptation_stream_mode = normalize_adaptation_stream_mode(getattr(config, "adaptation_stream_mode", "dual"))
        s.adaptation_stream_priority = int(getattr(config, "adaptation_stream_priority", _DEFAULT_ADAPTATION_STREAM_PRIORITY))
        s.trace_per_request_losses = bool(config.trace_per_request_losses)
        s.trace_interval = max(1, int(config.trace_interval))
        s.trace_max_points = max(0, int(config.trace_max_points))
        self._clear_in_flight_step_state()
        if reset_resources:
            self._invalidate_training_resources()
        elif reset_runtime_cache:
            self._reset_runtime_cache()

    @property
    def using_model_bank(self) -> bool:
        s = self.state
        return s.per_request_models and s.per_request_model_bank

    def _build_model(self, *, hidden_size: int, device: torch.device, hidden_dtype: torch.dtype) -> _ResidualModel:
        rank = max(1, min(self.state.hidden_dim, hidden_size))
        return LowRankGatedResidualModel(
            hidden_size=hidden_size,
            rank=rank,
            use_output_layernorm=True,
            device=device,
            dtype=hidden_dtype,
        )

    def set_enabled(self, enabled: bool) -> None:
        self.state.enabled = bool(enabled)

    def _build_model_pair(
        self,
        *,
        hidden_size: int,
        device: torch.device,
        hidden_dtype: torch.dtype,
    ) -> tuple[_ResidualModel, _ResidualModel]:
        train_model = self._build_model(hidden_size=hidden_size, device=device, hidden_dtype=hidden_dtype).train()
        forward_model = self._build_model(hidden_size=hidden_size, device=device, hidden_dtype=hidden_dtype).eval()
        for p in forward_model.parameters():
            p.requires_grad_(False)
        forward_model.load_state_dict(train_model.state_dict())
        return train_model, forward_model

    def _reset_runtime_cache(self) -> None:
        s = self.state
        s.prompt_to_slot.clear()
        s.slot_to_prompt.clear()
        s.model_bank_next_slot = 0
        if s.sampling_lookup_dense is not None:
            s.sampling_lookup_dense.fill_(-1)
        s.sampling_lookup_size = 0
        s.sampling_lookup_keys = None
        s.sampling_lookup_values = None
        s.model_bank_slot_tensor_cache.clear()
        if s.model_bank_initializer is not None:
            s.model_bank_initializer.reset_runtime_state()
        for graph in s.graphs.values():
            self._reset_graph_state(graph)
        for graph in s.model_bank_slot_graphs.values():
            self._reset_graph_state(graph)
        s.model_bank_slot_graphs.clear()

    @staticmethod
    def _reset_graph_state(graph: _GraphState) -> None:
        graph.capture_state = "uncaptured"
        graph.replay_graph = _MISSING_REPLAY_GRAPH
        graph.rows = 0
        graph.slots = 0
        graph.active_rows = 0
        graph.external_inputs = False
        graph.disable_reason = ""
        graph.capture_attempt_count = 0
        graph.skip_not_enabled_count = 0
        graph.skip_missing_optimizer_state_count = 0
        graph.skip_wrong_device_count = 0
        graph.replay_attempt_count = 0
        graph.replay_hit_count = 0
        graph.replay_stage_miss_count = 0
        graph.kernel_fallback_count = 0
        graph.buffers.clear()

    def _reset_pipeline_runtime_state(self, *, reset_streams: bool) -> None:
        s = self.state
        if reset_streams:
            s.forward_stream = None
            s.train_stream = None
        s.src_ready_events = []
        s.tgt_ready_events = []
        s.src_staged_events = []
        s.tgt_done_events = []
        s.slot_train_done_events = []
        s.pending_train_queue.clear()
        s.current_step_slot = None
        s.ready_step_slot = None
        s.next_pipeline_slot = 0
        s.pipeline = None
        s.stats = None

    def _ensure_stats(self) -> None:
        s = self.state
        if s.hidden_size <= 0:
            raise RuntimeError("ESamp engine requires ensure_resources before stats buffers exist")
        if s.stats is None or s.stats.loss_sum.device != s.device:
            with torch.inference_mode(False):
                s.stats = _StatsBuffers(
                    loss_sum=torch.zeros((1,), device=s.device, dtype=torch.float32),
                    loss_count=torch.zeros((1,), device=s.device, dtype=torch.int64),
                )

    def _require_hidden_layout(self) -> tuple[torch.device, torch.dtype, int]:
        s = self.state
        if s.hidden_size <= 0:
            raise RuntimeError("ESamp engine requires ensure_resources before hidden metadata is available")
        return s.device, s.hidden_dtype, s.hidden_size

    def _require_shared(self) -> _SharedTrainResources:
        if (shared := self.state.shared) is None:
            raise RuntimeError("ESamp shared resources are unavailable before ensure_resources initializes them")
        return shared

    def _require_pipeline(self) -> _PipelineBuffers:
        if (pipeline := self.state.pipeline) is None:
            raise RuntimeError("ESamp pipeline buffers are unavailable before ensure_resources initializes them")
        return pipeline

    def _require_stats(self) -> _StatsBuffers:
        if (stats := self.state.stats) is None:
            raise RuntimeError("ESamp stats buffers are unavailable before ensure_resources initializes them")
        return stats

    def _require_model_bank_train_resources(self) -> tuple[_ModelBankParams, torch.optim.Optimizer]:
        s = self.state
        params = s.model_bank
        optimizer = s.model_bank_optimizer
        if params is None:
            raise RuntimeError("ESamp model bank forward requires initialized model bank parameter tensors")
        if optimizer is None:
            raise RuntimeError("ESamp model bank optimizer is unavailable before ensure_resources initializes it")
        return params, optimizer

    def _require_launch_device(self, hidden: torch.Tensor, *, expected: torch.device) -> None:
        if hidden.device != expected:
            raise RuntimeError(
                f"ESamp launch device mismatch: hidden rows are on {hidden.device}, but runtime resources are on {expected}"
            )

    def _require_prompt_rows(self, prompt_idxs: Sequence[int], active_rows: int) -> list[tuple[int, int]]:
        rows: list[tuple[int, int]] = []
        for row_i in range(max(0, int(active_rows))):
            if row_i >= len(prompt_idxs) or int(prompt_idxs[row_i]) < 0:
                raise RuntimeError("ESamp per-request training requires prompt metadata for every active row")
            rows.append((row_i, int(prompt_idxs[row_i])))
        return rows

    @staticmethod
    def _count_active_model_bank_slots(slot_ids: torch.Tensor, active_rows: int) -> int:
        active = slot_ids[: max(0, int(active_rows))]
        return max(1, int(torch.unique(active, sorted=False).numel()))

    def _model_bank_slot_capacity(self) -> int:
        if self.state.model_bank is not None:
            return int(self.state.model_bank.a.shape[0])
        configured = int(self.state.model_bank_slots)
        return configured if configured > 0 else int(self.state.pipeline_slots)

    def _ensure_per_request_entry(self, prompt_idx: int) -> _PerRequestEntry:
        s = self.state
        key = int(prompt_idx)
        entry = s.per_request.get(key)
        if entry is not None:
            return entry
        device = self._require_hidden_layout()[0]
        with torch.inference_mode(False):
            entry = _PerRequestEntry(
                loss_sum=torch.zeros((1,), device=device, dtype=torch.float32),
                loss_count=torch.zeros((1,), device=device, dtype=torch.int64),
            )
        s.per_request[key] = entry
        return entry

    def _ensure_per_request_trainer(self, prompt_idx: int) -> _PerRequestTrainer:
        s = self.state
        key = int(prompt_idx)
        if (trainer := s.per_request_trainers.get(key)) is not None:
            return trainer
        device, hidden_dtype, hidden_size = self._require_hidden_layout()
        stats = self._ensure_per_request_entry(key)
        with _grad_context():
            train_model, forward_model = self._build_model_pair(
                hidden_size=hidden_size,
                device=device,
                hidden_dtype=hidden_dtype,
            )
            trainer = _PerRequestTrainer(
                train_model=train_model,
                forward_model=forward_model,
                optimizer=_make_optimizer(train_model.parameters(), lr=s.lr, capturable=False),
                stats=stats,
            )
        s.per_request_trainers[key] = trainer
        return trainer

    def _ensure_shared_models(self) -> None:
        s = self.state
        if s.per_request_models:
            raise RuntimeError("ESamp shared models are inactive when per_request_models=True")
        if s.shared is not None:
            return
        device, hidden_dtype, hidden_size = self._require_hidden_layout()
        with _grad_context():
            train_model, forward_model = self._build_model_pair(
                hidden_size=hidden_size,
                device=device,
                hidden_dtype=hidden_dtype,
            )
            optimizer = _make_optimizer(
                train_model.parameters(),
                lr=s.lr,
                capturable=bool(s.model_bank_train_cudagraph),
            )
            s.shared = _SharedTrainResources(
                forward_model=forward_model,
                train_model=train_model,
                optimizer=optimizer,
            )

    def _ensure_model_bank_resources(self) -> None:
        s = self.state
        if not self.using_model_bank:
            raise RuntimeError("ESamp model bank resources are inactive unless model bank mode is enabled")
        device, hidden_dtype, hidden_size = self._require_hidden_layout()
        slots = int(s.model_bank_slots) if int(s.model_bank_slots) > 0 else int(s.pipeline_slots)
        rank = max(1, min(s.model_bank_rank, hidden_size))
        params = s.model_bank
        need_new = (
            params is None
            or tuple(params.a.shape) != (slots, hidden_size, rank)
            or params.a.device != device
            or params.a.dtype != hidden_dtype
        )
        if not need_new:
            return
        with _grad_context():
            a = torch.nn.Parameter(torch.randn((slots, hidden_size, rank), device=device, dtype=hidden_dtype) * 0.02)
            g = torch.nn.Parameter(torch.randn((slots, hidden_size, rank), device=device, dtype=hidden_dtype) * 0.02)
            b = torch.nn.Parameter(torch.randn((slots, rank, hidden_size), device=device, dtype=hidden_dtype) * 0.02)
            gate_bias = torch.nn.Parameter(torch.ones((slots, rank), device=device, dtype=hidden_dtype))
            out_ln_weight = torch.nn.Parameter(torch.ones((slots, hidden_size), device=device, dtype=hidden_dtype))
            out_ln_bias = torch.nn.Parameter(torch.zeros((slots, hidden_size), device=device, dtype=hidden_dtype))
            params: list[torch.nn.Parameter] = [a, g, b, gate_bias]
            if s.model_bank_use_output_layernorm:
                params.extend([out_ln_weight, out_ln_bias])
            s.model_bank = _ModelBankParams(
                a=a,
                g=g,
                b=b,
                gate_bias=gate_bias,
                out_ln_weight=out_ln_weight,
                out_ln_bias=out_ln_bias,
            )
            s.model_bank_optimizer = _make_model_bank_optimizer(
                params,
                lr=s.lr,
                capturable=bool(s.model_bank_train_cudagraph),
            )
            s.model_bank_rank_effective = rank
        self._ensure_sampling_lookup_storage()
        self._reset_runtime_cache()

    def _ensure_sampling_lookup_storage(self) -> None:
        s = self.state
        if s.hidden_size <= 0:
            return
        rows = int(self._require_pipeline().src.shape[1])
        capacity = _sampling_lookup_capacity(
            slots=self._model_bank_slot_capacity(),
            rows=rows,
        )
        if (
            s.sampling_lookup_dense is None
            or s.sampling_lookup_dense.device != s.device
            or int(s.sampling_lookup_dense.numel()) < capacity
            or bool(getattr(s.sampling_lookup_dense, "is_inference", lambda: False)())
        ):
            s.sampling_lookup_dense = torch.full((capacity,), fill_value=-1, device=s.device, dtype=torch.long)
        else:
            s.sampling_lookup_dense.fill_(-1)
        s.sampling_lookup_capacity = int(s.sampling_lookup_dense.numel())

    def _model_bank_forward(self, slot_ids: torch.Tensor, src: torch.Tensor, *, require_grad: bool = True) -> torch.Tensor:
        s = self.state
        params = s.model_bank
        if params is None:
            raise RuntimeError("ESamp model bank forward requires initialized model bank parameter tensors")
        backend = select_model_bank_forward_backend(
            s.model_bank_forward_backend,
            require_grad=bool(require_grad),
            device=src.device,
        )
        return backend.forward(
            slot_ids=slot_ids,
            src=src,
            params=params,
            use_output_layernorm=bool(s.model_bank_use_output_layernorm),
        )

    def _ensure_adaptation_streams(self, device: torch.device) -> None:
        s = self.state
        mode = normalize_adaptation_stream_mode(s.adaptation_stream_mode)
        if mode == "serial":
            s.forward_stream = None
            s.train_stream = None
            return
        priority = int(s.adaptation_stream_priority)
        if mode == "single":
            stream = s.forward_stream or s.train_stream or _make_esamp_stream(device, priority=priority)
            s.forward_stream = stream
            s.train_stream = stream
            return
        if s.forward_stream is None or s.forward_stream is s.train_stream:
            s.forward_stream = _make_esamp_stream(device, priority=priority)
        if s.train_stream is None or s.train_stream is s.forward_stream:
            s.train_stream = _make_esamp_stream(device, priority=priority)

    def _ensure_pipeline(self, *, rows: int, hidden_size: int, device: torch.device, hidden_dtype: torch.dtype) -> None:
        s = self.state
        slots = int(s.pipeline_slots)
        want_shape = (slots, rows, hidden_size)
        pipeline = s.pipeline
        if (
            pipeline is None
            or tuple(pipeline.src.shape) != want_shape
            or pipeline.src.device != device
            or pipeline.src.dtype != hidden_dtype
        ):
            self._reset_pipeline_runtime_state(reset_streams=False)
            self._ensure_adaptation_streams(device)
            s.src_ready_events = _ensure_events([], slots)
            s.tgt_ready_events = _ensure_events([], slots)
            s.src_staged_events = _ensure_events([], slots)
            s.tgt_done_events = _ensure_events([], slots)
            s.slot_train_done_events = _ensure_events([], slots)
            with torch.inference_mode(False):
                s.pipeline = _PipelineBuffers(
                    src=torch.empty(want_shape, device=device, dtype=hidden_dtype),
                    tgt=torch.empty(want_shape, device=device, dtype=hidden_dtype),
                )
            self._reset_runtime_cache()
            return
        self._ensure_adaptation_streams(device)
        s.src_ready_events = _ensure_events(s.src_ready_events, slots)
        s.tgt_ready_events = _ensure_events(s.tgt_ready_events, slots)
        s.src_staged_events = _ensure_events(s.src_staged_events, slots)
        s.tgt_done_events = _ensure_events(s.tgt_done_events, slots)
        s.slot_train_done_events = _ensure_events(s.slot_train_done_events, slots)

    def ensure_resources(
        self,
        *,
        device: torch.device,
        rows: int,
        hidden_size: int,
        hidden_dtype: torch.dtype,
    ) -> None:
        s = self.state
        device_changed = s.device != device
        layout_changed = (
            device_changed
            or s.hidden_size != int(hidden_size)
            or s.hidden_dtype != hidden_dtype
        )
        if device_changed:
            self._reset_pipeline_runtime_state(reset_streams=True)
        s.device = device
        s.hidden_size = int(hidden_size)
        s.hidden_dtype = hidden_dtype
        if layout_changed:
            s.per_request.clear()
            s.per_request_trainers.clear()
            s.shared = None
            s.model_bank_optimizer = None
            s.model_bank = None
            self._reset_runtime_cache()
        self._ensure_pipeline(rows=rows, hidden_size=hidden_size, device=device, hidden_dtype=hidden_dtype)
        self._ensure_stats()
        if not s.per_request_models:
            self._ensure_shared_models()
        if self.using_model_bank:
            self._ensure_model_bank_resources()

    def _graph(self, mode: _Mode) -> _GraphState:
        return self.state.graphs[mode]

    def _model_bank_slot_graph(self, slot: int) -> _GraphState:
        key = int(slot)
        graph = self.state.model_bank_slot_graphs.get(key)
        if graph is None:
            graph = _GraphState()
            self.state.model_bank_slot_graphs[key] = graph
        return graph

    def _graph_enabled(self, mode: _Mode) -> bool:
        s = self.state
        graph = self._graph(mode)
        stream_ok = normalize_adaptation_stream_mode(s.adaptation_stream_mode) == "serial" or s.train_stream is not None
        shared_ok = (
            mode == "shared"
            and not s.per_request_models
            and stream_ok
            and s.shared is not None
        )
        bank_ok = (
            mode == "model_bank"
            and self.using_model_bank
            and int(s.model_bank_flush_interval) == 1
            and stream_ok
            and s.model_bank_optimizer is not None
        )
        return s.model_bank_train_cudagraph and graph.capture_state != "disabled" and (shared_ok or bank_ok)

    def _ensure_graph_storage(
        self,
        mode: _Mode,
        *,
        rows: int,
        hidden: int,
        device: torch.device,
        dtype: torch.dtype,
        slots: int = 0,
        graph: _GraphState | None = None,
        src_input: torch.Tensor | None = None,
        tgt_input: torch.Tensor | None = None,
    ) -> _GraphState:
        graph = graph or self._graph(mode)
        if (src_input is None) != (tgt_input is None):
            raise RuntimeError("external graph inputs require both source and target tensors")
        external_inputs = src_input is not None
        if graph.rows == rows and graph.slots == slots and graph.external_inputs == external_inputs and graph.buffers:
            return graph
        graph.rows = rows
        graph.slots = slots
        graph.active_rows = 0
        graph.external_inputs = external_inputs
        graph.capture_state = "uncaptured"
        graph.replay_graph = _MISSING_REPLAY_GRAPH
        graph.disable_reason = ""
        graph.capture_attempt_count = 0
        graph.skip_not_enabled_count = 0
        graph.skip_missing_optimizer_state_count = 0
        graph.skip_wrong_device_count = 0
        graph.replay_attempt_count = 0
        graph.replay_hit_count = 0
        graph.replay_stage_miss_count = 0
        graph.kernel_fallback_count = 0
        with torch.inference_mode(False):
            if src_input is None:
                src_buffer = torch.zeros((rows, hidden), device=device, dtype=dtype)
                tgt_buffer = torch.zeros((rows, hidden), device=device, dtype=dtype)
            else:
                assert tgt_input is not None
                src_buffer = src_input[:rows]
                tgt_buffer = tgt_input[:rows]
            graph.buffers = {
                "src": src_buffer,
                "tgt": tgt_buffer,
                "valid": torch.zeros((rows,), device=device, dtype=torch.float32),
                "loss": torch.zeros((1,), device=device, dtype=torch.float32),
            }
            if mode == "model_bank":
                graph.buffers["slot_ids"] = torch.zeros((rows,), device=device, dtype=torch.long)
                graph.buffers["slot_sum"] = torch.zeros((slots,), device=device, dtype=torch.float32)
                graph.buffers["slot_cnt"] = torch.zeros((slots,), device=device, dtype=torch.float32)
        return graph

    def _capture_graph(
        self,
        mode: _Mode,
        *,
        rows: int | None = None,
        graph: _GraphState | None = None,
        src_input: torch.Tensor | None = None,
        tgt_input: torch.Tensor | None = None,
    ) -> bool:
        s = self.state
        graph = graph or self._graph(mode)
        graph.capture_attempt_count += 1
        if not self._graph_enabled(mode):
            graph.skip_not_enabled_count += 1
            return False
        if s.device.type != "cuda":
            graph.skip_wrong_device_count += 1
            return False
        pipeline = self._require_pipeline()
        train_stream = s.train_stream or torch.cuda.current_stream(device=s.device)
        if graph.capture_state == "captured":
            return True
        if mode == "shared":
            shared = self._require_shared()
            hidden_dtype = self._require_hidden_layout()[1]
            if _optimizer_requires_state_for_capture(shared.optimizer) and len(shared.optimizer.state) <= 0:
                graph.skip_missing_optimizer_state_count += 1
                return False
            graph = self._ensure_graph_storage(
                "shared",
                rows=int(rows) if rows is not None else int(pipeline.src.shape[1]),
                hidden=int(pipeline.src.shape[2]),
                device=s.device,
                dtype=hidden_dtype,
                graph=graph,
            )
        else:
            model_bank, model_bank_optimizer = self._require_model_bank_train_resources()
            if _optimizer_requires_state_for_capture(model_bank_optimizer) and len(model_bank_optimizer.state) <= 0:
                graph.skip_missing_optimizer_state_count += 1
                return False
            graph = self._ensure_graph_storage(
                "model_bank",
                rows=int(rows) if rows is not None else int(pipeline.src.shape[1]),
                hidden=int(model_bank.a.shape[1]),
                slots=int(model_bank.a.shape[0]),
                device=s.device,
                dtype=model_bank.a.dtype,
                graph=graph,
                src_input=src_input,
                tgt_input=tgt_input,
            )
        try:
            graph_obj = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                with torch.cuda.graph(graph_obj, stream=train_stream):
                    with _grad_context():
                        if mode == "shared":
                            shared.optimizer.zero_grad(set_to_none=False)
                            pred = shared.train_model(graph.buffers["src"])
                            row_mse = ((pred.float() - graph.buffers["tgt"].float()) ** 2).mean(dim=1)
                            loss = (row_mse * graph.buffers["valid"]).sum() / torch.clamp(graph.buffers["valid"].sum(), min=1.0)
                            loss.backward()
                            shared.optimizer.step()
                            _copy_parameters(shared.forward_model, shared.train_model)
                        else:
                            model_bank_optimizer.zero_grad(set_to_none=False)
                            pred = self._model_bank_forward(graph.buffers["slot_ids"], graph.buffers["src"])
                            row_mse = ((pred.float() - graph.buffers["tgt"].float()) ** 2).mean(dim=1)
                            weighted = row_mse * graph.buffers["valid"]
                            graph.buffers["slot_sum"].zero_()
                            graph.buffers["slot_cnt"].zero_()
                            graph.buffers["slot_sum"].scatter_add_(0, graph.buffers["slot_ids"], weighted)
                            graph.buffers["slot_cnt"].scatter_add_(0, graph.buffers["slot_ids"], graph.buffers["valid"])
                            slot_mean = graph.buffers["slot_sum"] / torch.clamp(graph.buffers["slot_cnt"], min=1.0)
                            active = (graph.buffers["slot_cnt"] > 0).to(dtype=slot_mean.dtype)
                            loss = (slot_mean * active).sum()
                            loss.backward()
                            model_bank_optimizer.step()
                        graph.buffers["loss"].copy_(loss.detach())
            graph.capture_state = "captured"
            graph.replay_graph = graph_obj
            graph.disable_reason = ""
            return True
        except Exception as exc:
            graph.capture_state = "disabled"
            graph.replay_graph = _MISSING_REPLAY_GRAPH
            graph.disable_reason = str(exc)
            return False

    def _stage_graph_inputs(
        self,
        mode: _Mode,
        *,
        src: torch.Tensor,
        tgt: torch.Tensor,
        active_rows: int,
        slot_ids: torch.Tensor | None = None,
        graph: _GraphState | None = None,
    ) -> bool:
        graph = graph or self._graph(mode)
        if not graph.buffers:
            return False
        n = int(active_rows)
        if n <= 0 or n > graph.rows:
            return False
        previous_active = max(0, min(int(graph.active_rows), int(graph.buffers["valid"].numel())))
        if previous_active > 0:
            graph.buffers["valid"][:previous_active].zero_()
        if not graph.external_inputs:
            graph.buffers["src"][:n].copy_(src[:n])
            graph.buffers["tgt"][:n].copy_(tgt[:n])
        graph.buffers["valid"][:n].fill_(1.0)
        if mode == "model_bank":
            if slot_ids is None or "slot_ids" not in graph.buffers:
                return False
            graph.buffers["slot_ids"][:n].copy_(slot_ids[:n])
        graph.active_rows = n
        return True

    def prepare_model_bank_initializer(
        self,
        *,
        model: object,
        target_layer: torch.nn.Module,
        target_resolved: str,
        hidden_size: int,
    ) -> None:
        initializer = self.state.model_bank_initializer
        if initializer is None:
            return
        initializer.prepare_from_model(
            engine=self,
            model=model,
            target_layer=target_layer,
            target_resolved=target_resolved,
            hidden_size=hidden_size,
        )

    # Assigns persistent prompt-to-slot state and invalidates lookup caches.
    def _assign_model_bank_slot(self, prompt_idx: int) -> int:
        s = self.state
        key = int(prompt_idx)
        existing = s.prompt_to_slot.get(key)
        if existing is not None:
            if s.model_bank_initializer is not None:
                s.model_bank_initializer.ensure_existing_slot(self, int(existing))
            return existing
        if s.model_bank_next_slot >= self._model_bank_slot_capacity():
            raise RuntimeError("ESamp engine model bank slots exhausted")
        slot = int(s.model_bank_next_slot)
        s.prompt_to_slot[key] = slot
        s.slot_to_prompt[slot] = key
        s.model_bank_next_slot += 1
        s.sampling_lookup_device = None
        s.sampling_lookup_size = -1
        s.sampling_lookup_keys = None
        s.sampling_lookup_values = None
        self._ensure_per_request_entry(key)
        if s.model_bank_initializer is not None:
            s.model_bank_initializer.on_slot_assigned(self, slot)
        return slot

    def _write_sampling_lookup_dense_entries(
        self,
        prompt_idxs: Sequence[int],
        slot_idxs: Sequence[int],
    ) -> None:
        s = self.state
        dense_lookup = s.sampling_lookup_dense
        if dense_lookup is None or not prompt_idxs:
            return
        if bool(getattr(dense_lookup, "is_inference", lambda: False)()):
            dense_lookup = dense_lookup.clone()
            s.sampling_lookup_dense = dense_lookup
        capacity = int(dense_lookup.numel())
        valid_prompt_idxs: list[int] = []
        valid_slot_idxs: list[int] = []
        for prompt_idx, slot_idx in zip(prompt_idxs, slot_idxs):
            prompt = int(prompt_idx)
            if 0 <= prompt < capacity:
                valid_prompt_idxs.append(prompt)
                valid_slot_idxs.append(int(slot_idx))
        if not valid_prompt_idxs:
            return
        prompt_tensor = torch.as_tensor(valid_prompt_idxs, device=dense_lookup.device, dtype=torch.long)
        slot_tensor = torch.as_tensor(valid_slot_idxs, device=dense_lookup.device, dtype=dense_lookup.dtype)
        dense_lookup.index_copy_(0, prompt_tensor, slot_tensor)

    def _model_bank_slot_tensor(self, slot_ids: Sequence[int], device: torch.device) -> torch.Tensor:
        key = tuple(int(slot_id) for slot_id in slot_ids)
        cached = self.state.model_bank_slot_tensor_cache.get(key)
        if cached is not None and cached.device == device:
            return cached
        tensor = torch.as_tensor(key, device=device, dtype=torch.long)
        self.state.model_bank_slot_tensor_cache[key] = tensor
        return tensor

    def assign_sampling_model_bank_slots(self, prompt_idxs: torch.Tensor) -> bool:
        s = self.state
        if not self.using_model_bank:
            return not s.per_request_models
        self._ensure_sampling_lookup_storage()
        dense_lookup = s.sampling_lookup_dense
        if dense_lookup is None:
            return False
        prompt_tensor = prompt_idxs.detach()
        if prompt_tensor.device.type == "cpu":
            prompt_list = [int(x) for x in prompt_tensor.tolist()]
        else:
            prompt_list = [int(x) for x in prompt_tensor.cpu().tolist()]
        supported = True
        prompt_updates: list[int] = []
        slot_updates: list[int] = []
        for prompt_idx in prompt_list:
            if prompt_idx < 0:
                continue
            slot = self._assign_model_bank_slot(prompt_idx)
            prompt_updates.append(int(prompt_idx))
            slot_updates.append(int(slot))
            if prompt_idx >= int(dense_lookup.numel()):
                supported = False
                continue
        self._write_sampling_lookup_dense_entries(prompt_updates, slot_updates)
        return supported

    def _model_bank_slot_lookup_for_device(self, device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor] | None:
        s = self.state
        if not s.prompt_to_slot:
            return None
        dense_lookup = s.sampling_lookup_dense if (
            s.sampling_lookup_dense is not None and s.sampling_lookup_dense.device == device
        ) else None
        if (
            s.sampling_lookup_device == device
            and s.sampling_lookup_size == len(s.prompt_to_slot)
            and s.sampling_lookup_keys is not None
            and s.sampling_lookup_values is not None
        ):
            return dense_lookup, s.sampling_lookup_keys, s.sampling_lookup_values
        key_tensor = torch.as_tensor(list(s.prompt_to_slot.keys()), device=device, dtype=torch.long)
        value_tensor = torch.as_tensor(list(s.prompt_to_slot.values()), device=device, dtype=torch.long)
        sort_idx = torch.argsort(key_tensor)
        s.sampling_lookup_device = device
        s.sampling_lookup_size = len(s.prompt_to_slot)
        s.sampling_lookup_keys = key_tensor.index_select(0, sort_idx)
        s.sampling_lookup_values = value_tensor.index_select(0, sort_idx)
        return dense_lookup, s.sampling_lookup_keys, s.sampling_lookup_values

    def _train_shared_kernel(
        self,
        *,
        shared: _SharedTrainResources,
        stats: _StatsBuffers,
        src: torch.Tensor,
        tgt: torch.Tensor,
        active_rows: int,
    ) -> None:
        pred = shared.train_model(src[:active_rows])
        loss = F.mse_loss(pred.float(), tgt[:active_rows].float())
        shared.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        shared.optimizer.step()
        _copy_parameters(shared.forward_model, shared.train_model)
        stats.loss_sum.add_(loss.detach())
        stats.loss_count.add_(1)

    def _maybe_run_shared_graph(self, *, src: torch.Tensor, tgt: torch.Tensor, active_rows: int) -> bool:
        if not self._graph_enabled("shared"):
            return False
        self._ensure_graph_storage("shared", rows=int(src.shape[0]), hidden=int(src.shape[1]), device=src.device, dtype=src.dtype)
        return self._graph("shared").capture_state == "captured" and self._replay_graph_with_disable_fallback(
            "shared",
            src=src,
            tgt=tgt,
            active_rows=active_rows,
            count_delta=1,
        )

    def _train_shared(self, src: torch.Tensor, tgt: torch.Tensor, active_rows: int) -> None:
        k = int(active_rows)
        if k <= 0:
            return
        shared = self._require_shared()
        stats = self._require_stats()
        if self._maybe_run_shared_graph(src=src, tgt=tgt, active_rows=k):
            return
        self._train_shared_kernel(shared=shared, stats=stats, src=src, tgt=tgt, active_rows=k)
        if self._graph_enabled("shared") and self._graph("shared").capture_state == "uncaptured":
            self._capture_graph("shared")

    def _train_per_request_step(
        self,
        *,
        stats: _StatsBuffers,
        trainer: _PerRequestTrainer,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> None:
        pred = trainer.train_model(src)
        loss = F.mse_loss(pred.float(), tgt.float())
        trainer.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainer.optimizer.step()
        _copy_parameters(trainer.forward_model, trainer.train_model)
        stats.loss_sum.add_(loss.detach())
        stats.loss_count.add_(1)
        trainer.stats.loss_sum.add_(loss.detach())
        trainer.stats.loss_count.add_(1)
        if self.state.trace_per_request_losses:
            _record_trace_sample(
                trainer.stats,
                loss,
                interval=self.state.trace_interval,
                max_points=self.state.trace_max_points,
            )

    def _train_per_request(
        self,
        src_buf: torch.Tensor,
        tgt_buf: torch.Tensor,
        active_rows: int,
        prompt_idxs: Sequence[int],
    ) -> None:
        stats = self._require_stats()
        prompt_groups: dict[int, list[int]] = {}
        for row_i, prompt_idx in self._require_prompt_rows(prompt_idxs, active_rows):
            prompt_groups.setdefault(prompt_idx, []).append(row_i)
        for prompt_idx, row_ids in prompt_groups.items():
            ridx = torch.as_tensor(row_ids, device=src_buf.device, dtype=torch.long)
            src = src_buf.index_select(0, ridx)
            tgt = tgt_buf.index_select(0, ridx)
            self._train_per_request_step(
                stats=stats,
                trainer=self._ensure_per_request_trainer(prompt_idx),
                src=src,
                tgt=tgt,
            )

    def _train_model_bank_kernel(
        self,
        *,
        optimizer: torch.optim.AdamW,
        stats: _StatsBuffers,
        slot_tensor: torch.Tensor,
        unique_slot_ids: Sequence[int],
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> None:
        pred = self._model_bank_forward(slot_tensor, src)
        row_mse = ((pred.float() - tgt.float()) ** 2).mean(dim=1)
        if not unique_slot_ids:
            return
        unique_slot_tensor = torch.as_tensor(unique_slot_ids, device=slot_tensor.device, dtype=torch.long)
        slot_pos = torch.searchsorted(unique_slot_tensor, slot_tensor)
        slot_loss_sums = torch.zeros((int(unique_slot_tensor.numel()),), device=row_mse.device, dtype=row_mse.dtype)
        slot_loss_counts = torch.zeros_like(slot_loss_sums)
        slot_loss_sums.index_add_(0, slot_pos, row_mse)
        slot_loss_counts.index_add_(0, slot_pos, torch.ones_like(row_mse))
        slot_losses_tensor = slot_loss_sums / slot_loss_counts.clamp_min(1.0)
        for slot_idx, slot_id in enumerate(unique_slot_ids):
            slot_loss = slot_losses_tensor[int(slot_idx)]
            prompt_idx = self.state.slot_to_prompt[int(slot_id)]
            entry = self.state.per_request[prompt_idx]
            entry.loss_sum.add_(slot_loss.detach())
            entry.loss_count.add_(1)
            if self.state.trace_per_request_losses:
                _record_trace_sample(
                    entry,
                    slot_loss,
                    interval=self.state.trace_interval,
                    max_points=self.state.trace_max_points,
                )
        loss = slot_losses_tensor.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        stats.loss_sum.add_(loss.detach())
        stats.loss_count.add_(int(slot_losses_tensor.numel()))

    def _maybe_run_model_bank_graph(
        self,
        *,
        src: torch.Tensor,
        tgt: torch.Tensor,
        slot_tensor: torch.Tensor,
        active_slot_count: int | None = None,
        pipeline_slot: int | None = None,
    ) -> bool:
        graph = self._model_bank_slot_graph(int(pipeline_slot)) if pipeline_slot is not None else self._graph("model_bank")
        return self._graph_enabled("model_bank") and graph.capture_state == "captured" and self._replay_graph_with_disable_fallback(
            "model_bank",
            src=src,
            tgt=tgt,
            active_rows=int(slot_tensor.numel()),
            count_delta=int(active_slot_count) if active_slot_count is not None else self._count_active_model_bank_slots(slot_tensor, int(slot_tensor.numel())),
            slot_ids=slot_tensor,
            graph=graph,
        )

    def _train_model_bank(
        self,
        src_buf: torch.Tensor,
        tgt_buf: torch.Tensor,
        active_rows: int,
        prompt_idxs: Sequence[int],
        pipeline_slot: int | None = None,
    ) -> None:
        s = self.state
        stats = self._require_stats()
        _, model_bank_optimizer = self._require_model_bank_train_resources()
        prompt_rows = self._require_prompt_rows(prompt_idxs, active_rows)
        if not prompt_rows:
            return
        selected_rows: list[int] = []
        selected_slots: list[int] = []
        selected_prompts: list[int] = []
        seen_slots: set[int] = set()
        new_prompt_idxs: list[int] = []
        new_slot_ids: list[int] = []
        for row_i, prompt_idx in prompt_rows:
            had_slot = int(prompt_idx) in s.prompt_to_slot
            slot_id = self._assign_model_bank_slot(prompt_idx)
            if not had_slot:
                new_prompt_idxs.append(int(prompt_idx))
                new_slot_ids.append(int(slot_id))
            if int(slot_id) in seen_slots:
                continue
            seen_slots.add(int(slot_id))
            selected_rows.append(int(row_i))
            selected_slots.append(int(slot_id))
            selected_prompts.append(int(prompt_idx))
        if not selected_rows:
            return
        if new_prompt_idxs:
            self._write_sampling_lookup_dense_entries(new_prompt_idxs, new_slot_ids)
        row_ids = selected_rows
        slot_ids = selected_slots
        unique_slot_ids = sorted({int(slot_id) for slot_id in slot_ids})
        rows_are_prefix = row_ids == list(range(len(row_ids)))
        if rows_are_prefix:
            src = src_buf[: len(row_ids)]
            tgt = tgt_buf[: len(row_ids)]
        else:
            ridx = torch.as_tensor(row_ids, device=src_buf.device, dtype=torch.long)
            src = src_buf.index_select(0, ridx)
            tgt = tgt_buf.index_select(0, ridx)
        slot_tensor = self._model_bank_slot_tensor(slot_ids, src_buf.device)
        if s.model_bank_initializer is not None:
            s.model_bank_initializer.maybe_prepare_slots(self, slot_tensor, src, tgt)
        if self._maybe_run_model_bank_graph(
            src=src,
            tgt=tgt,
            slot_tensor=slot_tensor,
            active_slot_count=len(unique_slot_ids),
            pipeline_slot=int(pipeline_slot) if rows_are_prefix and pipeline_slot is not None else None,
        ):
            return
        self._graph("model_bank").kernel_fallback_count += 1
        self._train_model_bank_kernel(
            optimizer=model_bank_optimizer,
            stats=stats,
            slot_tensor=slot_tensor,
            unique_slot_ids=unique_slot_ids,
            src=src,
            tgt=tgt,
        )
        if self._graph_enabled("model_bank"):
            graph = self._model_bank_slot_graph(int(pipeline_slot)) if rows_are_prefix and pipeline_slot is not None else self._graph("model_bank")
            if graph.capture_state == "uncaptured":
                self._capture_graph(
                    "model_bank",
                    rows=int(slot_tensor.numel()),
                    graph=graph,
                    src_input=src if rows_are_prefix and pipeline_slot is not None else None,
                    tgt_input=tgt if rows_are_prefix and pipeline_slot is not None else None,
                )

    def _train_slot(self, slot: int, active_rows: int, prompt_idxs: Sequence[int]) -> None:
        pipeline = self._require_pipeline()
        src_buf = pipeline.src[int(slot)]
        tgt_buf = pipeline.tgt[int(slot)]
        if self.using_model_bank:
            self._train_model_bank(src_buf, tgt_buf, active_rows, prompt_idxs, pipeline_slot=int(slot))
        elif self.state.per_request_models:
            self._train_per_request(src_buf, tgt_buf, active_rows, prompt_idxs)
        else:
            self._train_shared(src_buf, tgt_buf, active_rows)

    def _train_external(self, src: torch.Tensor, tgt: torch.Tensor, active_rows: int, prompt_idxs: Sequence[int]) -> None:
        k = int(active_rows)
        if k <= 0:
            return
        src_view = src[:k]
        tgt_view = tgt[:k]
        if self.using_model_bank:
            self._train_model_bank(src_view, tgt_view, k, prompt_idxs, pipeline_slot=int(src_view.data_ptr()))
        elif self.state.per_request_models:
            self._train_per_request(src_view, tgt_view, k, prompt_idxs)
        else:
            self._train_shared(src_view, tgt_view, k)

    def _launch_forward_kernel(
        self,
        *,
        pipeline: _PipelineBuffers,
        forward_stream: torch.cuda.Stream | None,
        slot: int,
        source_hidden: torch.Tensor,
    ) -> None:
        s = self.state
        src_ready = s.src_ready_events[slot]
        src_staged = s.src_staged_events[slot]
        done_event = s.slot_train_done_events[slot]
        current_stream = torch.cuda.current_stream(device=source_hidden.device)
        src_ready.record(current_stream)
        if forward_stream is None:
            current_stream.wait_event(done_event)
            _copy_rows(pipeline.src[slot], source_hidden)
            src_staged.record(current_stream)
            return
        with torch.cuda.stream(forward_stream):
            forward_stream.wait_event(done_event)
            forward_stream.wait_event(src_ready)
            _copy_rows(pipeline.src[slot], source_hidden)
            src_staged.record(forward_stream)

    def _launch_step_kernel(
        self,
        *,
        pipeline: _PipelineBuffers,
        slot: int,
        source_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
    ) -> None:
        s = self.state
        if int(source_hidden.shape[0]) != int(target_hidden.shape[0]):
            raise RuntimeError("ESamp step launch requires source and target batches with the same row count")
        current_stream = torch.cuda.current_stream(device=source_hidden.device)
        current_stream.wait_event(s.slot_train_done_events[slot])
        _copy_rows(pipeline.src[slot], source_hidden)
        _copy_rows(pipeline.tgt[slot], target_hidden)
        s.src_staged_events[slot].record(current_stream)
        s.tgt_done_events[slot].record(current_stream)

    def launch_forward(self, source_hidden: torch.Tensor) -> None:
        s = self.state
        if not s.enabled:
            return
        if s.current_step_slot is not None or s.ready_step_slot is not None or s.pending_train_queue:
            raise RuntimeError("ESamp launch order violation: previous step has not been flushed")
        pipeline = self._require_pipeline()
        self._require_launch_device(source_hidden, expected=pipeline.src.device)
        forward_stream = s.forward_stream
        if forward_stream is None and normalize_adaptation_stream_mode(s.adaptation_stream_mode) != "serial":
            raise RuntimeError("ESamp forward stream is unavailable before ensure_resources initializes it")
        slot = int(s.next_pipeline_slot % int(s.pipeline_slots))
        s.next_pipeline_slot = (slot + 1) % int(s.pipeline_slots)
        s.current_step_slot = slot
        s.ready_step_slot = None
        self._launch_forward_kernel(
            pipeline=pipeline,
            forward_stream=forward_stream,
            slot=slot,
            source_hidden=source_hidden,
        )

    def _launch_target_kernel(
        self,
        *,
        pipeline: _PipelineBuffers,
        forward_stream: torch.cuda.Stream | None,
        slot: int,
        target_hidden: torch.Tensor,
    ) -> None:
        s = self.state
        tgt_ready = s.tgt_ready_events[slot]
        tgt_done = s.tgt_done_events[slot]
        current_stream = torch.cuda.current_stream(device=target_hidden.device)
        tgt_ready.record(current_stream)
        if forward_stream is None:
            _copy_rows(pipeline.tgt[slot], target_hidden)
            tgt_done.record(current_stream)
            s.current_step_slot = None
            s.ready_step_slot = slot
            return
        with torch.cuda.stream(forward_stream):
            forward_stream.wait_event(tgt_ready)
            _copy_rows(pipeline.tgt[slot], target_hidden)
            tgt_done.record(forward_stream)
        s.current_step_slot = None
        s.ready_step_slot = slot

    def launch_target(self, target_hidden: torch.Tensor) -> None:
        s = self.state
        if not s.enabled:
            return
        pipeline = self._require_pipeline()
        self._require_launch_device(target_hidden, expected=pipeline.tgt.device)
        forward_stream = s.forward_stream
        if forward_stream is None and normalize_adaptation_stream_mode(s.adaptation_stream_mode) != "serial":
            raise RuntimeError("ESamp forward stream is unavailable before ensure_resources initializes it")
        slot = s.current_step_slot
        if slot is None:
            raise RuntimeError("ESamp target launch requires an active pipeline slot")
        self._launch_target_kernel(
            pipeline=pipeline,
            forward_stream=forward_stream,
            slot=int(slot),
            target_hidden=target_hidden,
        )

    def launch_step(self, source_hidden: torch.Tensor, target_hidden: torch.Tensor) -> None:
        s = self.state
        if not s.enabled:
            return
        if s.current_step_slot is not None or s.ready_step_slot is not None or s.pending_train_queue:
            raise RuntimeError("ESamp launch order violation: previous step has not been flushed")
        pipeline = self._require_pipeline()
        self._require_launch_device(source_hidden, expected=pipeline.src.device)
        self._require_launch_device(target_hidden, expected=pipeline.tgt.device)
        slot = int(s.next_pipeline_slot % int(s.pipeline_slots))
        s.next_pipeline_slot = (slot + 1) % int(s.pipeline_slots)
        self._launch_step_kernel(
            pipeline=pipeline,
            slot=slot,
            source_hidden=source_hidden,
            target_hidden=target_hidden,
        )
        s.current_step_slot = None
        s.ready_step_slot = slot

    def launch_delayed_backward(self, active_rows: int, prompt_idxs: Sequence[int] | None = None) -> None:
        s = self.state
        if not s.enabled:
            return
        pipeline = self._require_pipeline()
        serial = normalize_adaptation_stream_mode(s.adaptation_stream_mode) == "serial"
        train_stream = s.train_stream
        if train_stream is None and not serial:
            raise RuntimeError("ESamp train stream is unavailable before ensure_resources initializes it")
        if (slot := s.ready_step_slot) is not None:
            s.pending_train_queue.append((int(slot), int(active_rows), tuple(int(x) for x in (prompt_idxs or ()))))
            s.ready_step_slot = None
        if serial:
            current_stream = torch.cuda.current_stream(device=pipeline.src.device)
            while s.pending_train_queue:
                queued_slot, queued_rows, queued_prompts = s.pending_train_queue.popleft()
                current_stream.wait_event(s.src_staged_events[queued_slot])
                current_stream.wait_event(s.tgt_done_events[queued_slot])
                with _grad_context():
                    self._train_slot(queued_slot, queued_rows, queued_prompts)
                s.slot_train_done_events[queued_slot].record(current_stream)
            return
        assert train_stream is not None
        with torch.cuda.stream(train_stream):
            while s.pending_train_queue:
                queued_slot, queued_rows, queued_prompts = s.pending_train_queue.popleft()
                train_stream.wait_event(s.src_staged_events[queued_slot])
                train_stream.wait_event(s.tgt_done_events[queued_slot])
                with _grad_context():
                    self._train_slot(queued_slot, queued_rows, queued_prompts)
                s.slot_train_done_events[queued_slot].record(train_stream)

    def launch_external_backward(
        self,
        source_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        active_rows: int,
        *,
        prompt_idxs: Sequence[int] | None = None,
        wait_ready: object | None = None,
        release: object | None = None,
    ) -> None:
        s = self.state
        if not s.enabled:
            return
        if int(source_hidden.shape[0]) != int(target_hidden.shape[0]):
            raise RuntimeError("ESamp external backward requires source/target hidden batches with the same row count")
        serial = normalize_adaptation_stream_mode(s.adaptation_stream_mode) == "serial"
        train_stream = s.train_stream
        if train_stream is None and not serial:
            raise RuntimeError("ESamp train stream is unavailable before ensure_resources initializes it")
        if serial:
            current_stream = torch.cuda.current_stream(device=source_hidden.device)
            try:
                if callable(wait_ready):
                    wait_ready(current_stream)
                with _grad_context():
                    self._train_external(source_hidden, target_hidden, int(active_rows), tuple(int(x) for x in (prompt_idxs or ())))
            finally:
                if callable(release):
                    release(current_stream)
            return
        assert train_stream is not None
        with torch.cuda.stream(train_stream):
            try:
                if callable(wait_ready):
                    wait_ready(train_stream)
                with _grad_context():
                    self._train_external(source_hidden, target_hidden, int(active_rows), tuple(int(x) for x in (prompt_idxs or ())))
            finally:
                if callable(release):
                    release(train_stream)

    def synchronize(self) -> None:
        s = self.state
        if s.forward_stream is not None:
            s.forward_stream.synchronize()
        if s.train_stream is not None:
            s.train_stream.synchronize()

    def read_and_reset_stats(self, sync: bool = True) -> ESampStats:
        s = self.state
        if sync:
            self.synchronize()
        if s.stats is None:
            if not s.enabled or s.hidden_size <= 0:
                return ESampStats(loss_avg=0.0, loss_count=0)
            raise RuntimeError("ESamp engine stats buffers are missing while the engine is enabled")
        stats = self._require_stats()
        count = int(stats.loss_count.item())
        avg = float(stats.loss_sum.item() / max(1, count))
        s.stats = _StatsBuffers(
            loss_sum=torch.zeros_like(stats.loss_sum),
            loss_count=torch.zeros_like(stats.loss_count),
        )
        return ESampStats(loss_avg=avg, loss_count=count)

    def read_and_reset_per_request_stats(self, sync: bool = True) -> dict[int, ESampStats]:
        if sync:
            self.synchronize()
        out: dict[int, ESampStats] = {}
        for prompt_idx, entry in sorted(self.state.per_request.items()):
            count = int(entry.loss_count.item())
            avg = float(entry.loss_sum.item() / max(1, count))
            out[int(prompt_idx)] = ESampStats(
                loss_avg=avg,
                loss_count=count,
                trace_losses=tuple(entry.trace_losses),
            )
            entry.loss_sum = torch.zeros_like(entry.loss_sum)
            entry.loss_count = torch.zeros_like(entry.loss_count)
            entry.trace_losses = []
            entry.trace_update_count = 0
        return out


_EngineStorage = _EngineState

def group_row_indices_by_prompt(prompt_idxs: Sequence[int], active_rows: int) -> dict[int, list[int]]:
    return _group_rows(prompt_idxs, active_rows)


def copy_active_rows_into_buffer(dst: torch.Tensor, src: torch.Tensor) -> int:
    return _copy_rows(dst, src)


class ESampTrainEngine(_ESampEngineCore):
    """Public ESamp engine surface."""

    def __setattr__(self, name: str, value) -> None:
        if name == "state" and "state" in self.__dict__:
            raise AttributeError("state is not part of the supported mutable ESamp engine surface")
        if "state" not in self.__dict__ or name in self.__dict__ or hasattr(type(self), name):
            object.__setattr__(self, name, value)
            return
        raise AttributeError(f"{name} is not part of the supported ESamp engine surface")
    def _model_bank_forward_locked(self, slot_ids: torch.Tensor, src: torch.Tensor, *, require_grad: bool = True) -> torch.Tensor:
        return super()._model_bank_forward(slot_ids, src, require_grad=bool(require_grad))

    def predict_hidden_for_sampling_capture(
        self,
        source_hidden: torch.Tensor,
        prompt_idxs: torch.Tensor,
        *,
        out_pred_hidden: torch.Tensor,
        out_valid_mask: torch.Tensor,
    ) -> bool:
        if source_hidden.ndim != 2:
            raise ValueError(f"ESamp capture prediction expects rank-2 hidden rows, got shape={tuple(source_hidden.shape)}")
        rows = int(source_hidden.shape[0])
        if rows <= 0 or not self.state.enabled:
            out_valid_mask.zero_()
            return False
        if int(out_valid_mask.numel()) < rows:
            raise ValueError("out_valid_mask is smaller than active source rows")
        if int(out_pred_hidden.shape[0]) < rows or int(out_pred_hidden.shape[1]) != int(source_hidden.shape[1]):
            raise ValueError("out_pred_hidden shape does not match active source rows")
        s = self.state
        with torch.no_grad():
            if self.using_model_bank:
                lookup = s.sampling_lookup_dense
                if lookup is None or int(lookup.numel()) <= 0:
                    out_valid_mask[:rows].zero_()
                    return False
                prompt_tensor = prompt_idxs[:rows].to(device=source_hidden.device, dtype=torch.long)
                valid_prompt = prompt_tensor >= 0
                safe_prompt = prompt_tensor.clamp(min=0, max=max(0, int(lookup.numel()) - 1))
                gathered_slots = lookup.index_select(0, safe_prompt)
                valid_mask = valid_prompt & (prompt_tensor < int(lookup.numel())) & (gathered_slots >= 0)
                slot_tensor = gathered_slots.clamp_min(0)
                out_pred_hidden[:rows].copy_(self._model_bank_forward(slot_tensor, source_hidden[:rows], require_grad=False))
                out_valid_mask[:rows].copy_(valid_mask)
                return True
            if s.per_request_models:
                out_valid_mask[:rows].zero_()
                return False
            shared = self._require_shared()
            out_pred_hidden[:rows].copy_(shared.forward_model(source_hidden[:rows]))
            out_valid_mask[:rows].fill_(True)
            return True

    def predict_hidden_for_sampling(
        self,
        source_hidden: torch.Tensor,
        prompt_idxs: torch.Tensor,
        *,
        assume_all_model_bank_slots_ready: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source_hidden.ndim != 2:
            raise ValueError(f"ESamp sampling prediction expects rank-2 hidden rows, got shape={tuple(source_hidden.shape)}")
        rows = int(source_hidden.shape[0])
        row_device = source_hidden.device
        empty_rows = torch.empty((0,), device=row_device, dtype=torch.long)
        empty_pred = source_hidden.new_empty((0, int(source_hidden.shape[1])))
        if rows <= 0 or not self.state.enabled:
            return empty_rows, empty_pred
        s = self.state
        prompt_tensor = prompt_idxs[:rows].to(device=row_device, dtype=torch.long)
        with torch.no_grad():
            if self.using_model_bank:
                if prompt_tensor.numel() <= 0:
                    return empty_rows, empty_pred
                lookup = self._model_bank_slot_lookup_for_device(row_device)
                if lookup is None:
                    return empty_rows, empty_pred
                dense_lookup, sorted_keys, sorted_values = lookup
                candidate_prompts = prompt_tensor.clamp_min(0)
                if dense_lookup is not None:
                    in_range = candidate_prompts < int(dense_lookup.numel())
                    safe_prompts = candidate_prompts.clamp(max=max(0, int(dense_lookup.numel()) - 1))
                    gathered_slots = dense_lookup.index_select(0, safe_prompts)
                    if assume_all_model_bank_slots_ready:
                        row_ids = torch.arange(rows, device=row_device, dtype=torch.long)
                        pred = self._model_bank_forward(gathered_slots.clamp_min(0), source_hidden[:rows], require_grad=False)
                        return row_ids, pred
                    valid_prompt = prompt_tensor >= 0
                    active_mask = valid_prompt & in_range & (gathered_slots >= 0)
                    row_ids = active_mask.nonzero(as_tuple=False).view(-1)
                    if row_ids.numel() <= 0:
                        return empty_rows, empty_pred
                    slot_tensor = gathered_slots.index_select(0, row_ids)
                else:
                    valid_prompt = prompt_tensor >= 0
                    positions = torch.searchsorted(sorted_keys, candidate_prompts)
                    in_range = positions < int(sorted_keys.numel())
                    safe_positions = positions.clamp(max=max(0, int(sorted_keys.numel()) - 1))
                    matched = in_range & (
                        sorted_keys.index_select(0, safe_positions) == candidate_prompts
                    )
                    active_mask = valid_prompt & matched
                    row_ids = active_mask.nonzero(as_tuple=False).view(-1)
                    if row_ids.numel() <= 0:
                        return empty_rows, empty_pred
                    slot_tensor = sorted_values.index_select(0, safe_positions.index_select(0, row_ids))
                pred = self._model_bank_forward(slot_tensor, source_hidden.index_select(0, row_ids), require_grad=False)
                return row_ids, pred
            if s.per_request_models:
                valid_prompt = prompt_tensor >= 0
                row_chunks: list[torch.Tensor] = []
                pred_chunks: list[torch.Tensor] = []
                for prompt_idx_t in torch.unique(prompt_tensor[valid_prompt], sorted=True):
                    prompt_idx = int(prompt_idx_t.item())
                    trainer = s.per_request_trainers.get(prompt_idx)
                    if trainer is None:
                        continue
                    row_ids = (prompt_tensor == prompt_idx).nonzero(as_tuple=False).view(-1)
                    if row_ids.numel() <= 0:
                        continue
                    row_chunks.append(row_ids)
                    pred_chunks.append(trainer.forward_model(source_hidden.index_select(0, row_ids)))
                if not row_chunks:
                    return empty_rows, empty_pred
                return (torch.cat(row_chunks, dim=0), torch.cat(pred_chunks, dim=0))
            shared = self._require_shared()
            row_ids = torch.arange(rows, device=row_device, dtype=torch.long)
            return row_ids, shared.forward_model(source_hidden)

    def _replay_graph_with_disable_fallback(
        self,
        mode: _Mode,
        *,
        src: torch.Tensor,
        tgt: torch.Tensor,
        active_rows: int,
        count_delta: int,
        slot_ids: torch.Tensor | None = None,
        graph: _GraphState | None = None,
    ) -> bool:
        graph = graph or self._graph(mode)
        if graph.capture_state != "captured":
            return False
        graph.replay_attempt_count += 1
        stats = self._require_stats()
        if not self._stage_graph_inputs(mode, src=src, tgt=tgt, active_rows=active_rows, slot_ids=slot_ids, graph=graph):
            graph.replay_stage_miss_count += 1
            return False
        try:
            with torch.inference_mode():
                assert not isinstance(graph.replay_graph, _ReplayGraphMissing)
                graph.replay_graph.replay()
        except Exception as exc:
            graph.capture_state = "disabled"
            graph.replay_graph = _MISSING_REPLAY_GRAPH
            graph.disable_reason = str(exc)
            return False
        stats.loss_sum.add_(graph.buffers["loss"].detach())
        stats.loss_count.add_(count_delta)
        graph.replay_hit_count += 1
        return True

    def _train_shared_batch_graph_locked(self, src: torch.Tensor, tgt: torch.Tensor, active_rows: int) -> bool:
        return self._replay_graph_with_disable_fallback(
            "shared",
            src=src,
            tgt=tgt,
            active_rows=active_rows,
            count_delta=1,
        )

    def _train_model_bank_batch_graph_locked(self, slot_ids: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor) -> bool:
        return self._replay_graph_with_disable_fallback(
            "model_bank",
            src=src,
            tgt=tgt,
            active_rows=int(slot_ids.numel()),
            count_delta=self._count_active_model_bank_slots(slot_ids, int(slot_ids.numel())),
            slot_ids=slot_ids,
        )


__all__ = ["ESampTrainEngine", "ESampStats", "copy_active_rows_into_buffer", "group_row_indices_by_prompt"]
