#!/usr/bin/env python3
"""BaseConsumer wrapper around the ESamp training engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from tllm.common.path_resolution import candidate_capture_paths
from tllm.common.state import resolve_object_by_path
from tllm.consumers.base import BaseConsumer
from tllm.consumers.esamp.config import ESampConsumerConfig
from tllm.consumers.esamp.engine import ESampTrainEngine, ESampStats
from tllm.consumers.esamp.initializers.svd import build_model_bank_initializer
from tllm.consumers.esamp.sampler_provider import ESampSamplerModifierProvider
from tllm.contracts.gpu_stage import DeviceTensorLease
from tllm.contracts.port_bundle import PortBundle
from tllm.contracts.request_meta_view import RowBatchMeta
from tllm.contracts.runtime_context import RuntimeContext
from tllm.ports.base import ConsumerFlow
from tllm.ports.request_meta import RequestMeta
from tllm.ports.residual_stream import ResidualStream


@dataclass(slots=True)
class _TrainingRows:
    source_hidden: torch.Tensor
    target_hidden: torch.Tensor
    prompt_idxs: list[int]
    request_meta_is_first_per_prompt: bool


class ESampConsumer(BaseConsumer):
    """Dispatch-plan friendly ESamp wrapper around the formal engine."""

    @staticmethod
    def _build_engine(config: ESampConsumerConfig) -> ESampTrainEngine:
        return ESampTrainEngine(
            hidden_dim=int(config.distiller_hidden_dim),
            lr=float(config.distiller_lr),
            enabled=bool(config.enable_esamp_training),
            per_request_models=bool(config.per_request_models),
            per_request_model_bank=bool(config.per_request_model_bank),
            model_bank_slots=int(config.model_bank_slots),
            model_bank_flush_interval=int(config.model_bank_flush_interval),
            model_bank_rank=int(config.model_bank_rank),
            model_bank_use_output_layernorm=bool(config.model_bank_use_output_layernorm),
            model_bank_initializer=build_model_bank_initializer(config.model_bank_initializer),
            model_bank_train_cudagraph=bool(config.model_bank_train_cudagraph),
            model_bank_forward_backend=str(config.model_bank_forward_backend),
            adaptation_pipeline_slots=int(config.adaptation_pipeline_slots),
            adaptation_stream_mode=str(config.adaptation_stream_mode),
            adaptation_stream_priority=int(config.adaptation_stream_priority),
            trace_per_request_losses=bool(config.trace_per_request_losses),
            trace_interval=int(config.trace_interval),
            trace_max_points=int(config.trace_max_points),
        )

    def __init__(
        self,
        config: ESampConsumerConfig,
        *,
        engine: Optional[ESampTrainEngine] = None,
    ) -> None:
        self.config = config
        self._engine = engine or self._build_engine(config)
        self._target_resolved_path = str(config.target_layer_path)
        self._last_active_rows = 0
        self._last_prompt_idxs: tuple[int, ...] = ()
        self._runtime_layers_cache_key: tuple[int, str, str] | None = None
        self._runtime_source_layer: Optional[torch.nn.Module] = None
        self._runtime_target_layer: Optional[torch.nn.Module] = None
        self._runtime_initializer_key: tuple[int, str, int, str, torch.dtype] | None = None
        self._runtime_resource_key: tuple[object, ...] | None = None
        self._sampler_provider = ESampSamplerModifierProvider(config=self.config, engine=self._engine)

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

    def flows(self) -> Sequence[ConsumerFlow]:
        if not bool(self.config.enable_esamp_training):
            return ()
        return [
            ConsumerFlow(
                reads=(
                    ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                    ResidualStream.read(layer=-1, site="block_output", phase="decode", role="target"),
                    RequestMeta.read(),
                ),
                writes=(),
                window="out_of_band",
                delivery="device_lease",
                ownership="runtime_lease",
                row_compaction="first_per_prompt" if bool(self.config.per_request_model_bank) else "none",
                max_bundle_rows=int(self.config.model_bank_slots) if bool(self.config.per_request_model_bank) else 0,
                bundle_key=("engine_step_id", "phase"),
            )
        ]

    def update_config(self, config: ESampConsumerConfig) -> None:
        self.config = config
        self._sampler_provider.config = config
        self._engine.configure(config)
        self._runtime_layers_cache_key = None
        self._runtime_source_layer = None
        self._runtime_target_layer = None
        self._runtime_initializer_key = None
        self._runtime_resource_key = None

    def sampler_modifier_provider(self) -> ESampSamplerModifierProvider:
        return self._sampler_provider

    @staticmethod
    def _infer_hidden_dtype(layer: torch.nn.Module) -> torch.dtype:
        for p in layer.parameters():
            return p.dtype
        for b in layer.buffers():
            return b.dtype
        return torch.float32

    def _resolve_layer_with_fallback(self, model: torch.nn.Module, path: str) -> tuple[torch.nn.Module, str]:
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
        raise RuntimeError(f"Cannot resolve ESamp layer path `{path}`. Tried: {details}")

    def _maybe_prepare_initializer(self, *, model: torch.nn.Module, target_layer: torch.nn.Module, target_resolved: str) -> None:
        hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
        if hidden_size <= 0:
            return
        self._engine.prepare_model_bank_initializer(
            model=model,
            target_layer=target_layer,
            target_resolved=target_resolved,
            hidden_size=hidden_size,
        )

    def _resource_config_key(self) -> tuple[object, ...]:
        return (
            int(self.config.distiller_hidden_dim),
            float(self.config.distiller_lr),
            bool(self.config.per_request_models),
            bool(self.config.per_request_model_bank),
            int(self.config.model_bank_slots),
            int(self.config.model_bank_flush_interval),
            int(self.config.model_bank_rank),
            bool(self.config.model_bank_use_output_layernorm),
            self.config.model_bank_initializer,
            bool(self.config.model_bank_train_cudagraph),
            str(self.config.model_bank_forward_backend),
            int(self.config.adaptation_pipeline_slots),
            str(self.config.adaptation_stream_mode),
            int(self.config.adaptation_stream_priority),
        )

    def _resolve_runtime_layers(
        self, model: torch.nn.Module
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        source_layer, source_resolved = self._resolve_layer_with_fallback(model, self.config.source_layer_path)
        target_layer, target_resolved = self._resolve_layer_with_fallback(model, self.config.target_layer_path)
        self._target_resolved_path = target_resolved
        return source_layer, target_layer

    def _runtime_layers_for_model(self, model: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module]:
        cache_key = (id(model), str(self.config.source_layer_path), str(self.config.target_layer_path))
        if (
            self._runtime_layers_cache_key == cache_key
            and self._runtime_source_layer is not None
            and self._runtime_target_layer is not None
        ):
            return self._runtime_source_layer, self._runtime_target_layer
        source_layer, target_layer = self._resolve_runtime_layers(model)
        self._runtime_layers_cache_key = cache_key
        self._runtime_source_layer = source_layer
        self._runtime_target_layer = target_layer
        self._runtime_initializer_key = None
        return source_layer, target_layer

    def _ensure_runtime_resources(
        self,
        ctx: RuntimeContext,
        *,
        rows_hidden: Optional[torch.Tensor] = None,
    ) -> None:
        model = ctx.model
        source_layer = None
        target_layer = None
        if model is not None:
            source_layer, target_layer = self._runtime_layers_for_model(model)

        hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
        hidden_dtype = self._infer_hidden_dtype(source_layer) if source_layer is not None else None
        if rows_hidden is not None and rows_hidden.ndim >= 2:
            hidden_size = max(hidden_size, int(rows_hidden.shape[1]))
            hidden_dtype = hidden_dtype or rows_hidden.dtype
        if hidden_size <= 0 or hidden_dtype is None:
            raise RuntimeError("ESamp consumer cannot infer hidden_size/hidden_dtype for runtime resources")
        rows = int(self.config.graph_scratch_rows)
        if rows <= 0:
            runner_rows = int(getattr(ctx.runner, "max_num_reqs", 0) or 0)
            rows = runner_rows
        if rows_hidden is not None:
            rows = max(rows, int(rows_hidden.shape[0]))
        if rows <= 0:
            raise RuntimeError("ESamp consumer requires positive rows via graph_scratch_rows, runner.max_num_reqs, or rows_hidden")
        device = rows_hidden.device if rows_hidden is not None else ctx.device
        resource_key = (
            id(model),
            str(device),
            hidden_dtype,
            int(hidden_size),
            int(rows),
            self._resource_config_key(),
        )
        if self._runtime_resource_key == resource_key:
            return
        self._engine.ensure_resources(
            device=device,
            rows=rows,
            hidden_size=hidden_size,
            hidden_dtype=hidden_dtype,
        )
        initializer_key = (
            id(model),
            str(self._target_resolved_path),
            int(hidden_size),
            str(device),
            hidden_dtype,
        ) if model is not None else None
        if model is not None and target_layer is not None and self._runtime_initializer_key != initializer_key:
            self._maybe_prepare_initializer(model=model, target_layer=target_layer, target_resolved=self._target_resolved_path)
            self._runtime_initializer_key = initializer_key
        self._runtime_resource_key = resource_key

    @staticmethod
    def _first_row_per_prompt(prompt_idxs: Sequence[int]) -> list[int]:
        seen: set[int] = set()
        row_ids: list[int] = []
        for row_i, prompt_idx in enumerate(prompt_idxs):
            prompt_key = int(prompt_idx)
            if prompt_key < 0 or prompt_key in seen:
                continue
            seen.add(prompt_key)
            row_ids.append(int(row_i))
        return row_ids

    def _extract_training_rows(
        self,
        bundle: PortBundle,
    ) -> _TrainingRows | None:
        stage_lease = bundle.entries.get("device_lease")
        uses_device_lease = isinstance(stage_lease, DeviceTensorLease)
        if uses_device_lease:
            source_hidden = stage_lease.entries.get("source")
            target_hidden = stage_lease.entries.get("target")
            active_rows = max(0, int(stage_lease.active_rows))
            if isinstance(source_hidden, torch.Tensor) and active_rows < int(source_hidden.shape[0]):
                source_hidden = source_hidden[:active_rows]
            if isinstance(target_hidden, torch.Tensor) and active_rows < int(target_hidden.shape[0]):
                target_hidden = target_hidden[:active_rows]
        else:
            source_hidden = bundle.entries.get("source")
            target_hidden = bundle.entries.get("target")
        request_meta = bundle.entries.get("request_meta")
        if not isinstance(source_hidden, torch.Tensor) or not isinstance(target_hidden, torch.Tensor):
            return None
        if int(source_hidden.shape[0]) != int(target_hidden.shape[0]):
            raise RuntimeError("ESamp consumer requires source/target hidden batches to have the same row count")

        if not isinstance(request_meta, RowBatchMeta):
            raise RuntimeError("ESamp consumer requires request_meta to be RowBatchMeta")
        if uses_device_lease and len(request_meta.request_ids) != int(source_hidden.shape[0]):
            raise RuntimeError("ESamp consumer requires row metadata to match active lease rows")
        prompt_idxs = [int(prompt_idx) for prompt_idx in request_meta.prompt_idxs]
        request_meta_is_first_per_prompt = str(request_meta.row_compaction) == "first_per_prompt"
        while len(prompt_idxs) < int(source_hidden.shape[0]):
            prompt_idxs.append(-1)

        return _TrainingRows(
            source_hidden=source_hidden,
            target_hidden=target_hidden,
            prompt_idxs=prompt_idxs,
            request_meta_is_first_per_prompt=request_meta_is_first_per_prompt,
        )

    def consume_bundle(self, bundle: PortBundle, ctx: RuntimeContext) -> None:
        if not bool(self.config.enable_esamp_training):
            return
        rows = self._extract_training_rows(bundle)
        if rows is None:
            return
        source_hidden = rows.source_hidden
        target_hidden = rows.target_hidden
        prompt_idxs = rows.prompt_idxs

        if bool(self.config.per_request_model_bank) and not rows.request_meta_is_first_per_prompt:
            row_ids = self._first_row_per_prompt(prompt_idxs[: int(source_hidden.shape[0])])
            if row_ids:
                row_tensor = torch.as_tensor(row_ids, device=source_hidden.device, dtype=torch.long)
                source_hidden = source_hidden.index_select(0, row_tensor)
                target_hidden = target_hidden.index_select(0, row_tensor)
                prompt_idxs = [prompt_idxs[row_i] for row_i in row_ids]
            else:
                source_hidden = source_hidden[:0]
                target_hidden = target_hidden[:0]
                prompt_idxs = []

        self._ensure_runtime_resources(ctx, rows_hidden=source_hidden)
        self._last_active_rows = int(source_hidden.shape[0])
        self._last_prompt_idxs = tuple(prompt_idxs[: int(source_hidden.shape[0])])
        self._engine.launch_step(source_hidden, target_hidden)

    def on_step_end(self, ctx: RuntimeContext) -> None:
        self._engine.launch_delayed_backward(
            int(self._last_active_rows),
            prompt_idxs=self._last_prompt_idxs,
        )
        self._last_active_rows = 0
        self._last_prompt_idxs = ()

    def set_enabled(self, enabled: bool) -> None:
        self.config.enable_esamp_training = bool(enabled)
        self._engine.set_enabled(bool(enabled))

    def synchronize(self) -> None:
        self._engine.synchronize()

    def read_and_reset_stats(self, sync: bool = True) -> ESampStats:
        return self._engine.read_and_reset_stats(sync=sync)

    def read_and_reset_per_request_stats(self, sync: bool = True):
        return self._engine.read_and_reset_per_request_stats(sync=sync)
