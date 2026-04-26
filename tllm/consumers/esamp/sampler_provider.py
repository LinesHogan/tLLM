#!/usr/bin/env python3
"""ESamp-owned provider for same-step sampler intervention."""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.autograd.profiler import record_function

from tllm.consumers.esamp.config import ESampConsumerConfig
from tllm.consumers.esamp.engine import ESampTrainEngine
from tllm.runtime.residual_runtime import SamplerPrecomputeCache
from tllm.runtime.sampler_bridge.provider import SamplerModifierProvider
from tllm.runtime.sampler_bridge.types import CandidateModifierState, SamplerStepView

_SAMPLER_PRECOMPUTE_STREAM_PRIORITY = 2
_ENABLE_DISTILLER_RECORD_FUNCTION = os.getenv("TLLM_ENABLE_DISTILLER_RECORD_FUNCTION", "") == "1"


def _maybe_record_function(name: str):
    return record_function(name) if _ENABLE_DISTILLER_RECORD_FUNCTION else nullcontext()


def _resolve_lm_head(model: Any) -> Any:
    candidates = [
        getattr(model, "lm_head", None),
        getattr(getattr(model, "model", None), "lm_head", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "lm_head", None),
        (model.get_output_embeddings() if callable(getattr(model, "get_output_embeddings", None)) else None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "weight"):
            return candidate
    raise RuntimeError("ESamp sampler provider could not resolve lm_head from model")


def _project_dense_logits(
    *,
    pred_hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    with _maybe_record_function("distiller.precompute_dense_logits"):
        dense = F.linear(pred_hidden.to(dtype=weight.dtype), weight, bias)
        return dense.to(device=pred_hidden.device, dtype=torch.float32)


def _write_dense_logits(
    *,
    pred_hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor | None,
) -> torch.Tensor:
    dense = _project_dense_logits(pred_hidden=pred_hidden, weight=weight, bias=bias)
    if (
        out is not None
        and out.device == dense.device
        and out.dtype == dense.dtype
        and int(out.shape[0]) >= int(dense.shape[0])
        and int(out.shape[1]) == int(dense.shape[1])
    ):
        out[: int(dense.shape[0])].copy_(dense)
        return out
    return dense


def _make_precompute_stream(device: torch.device) -> torch.cuda.Stream:
    # Run distiller work on a lower-priority stream so the main vLLM graph keeps first claim on SM time.
    return torch.cuda.Stream(device=device, priority=_SAMPLER_PRECOMPUTE_STREAM_PRIORITY)


def _first_prompt_rows(prompt_idxs: Sequence[int]) -> tuple[list[int], list[int], list[int]]:
    seen: dict[int, int] = {}
    row_ids: list[int] = []
    prompt_unique: list[int] = []
    row_map: list[int] = []
    for row_i, prompt_idx in enumerate(prompt_idxs):
        prompt_key = int(prompt_idx)
        mapped = seen.get(prompt_key)
        if mapped is None and prompt_key >= 0:
            mapped = len(row_ids)
            seen[prompt_key] = mapped
            row_ids.append(int(row_i))
            prompt_unique.append(prompt_key)
        row_map.append(-1 if mapped is None else int(mapped))
    return row_ids, prompt_unique, row_map


def _runtime_prompt_tensor(*, runtime: Any, decode_count: int, device: torch.device) -> torch.Tensor | None:
    count = int(decode_count)
    if count <= 0:
        return None
    prompt_idx_tensor = getattr(runtime, "decode_prompt_idx_tensor", None)
    if isinstance(prompt_idx_tensor, torch.Tensor) and int(prompt_idx_tensor.numel()) >= count:
        return prompt_idx_tensor[:count].to(device=device, dtype=torch.long)
    prompt_idxs = tuple(int(x) for x in list(getattr(runtime, "decode_prompt_idxs", []))[:count])
    if len(prompt_idxs) != count:
        return None
    return torch.as_tensor(prompt_idxs, device=device, dtype=torch.long)


@dataclass
class ESampSamplerModifierProvider(SamplerModifierProvider):
    config: ESampConsumerConfig
    engine: ESampTrainEngine
    _cached_lm_head_owner: Any = None
    _cached_lm_head: Any = None
    _cached_lm_head_weight: torch.Tensor | None = None
    _cached_lm_head_bias: torch.Tensor | None = None

    def is_active(self) -> bool:
        return (
            bool(self.config.enable_distiller_intervention)
            and bool(self.config.enable_esamp_training)
            and float(getattr(self.config, "distiller_beta", 0.0) or 0.0) != 0.0
        )

    def _get_lm_head(self, model: Any) -> Any:
        if model is self._cached_lm_head_owner and self._cached_lm_head is not None:
            return self._cached_lm_head
        lm_head = _resolve_lm_head(model)
        self._cached_lm_head_owner = model
        self._cached_lm_head = lm_head
        self._cached_lm_head_weight = lm_head.weight
        self._cached_lm_head_bias = getattr(lm_head, "bias", None)
        return lm_head

    def _get_lm_head_params(self, model: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        if model is not self._cached_lm_head_owner or self._cached_lm_head is None:
            self._get_lm_head(model)
        assert self._cached_lm_head_weight is not None
        return self._cached_lm_head_weight, self._cached_lm_head_bias

    def ensure_runtime_buffers(self, *, runtime: Any, runner: Any) -> None:
        if not self.is_active():
            return
        source_path = str(getattr(runtime, "source_resolved_path", "") or "").strip()
        source_hidden = getattr(runtime, "tap_decode_hidden", {}).get(source_path)
        if not isinstance(source_hidden, torch.Tensor):
            return
        dense_vocab = None
        if self.config.distiller_sampler_backend in {"pre_filter_dense", "post_filter_dense_cache"}:
            weight, _ = self._get_lm_head_params(getattr(runner, "model", None))
            dense_vocab = int(weight.shape[0])
        runtime.sampler_precompute.ensure_buffers(
            source_hidden=source_hidden,
            dense_vocab=dense_vocab,
        )

    def maybe_prepare_decode_step(self, *, runtime: Any, runner: Any) -> None:
        runtime.sampler_precompute.reset_decode_step(int(getattr(runtime, "event_step_id", -1)))
        if not self.is_active():
            return
        self.ensure_runtime_buffers(runtime=runtime, runner=runner)
        decode_count = int(getattr(runtime, "decode_count", 0) or 0)
        prompt_tensor = _runtime_prompt_tensor(runtime=runtime, decode_count=decode_count, device=torch.device("cpu"))
        if prompt_tensor is None:
            return
        if self.engine.state.per_request_models and not self.engine.using_model_bank:
            return
        if self.engine.using_model_bank:
            runtime.sampler_precompute.source_enabled = bool(
                self.engine.assign_sampling_model_bank_slots(prompt_tensor)
            )
            runtime.sampler_precompute.all_rows = bool(runtime.sampler_precompute.source_enabled)
            return
        runtime.sampler_precompute.source_enabled = True
        runtime.sampler_precompute.all_rows = True

    def maybe_capture_source_precompute(
        self,
        *,
        runtime: Any,
        runner: Any,
        layer_path: str,
    ) -> None:
        if str(layer_path).strip() != str(getattr(runtime, "source_resolved_path", "") or "").strip():
            return
        if not self.is_active():
            return
        if not bool(getattr(runtime, "sampler_allow_source_capture", False)):
            return
        source_hidden = getattr(runtime, "tap_decode_hidden", {}).get(str(layer_path).strip())
        prompt_idx_tensor = getattr(runtime, "decode_prompt_idx_buf", None)
        if not isinstance(source_hidden, torch.Tensor) or not isinstance(prompt_idx_tensor, torch.Tensor):
            return
        dense_vocab = None
        if self.config.distiller_sampler_backend == "pre_filter_dense":
            weight, _ = self._get_lm_head_params(getattr(runner, "model", None))
            dense_vocab = int(weight.shape[0])
        buffers = runtime.sampler_precompute.ensure_buffers(
            source_hidden=source_hidden,
            dense_vocab=dense_vocab,
        )
        dense_out = buffers.dense_logits

        def _run_capture(captured_hidden: torch.Tensor, captured_prompt_idxs: torch.Tensor) -> None:
            self.engine.predict_hidden_for_sampling_capture(
                captured_hidden,
                captured_prompt_idxs,
                out_pred_hidden=buffers.pred_hidden,
                out_valid_mask=buffers.valid_mask,
            )
            if self.config.distiller_sampler_backend == "pre_filter_dense":
                weight, bias = self._get_lm_head_params(getattr(runner, "model", None))
                buffers.dense_logits = _write_dense_logits(
                    pred_hidden=buffers.pred_hidden,
                    weight=weight,
                    bias=bias,
                    out=dense_out,
                )
            runtime.sampler_precompute.source_capture_step_id = int(getattr(runtime, "event_step_id", -1))

        allow_async = bool(getattr(runtime, "sampler_allow_source_async", False))
        if source_hidden.device.type != "cuda" or not allow_async:
            buffers.source_hidden.copy_(source_hidden)
            buffers.prompt_idx.copy_(prompt_idx_tensor)
            _run_capture(buffers.source_hidden, buffers.prompt_idx)
            runtime.sampler_precompute.event = None
            return

        stream = getattr(runtime.sampler_precompute, "stream", None)
        if stream is None:
            stream = _make_precompute_stream(source_hidden.device)
            runtime.sampler_precompute.stream = stream
        event = getattr(runtime.sampler_precompute, "event", None)
        if event is None:
            event = torch.cuda.Event(blocking=False)
            runtime.sampler_precompute.event = event
        with torch.cuda.stream(stream), torch.no_grad():
            stream.wait_stream(torch.cuda.current_stream(device=source_hidden.device))
            timing_start = None
            timing_end = None
            if bool(getattr(runtime.sampler_precompute, "timing_enabled", False)):
                timing_start = torch.cuda.Event(enable_timing=True, blocking=False)
                timing_end = torch.cuda.Event(enable_timing=True, blocking=False)
                timing_start.record(stream)
            with _maybe_record_function("distiller.capture_precompute"):
                buffers.source_hidden.copy_(source_hidden)
                buffers.prompt_idx.copy_(prompt_idx_tensor)
                _run_capture(buffers.source_hidden, buffers.prompt_idx)
            if timing_end is not None:
                timing_end.record(stream)
                runtime.sampler_precompute.precompute_event_pairs.append((timing_start, timing_end))
            event.record(stream)

    def maybe_schedule_precompute(self, *, runtime: Any, runner: Any) -> None:
        with _maybe_record_function("distiller.schedule_precompute"):
            if bool(getattr(runtime.sampler_precompute, "timing_enabled", False)):
                runtime.sampler_precompute.schedule_attempt_count += 1
            if not self.is_active():
                return
            decode_count = int(getattr(runtime, "decode_count", 0) or 0)
            source_path = str(getattr(runtime, "source_resolved_path", "") or "").strip()
            if decode_count <= 0 or not source_path:
                return
            step_id = int(getattr(runtime, "event_step_id", -1))
            if bool(getattr(runtime.sampler_precompute, "source_enabled", False)):
                captured = runtime.sampler_precompute.captured_rows_for_step(
                    step_id=step_id,
                    decode_count=decode_count,
                )
                event = getattr(runtime.sampler_precompute, "event", None)
                if captured is not None:
                    if event is not None and captured.pred_hidden.device.type == "cuda":
                        torch.cuda.current_stream(device=captured.pred_hidden.device).wait_event(event)
                    runtime.sampler_precompute.cache = captured
                    runtime.sampler_precompute.precomputed_step_id = step_id
                    if bool(getattr(runtime.sampler_precompute, "timing_enabled", False)):
                        runtime.sampler_precompute.schedule_hit_count += 1
                    return
            source_hidden = getattr(runtime, "tap_decode_hidden", {}).get(source_path)
            if not isinstance(source_hidden, torch.Tensor) or int(source_hidden.shape[0]) < decode_count:
                return
            prompt_input = _runtime_prompt_tensor(runtime=runtime, decode_count=decode_count, device=source_hidden.device)
            if prompt_input is None:
                return
            full_row_map_tensor = None
            if self.engine.using_model_bank and self.config.distiller_sampler_backend in {"post_filter_exact", "post_filter_dense_cache"}:
                prompt_list = tuple(int(x) for x in prompt_input.detach().cpu().tolist())
                if len(prompt_list) == decode_count:
                    unique_rows, unique_prompts, row_map = _first_prompt_rows(prompt_list)
                    if unique_rows and len(unique_rows) < decode_count:
                        unique_row_tensor = torch.as_tensor(unique_rows, device=source_hidden.device, dtype=torch.long)
                        active_hidden = source_hidden.index_select(0, unique_row_tensor)
                        prompt_input = torch.as_tensor(unique_prompts, device=source_hidden.device, dtype=torch.long)
                        full_row_map_tensor = torch.as_tensor(row_map, device=source_hidden.device, dtype=torch.long)
                    else:
                        active_hidden = source_hidden[:decode_count]
                else:
                    active_hidden = source_hidden[:decode_count]
            else:
                active_hidden = source_hidden[:decode_count]

            def _store(row_ids: torch.Tensor, pred_hidden: torch.Tensor) -> None:
                store_row_ids = row_ids
                if full_row_map_tensor is not None:
                    store_row_ids = torch.arange(decode_count, device=row_ids.device, dtype=torch.long)
                runtime.sampler_precompute.store_cache(
                    step_id=int(getattr(runtime, "event_step_id", -1)),
                    row_ids=store_row_ids,
                    pred_hidden=pred_hidden,
                    pred_hidden_row_map=full_row_map_tensor,
                    all_rows=int(store_row_ids.numel()) == decode_count,
                )

            if active_hidden.device.type != "cuda":
                row_ids, pred_hidden = self.engine.predict_hidden_for_sampling(
                    active_hidden,
                    prompt_input,
                    assume_all_model_bank_slots_ready=bool(self.engine.using_model_bank),
                )
                _store(row_ids, pred_hidden)
                if self.config.distiller_sampler_backend in {"pre_filter_dense", "post_filter_dense_cache"} and int(row_ids.numel()) > 0:
                    weight, bias = self._get_lm_head_params(getattr(runner, "model", None))
                    cache = runtime.sampler_precompute.cache_for_step(int(getattr(runtime, "event_step_id", -1)))
                    if cache is not None:
                        cache.dense_logits = _project_dense_logits(
                            pred_hidden=pred_hidden,
                            weight=weight,
                            bias=bias,
                        )
                runtime.sampler_precompute.event = None
                return

            stream = getattr(runtime.sampler_precompute, "stream", None)
            if stream is None:
                stream = _make_precompute_stream(active_hidden.device)
                runtime.sampler_precompute.stream = stream
            event = getattr(runtime.sampler_precompute, "event", None)
            if event is None:
                event = torch.cuda.Event(blocking=False)
                runtime.sampler_precompute.event = event
            with torch.cuda.stream(stream), torch.no_grad():
                stream.wait_stream(torch.cuda.current_stream(device=active_hidden.device))
                timing_start = None
                timing_end = None
                if bool(getattr(runtime.sampler_precompute, "timing_enabled", False)):
                    timing_start = torch.cuda.Event(enable_timing=True, blocking=False)
                    timing_end = torch.cuda.Event(enable_timing=True, blocking=False)
                    timing_start.record(stream)
                with _maybe_record_function("distiller.precompute_hidden"):
                    row_ids, pred_hidden = self.engine.predict_hidden_for_sampling(
                        active_hidden,
                        prompt_input,
                        assume_all_model_bank_slots_ready=bool(self.engine.using_model_bank),
                    )
                _store(row_ids, pred_hidden)
                if self.config.distiller_sampler_backend in {"pre_filter_dense", "post_filter_dense_cache"} and int(row_ids.numel()) > 0:
                    weight, bias = self._get_lm_head_params(getattr(runner, "model", None))
                    cache = runtime.sampler_precompute.cache_for_step(int(getattr(runtime, "event_step_id", -1)))
                    if cache is not None:
                        cache.dense_logits = _project_dense_logits(
                            pred_hidden=pred_hidden,
                            weight=weight,
                            bias=bias,
                        )
                if timing_end is not None:
                    timing_end.record(stream)
                    runtime.sampler_precompute.precompute_event_pairs.append((timing_start, timing_end))
                if bool(getattr(runtime.sampler_precompute, "timing_enabled", False)):
                    runtime.sampler_precompute.schedule_hit_count += 1
                event.record(stream)

    def prepare_step(self, view: SamplerStepView) -> CandidateModifierState | None:
        with _maybe_record_function("distiller.prepare_step"):
            if not self.is_active():
                return None
            runtime = getattr(view.runner, "_tllm_runtime", None)
            if runtime is not None and int(getattr(runtime.sampler_precompute, "port_publish_step_id", -1)) != int(view.engine_step_id):
                return None
            cache = None
            if runtime is not None:
                event = getattr(runtime.sampler_precompute, "event", None)
                if event is not None and view.device.type == "cuda":
                    current_stream = torch.cuda.current_stream(device=view.device)
                    timing_start = None
                    timing_end = None
                    if bool(getattr(runtime.sampler_precompute, "timing_enabled", False)):
                        timing_start = torch.cuda.Event(enable_timing=True, blocking=False)
                        timing_end = torch.cuda.Event(enable_timing=True, blocking=False)
                        timing_start.record(current_stream)
                    current_stream.wait_event(event)
                    if timing_end is not None:
                        timing_end.record(current_stream)
                        runtime.sampler_precompute.wait_event_pairs.append((timing_start, timing_end))
                cache = runtime.sampler_precompute.cache_for_step(int(view.engine_step_id))
                if cache is None:
                    cache = runtime.sampler_precompute.captured_rows_for_step(
                        step_id=int(view.engine_step_id),
                        decode_count=int(view.decode_count),
                    )
            if cache is None:
                prompt_input = (
                    view.prompt_idx_tensor.to(device=view.device, dtype=torch.long)
                    if view.prompt_idx_tensor is not None
                    else torch.as_tensor(tuple(int(x) for x in view.prompt_idxs), device=view.device, dtype=torch.long)
                )
                fallback_timing_start = None
                fallback_timing_end = None
                if runtime is not None and bool(getattr(runtime.sampler_precompute, "timing_enabled", False)) and view.device.type == "cuda":
                    current_stream = torch.cuda.current_stream(device=view.device)
                    fallback_timing_start = torch.cuda.Event(enable_timing=True, blocking=False)
                    fallback_timing_end = torch.cuda.Event(enable_timing=True, blocking=False)
                    fallback_timing_start.record(current_stream)
                with _maybe_record_function("distiller.prepare_step.fallback_predict"):
                    row_ids, pred_hidden = self.engine.predict_hidden_for_sampling(
                        view.source_hidden,
                        prompt_input,
                        assume_all_model_bank_slots_ready=bool(self.engine.using_model_bank),
                    )
                if runtime is not None and fallback_timing_end is not None:
                    fallback_timing_end.record(torch.cuda.current_stream(device=view.device))
                    runtime.sampler_precompute.fallback_event_pairs.append((fallback_timing_start, fallback_timing_end))
                cache = (
                    runtime.sampler_precompute.store_cache(
                        step_id=int(view.engine_step_id),
                        row_ids=row_ids,
                        pred_hidden=pred_hidden,
                        all_rows=int(row_ids.numel()) == int(view.decode_count),
                    )
                    if runtime is not None
                    else SamplerPrecomputeCache(
                        step_id=int(view.engine_step_id),
                        row_ids=row_ids,
                        pred_hidden=pred_hidden,
                        all_rows=int(row_ids.numel()) == int(view.decode_count),
                    )
                )
            if int(cache.row_ids.numel()) <= 0:
                return None
            lm_head = self._get_lm_head(view.model)
            if runtime is not None:
                runtime.sampler_precompute.port_consume_step_id = int(view.engine_step_id)
            return CandidateModifierState(
                beta=float(self.config.distiller_beta),
                backend=self.config.distiller_sampler_backend,
                affected_row_ids=cache.row_ids,
                pred_hidden=cache.pred_hidden,
                lm_head_weight=lm_head.weight,
                lm_head_bias=getattr(lm_head, "bias", None),
                precomputed_dense_logits=cache.dense_logits,
                pred_hidden_row_map=cache.pred_hidden_row_map,
            )
