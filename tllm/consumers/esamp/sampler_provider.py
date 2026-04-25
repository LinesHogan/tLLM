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
        isinstance(out, torch.Tensor)
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
        rows = int(source_hidden.shape[0])
        hidden = int(source_hidden.shape[1])
        if (
            getattr(runtime, "sampler_precomputed_pred_hidden_full", None) is None
            or tuple(runtime.sampler_precomputed_pred_hidden_full.shape) != (rows, hidden)
            or runtime.sampler_precomputed_pred_hidden_full.device != source_hidden.device
            or runtime.sampler_precomputed_pred_hidden_full.dtype != source_hidden.dtype
        ):
            runtime.sampler_precomputed_pred_hidden_full = torch.empty(
                (rows, hidden),
                device=source_hidden.device,
                dtype=source_hidden.dtype,
            )
        if (
            getattr(runtime, "sampler_precomputed_valid_mask", None) is None
            or int(runtime.sampler_precomputed_valid_mask.numel()) != rows
            or runtime.sampler_precomputed_valid_mask.device != source_hidden.device
        ):
            runtime.sampler_precomputed_valid_mask = torch.zeros((rows,), device=source_hidden.device, dtype=torch.bool)
        if (
            getattr(runtime, "sampler_precompute_source_hidden_full", None) is None
            or tuple(runtime.sampler_precompute_source_hidden_full.shape) != (rows, hidden)
            or runtime.sampler_precompute_source_hidden_full.device != source_hidden.device
            or runtime.sampler_precompute_source_hidden_full.dtype != source_hidden.dtype
        ):
            runtime.sampler_precompute_source_hidden_full = torch.empty(
                (rows, hidden),
                device=source_hidden.device,
                dtype=source_hidden.dtype,
            )
        if (
            getattr(runtime, "sampler_precompute_prompt_idx_full", None) is None
            or int(runtime.sampler_precompute_prompt_idx_full.numel()) != rows
            or runtime.sampler_precompute_prompt_idx_full.device != source_hidden.device
        ):
            runtime.sampler_precompute_prompt_idx_full = torch.empty(
                (rows,),
                device=source_hidden.device,
                dtype=torch.long,
            )
        if (
            getattr(runtime, "sampler_precomputed_all_row_ids", None) is None
            or int(runtime.sampler_precomputed_all_row_ids.numel()) != rows
            or runtime.sampler_precomputed_all_row_ids.device != source_hidden.device
        ):
            runtime.sampler_precomputed_all_row_ids = torch.arange(rows, device=source_hidden.device, dtype=torch.long)
        if self.config.distiller_sampler_backend in {"pre_filter_dense", "post_filter_dense_cache"}:
            weight, _ = self._get_lm_head_params(getattr(runner, "model", None))
            vocab = int(weight.shape[0])
            if (
                getattr(runtime, "sampler_precomputed_dense_logits_full", None) is None
                or tuple(runtime.sampler_precomputed_dense_logits_full.shape) != (rows, vocab)
                or runtime.sampler_precomputed_dense_logits_full.device != source_hidden.device
                or runtime.sampler_precomputed_dense_logits_full.dtype != torch.float32
            ):
                runtime.sampler_precomputed_dense_logits_full = torch.empty(
                    (rows, vocab),
                    device=source_hidden.device,
                    dtype=torch.float32,
                )

    def maybe_prepare_decode_step(self, *, runtime: Any, runner: Any) -> None:
        runtime.sampler_source_precompute_enabled = False
        runtime.sampler_source_capture_step_id = -1
        runtime.distiller_port_capture_step_id = -1
        runtime.distiller_port_publish_step_id = -1
        runtime.distiller_port_consume_step_id = -1
        runtime.sampler_precomputed_step_id = int(getattr(runtime, "event_step_id", -1))
        runtime.sampler_precomputed_row_ids = None
        runtime.sampler_precomputed_pred_hidden = None
        runtime.sampler_precomputed_dense_logits = None
        runtime.sampler_precomputed_dense_logits_full = None
        runtime.sampler_precomputed_pred_hidden_row_map = None
        runtime.sampler_precomputed_all_rows = False
        if not self.is_active():
            return
        self.ensure_runtime_buffers(runtime=runtime, runner=runner)
        valid_mask = getattr(runtime, "sampler_precomputed_valid_mask", None)
        if isinstance(valid_mask, torch.Tensor):
            valid_mask.zero_()
        decode_count = int(getattr(runtime, "decode_count", 0) or 0)
        prompt_idx_tensor = getattr(runtime, "decode_prompt_idx_tensor", None)
        if decode_count <= 0 or not isinstance(prompt_idx_tensor, torch.Tensor):
            return
        if self.engine.state.per_request_models and not self.engine.using_model_bank:
            return
        if self.engine.using_model_bank:
            prompt_input: Sequence[int] | torch.Tensor
            prompt_idxs = list(getattr(runtime, "decode_prompt_idxs", []))[:decode_count]
            if len(prompt_idxs) == decode_count:
                prompt_input = prompt_idxs
            else:
                prompt_input = prompt_idx_tensor[:decode_count]
            runtime.sampler_source_precompute_enabled = bool(
                self.engine.prepare_sampling_slots_for_step(prompt_input)
            )
            runtime.sampler_precomputed_all_rows = bool(runtime.sampler_source_precompute_enabled)
            return
        runtime.sampler_source_precompute_enabled = True
        runtime.sampler_precomputed_all_rows = True

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
        pred_hidden_full = getattr(runtime, "sampler_precomputed_pred_hidden_full", None)
        valid_mask = getattr(runtime, "sampler_precomputed_valid_mask", None)
        source_hidden_full = getattr(runtime, "sampler_precompute_source_hidden_full", None)
        prompt_idx_full = getattr(runtime, "sampler_precompute_prompt_idx_full", None)
        if (
            not isinstance(source_hidden, torch.Tensor)
            or not isinstance(prompt_idx_tensor, torch.Tensor)
            or not isinstance(pred_hidden_full, torch.Tensor)
            or not isinstance(valid_mask, torch.Tensor)
            or not isinstance(source_hidden_full, torch.Tensor)
            or not isinstance(prompt_idx_full, torch.Tensor)
        ):
            return
        dense_logits_full = getattr(runtime, "sampler_precomputed_dense_logits_full", None)

        def _run_capture(captured_hidden: torch.Tensor, captured_prompt_idxs: torch.Tensor) -> None:
            self.engine.predict_hidden_for_sampling_capture(
                captured_hidden,
                captured_prompt_idxs,
                out_pred_hidden=pred_hidden_full,
                out_valid_mask=valid_mask,
            )
            if self.config.distiller_sampler_backend == "pre_filter_dense":
                weight, bias = self._get_lm_head_params(getattr(runner, "model", None))
                runtime.sampler_precomputed_dense_logits_full = _write_dense_logits(
                    pred_hidden=pred_hidden_full,
                    weight=weight,
                    bias=bias,
                    out=dense_logits_full,
                )
            runtime.sampler_source_capture_step_id = int(getattr(runtime, "event_step_id", -1))

        allow_async = bool(getattr(runtime, "sampler_allow_source_async", False))
        if source_hidden.device.type != "cuda" or not allow_async:
            source_hidden_full.copy_(source_hidden)
            prompt_idx_full.copy_(prompt_idx_tensor)
            _run_capture(source_hidden_full, prompt_idx_full)
            runtime.sampler_precompute_event = None
            return

        stream = getattr(runtime, "sampler_precompute_stream", None)
        if stream is None:
            stream = _make_precompute_stream(source_hidden.device)
            runtime.sampler_precompute_stream = stream
        event = getattr(runtime, "sampler_precompute_event", None)
        if event is None:
            event = torch.cuda.Event(blocking=False)
            runtime.sampler_precompute_event = event
        with torch.cuda.stream(stream), torch.no_grad():
            stream.wait_stream(torch.cuda.current_stream(device=source_hidden.device))
            timing_start = None
            timing_end = None
            if bool(getattr(runtime, "distiller_timing_enabled", False)):
                timing_start = torch.cuda.Event(enable_timing=True, blocking=False)
                timing_end = torch.cuda.Event(enable_timing=True, blocking=False)
                timing_start.record(stream)
            with _maybe_record_function("distiller.capture_precompute"):
                source_hidden_full.copy_(source_hidden)
                prompt_idx_full.copy_(prompt_idx_tensor)
                _run_capture(source_hidden_full, prompt_idx_full)
            if timing_end is not None:
                timing_end.record(stream)
                runtime.distiller_precompute_event_pairs.append((timing_start, timing_end))
            event.record(stream)

    def maybe_schedule_precompute(self, *, runtime: Any, runner: Any) -> None:
        with _maybe_record_function("distiller.schedule_precompute"):
            if bool(getattr(runtime, "distiller_timing_enabled", False)):
                runtime.distiller_schedule_attempt_count += 1
            if not self.is_active():
                return
            decode_count = int(getattr(runtime, "decode_count", 0) or 0)
            source_path = str(getattr(runtime, "source_resolved_path", "") or "").strip()
            if decode_count <= 0 or not source_path:
                return
            full_pred_hidden = getattr(runtime, "sampler_precomputed_pred_hidden_full", None)
            full_valid_mask = getattr(runtime, "sampler_precomputed_valid_mask", None)
            if (
                bool(getattr(runtime, "sampler_source_precompute_enabled", False))
                and isinstance(full_pred_hidden, torch.Tensor)
                and isinstance(full_valid_mask, torch.Tensor)
                and int(full_pred_hidden.shape[0]) >= decode_count
                and int(full_valid_mask.numel()) >= decode_count
                and int(getattr(runtime, "sampler_source_capture_step_id", -1)) == int(getattr(runtime, "event_step_id", -2))
            ):
                event = getattr(runtime, "sampler_precompute_event", None)
                if event is not None and full_pred_hidden.device.type == "cuda":
                    torch.cuda.current_stream(device=full_pred_hidden.device).wait_event(event)
                runtime.sampler_precomputed_step_id = int(getattr(runtime, "event_step_id", -1))
                runtime.sampler_precomputed_row_ids = None
                runtime.sampler_precomputed_pred_hidden = None
                runtime.sampler_precomputed_dense_logits = None
                if (
                    self.config.distiller_sampler_backend in {"pre_filter_dense", "post_filter_dense_cache"}
                    and bool(getattr(runtime, "sampler_precomputed_all_rows", False))
                ):
                    full_dense_logits = getattr(runtime, "sampler_precomputed_dense_logits_full", None)
                    if isinstance(full_dense_logits, torch.Tensor) and int(full_dense_logits.shape[0]) >= decode_count:
                        runtime.sampler_precomputed_dense_logits = full_dense_logits[:decode_count]
                if bool(getattr(runtime, "distiller_timing_enabled", False)):
                    runtime.distiller_schedule_hit_count += 1
                return
            source_hidden = getattr(runtime, "tap_decode_hidden", {}).get(source_path)
            if not isinstance(source_hidden, torch.Tensor) or int(source_hidden.shape[0]) < decode_count:
                return
            prompt_idx_tensor = getattr(runtime, "decode_prompt_idx_tensor", None)
            if isinstance(prompt_idx_tensor, torch.Tensor) and int(prompt_idx_tensor.numel()) >= decode_count:
                prompt_input: Sequence[int] | torch.Tensor = prompt_idx_tensor[:decode_count]
            else:
                prompt_input = tuple(int(x) for x in list(getattr(runtime, "decode_prompt_idxs", []))[:decode_count])
            if (isinstance(prompt_input, tuple) and len(prompt_input) != decode_count) or (
                isinstance(prompt_input, torch.Tensor) and int(prompt_input.numel()) != decode_count
            ):
                return
            full_row_map_tensor = None
            if self.engine.using_model_bank and self.config.distiller_sampler_backend in {"post_filter_exact", "post_filter_dense_cache"}:
                prompt_list = tuple(int(x) for x in list(getattr(runtime, "decode_prompt_idxs", []))[:decode_count])
                if len(prompt_list) == decode_count:
                    unique_rows, unique_prompts, row_map = _first_prompt_rows(prompt_list)
                    if unique_rows and len(unique_rows) < decode_count:
                        unique_row_tensor = torch.as_tensor(unique_rows, device=source_hidden.device, dtype=torch.long)
                        active_hidden = source_hidden.index_select(0, unique_row_tensor)
                        prompt_input = tuple(unique_prompts)
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
                    all_row_ids = getattr(runtime, "sampler_precomputed_all_row_ids", None)
                    if isinstance(all_row_ids, torch.Tensor) and int(all_row_ids.numel()) >= decode_count:
                        store_row_ids = all_row_ids[:decode_count]
                    else:
                        store_row_ids = torch.arange(decode_count, device=row_ids.device, dtype=torch.long)
                runtime.sampler_precomputed_step_id = int(getattr(runtime, "event_step_id", -1))
                runtime.sampler_precomputed_row_ids = store_row_ids
                runtime.sampler_precomputed_pred_hidden = pred_hidden
                runtime.sampler_precomputed_pred_hidden_row_map = full_row_map_tensor
                runtime.sampler_precomputed_dense_logits = None
                runtime.sampler_precomputed_dense_logits_full = None
                runtime.sampler_precomputed_all_rows = int(store_row_ids.numel()) == decode_count

            if active_hidden.device.type != "cuda":
                row_ids, pred_hidden = self.engine.predict_hidden_for_sampling(
                    active_hidden,
                    prompt_input,
                    assume_all_model_bank_slots_ready=bool(self.engine.using_model_bank),
                )
                _store(row_ids, pred_hidden)
                if self.config.distiller_sampler_backend in {"pre_filter_dense", "post_filter_dense_cache"} and int(row_ids.numel()) > 0:
                    weight, bias = self._get_lm_head_params(getattr(runner, "model", None))
                    runtime.sampler_precomputed_dense_logits = _project_dense_logits(
                        pred_hidden=pred_hidden,
                        weight=weight,
                        bias=bias,
                    )
                runtime.sampler_precompute_event = None
                return

            stream = getattr(runtime, "sampler_precompute_stream", None)
            if stream is None:
                stream = _make_precompute_stream(active_hidden.device)
                runtime.sampler_precompute_stream = stream
            event = getattr(runtime, "sampler_precompute_event", None)
            if event is None:
                event = torch.cuda.Event(blocking=False)
                runtime.sampler_precompute_event = event
            with torch.cuda.stream(stream), torch.no_grad():
                stream.wait_stream(torch.cuda.current_stream(device=active_hidden.device))
                timing_start = None
                timing_end = None
                if bool(getattr(runtime, "distiller_timing_enabled", False)):
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
                    runtime.sampler_precomputed_dense_logits = _project_dense_logits(
                        pred_hidden=pred_hidden,
                        weight=weight,
                        bias=bias,
                    )
                if timing_end is not None:
                    timing_end.record(stream)
                    runtime.distiller_precompute_event_pairs.append((timing_start, timing_end))
                if bool(getattr(runtime, "distiller_timing_enabled", False)):
                    runtime.distiller_schedule_hit_count += 1
                event.record(stream)

    def prepare_step(self, view: SamplerStepView) -> CandidateModifierState | None:
        with _maybe_record_function("distiller.prepare_step"):
            if not self.is_active():
                return None
            runtime = getattr(view.runner, "_tllm_runtime", None)
            if runtime is not None and int(getattr(runtime, "distiller_port_publish_step_id", -1)) != int(view.engine_step_id):
                return None
            row_ids = None
            pred_hidden = None
            pred_hidden_row_map = None
            cached_dense_logits = None
            if runtime is not None and int(getattr(runtime, "sampler_precomputed_step_id", -1)) == int(view.engine_step_id):
                event = getattr(runtime, "sampler_precompute_event", None)
                if event is not None and view.device.type == "cuda":
                    current_stream = torch.cuda.current_stream(device=view.device)
                    timing_start = None
                    timing_end = None
                    if bool(getattr(runtime, "distiller_timing_enabled", False)):
                        timing_start = torch.cuda.Event(enable_timing=True, blocking=False)
                        timing_end = torch.cuda.Event(enable_timing=True, blocking=False)
                        timing_start.record(current_stream)
                    current_stream.wait_event(event)
                    if timing_end is not None:
                        timing_end.record(current_stream)
                        runtime.distiller_wait_event_pairs.append((timing_start, timing_end))
                cached_row_ids = getattr(runtime, "sampler_precomputed_row_ids", None)
                cached_pred_hidden = getattr(runtime, "sampler_precomputed_pred_hidden", None)
                cached_dense_logits_row = getattr(runtime, "sampler_precomputed_dense_logits", None)
                if isinstance(cached_row_ids, torch.Tensor) and isinstance(cached_pred_hidden, torch.Tensor):
                    row_ids = cached_row_ids
                    pred_hidden = cached_pred_hidden
                    pred_hidden_row_map = getattr(runtime, "sampler_precomputed_pred_hidden_row_map", None)
                    cached_dense_logits = cached_dense_logits_row
                full_pred_hidden = getattr(runtime, "sampler_precomputed_pred_hidden_full", None)
                full_valid_mask = getattr(runtime, "sampler_precomputed_valid_mask", None)
                full_dense_logits = getattr(runtime, "sampler_precomputed_dense_logits_full", None)
                if (
                    row_ids is None
                    and isinstance(full_pred_hidden, torch.Tensor)
                    and isinstance(full_valid_mask, torch.Tensor)
                    and int(full_pred_hidden.shape[0]) >= int(view.decode_count)
                    and int(full_valid_mask.numel()) >= int(view.decode_count)
                    and int(getattr(runtime, "sampler_source_capture_step_id", -1)) == int(view.engine_step_id)
                ):
                    if bool(getattr(runtime, "sampler_precomputed_all_rows", False)):
                        row_ids = torch.arange(view.decode_count, device=view.device, dtype=torch.long)
                        pred_hidden = full_pred_hidden[: view.decode_count]
                        if isinstance(full_dense_logits, torch.Tensor) and int(full_dense_logits.shape[0]) >= int(view.decode_count):
                            cached_dense_logits = full_dense_logits[: view.decode_count]
                    else:
                        row_ids = full_valid_mask[: view.decode_count].nonzero(as_tuple=False).view(-1)
                        if int(row_ids.numel()) > 0:
                            pred_hidden = full_pred_hidden[: view.decode_count].index_select(0, row_ids)
                            if isinstance(full_dense_logits, torch.Tensor) and int(full_dense_logits.shape[0]) >= int(view.decode_count):
                                cached_dense_logits = full_dense_logits[: view.decode_count].index_select(0, row_ids)
            if row_ids is None or pred_hidden is None:
                prompt_input = view.prompt_idx_tensor if view.prompt_idx_tensor is not None else view.prompt_idxs
                fallback_timing_start = None
                fallback_timing_end = None
                if bool(getattr(runtime, "distiller_timing_enabled", False)) and view.device.type == "cuda":
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
                if fallback_timing_end is not None:
                    fallback_timing_end.record(torch.cuda.current_stream(device=view.device))
                    runtime.distiller_fallback_event_pairs.append((fallback_timing_start, fallback_timing_end))
                cached_dense_logits = None
            if int(row_ids.numel()) <= 0:
                return None
            lm_head = self._get_lm_head(view.model)
            if runtime is not None:
                runtime.distiller_port_consume_step_id = int(view.engine_step_id)
            return CandidateModifierState(
                beta=float(self.config.distiller_beta),
                backend=self.config.distiller_sampler_backend,
                affected_row_ids=row_ids,
                pred_hidden=pred_hidden,
                lm_head_weight=lm_head.weight,
                lm_head_bias=getattr(lm_head, "bias", None),
                precomputed_dense_logits=cached_dense_logits if isinstance(cached_dense_logits, torch.Tensor) else None,
                pred_hidden_row_map=pred_hidden_row_map if isinstance(pred_hidden_row_map, torch.Tensor) else None,
            )
