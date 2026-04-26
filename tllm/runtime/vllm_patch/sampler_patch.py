#!/usr/bin/env python3
"""Thin sampler patch that routes through the tLLM sampler bridge."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any

import torch
from torch.autograd.profiler import record_function

from tllm.runtime.sampler_bridge.min_p import apply_min_p

_PATCH_INSTALLED = False
_ORIG_VLLM_SAMPLER_SAMPLE = None
_SAMPLING_EPS = 1e-5
_ENABLE_DISTILLER_RECORD_FUNCTION = os.getenv("TLLM_ENABLE_DISTILLER_RECORD_FUNCTION", "") == "1"


def _maybe_record_function(name: str):
    return record_function(name) if _ENABLE_DISTILLER_RECORD_FUNCTION else nullcontext()


def _resolve_provider(runtime: Any) -> Any:
    consumer = getattr(runtime, "consumer", None)
    if consumer is None:
        return None
    getter = getattr(consumer, "sampler_modifier_provider", None)
    if not callable(getter):
        return None
    return getter()


def sample_with_optional_modifier(*args, **kwargs):
    from tllm.runtime.sampler_bridge.bridge import sample_with_optional_modifier as _impl

    return _impl(*args, **kwargs)


def build_sampler_step_view(*args, **kwargs):
    from tllm.runtime.sampler_bridge.runtime_view import build_sampler_step_view as _impl

    return _impl(*args, **kwargs)


def bind_runner_sampler(*, runtime: Any, runner: Any) -> None:
    sampler = getattr(runner, "sampler", None)
    if sampler is None:
        return
    setattr(sampler, "_tllm_runtime", runtime)
    setattr(sampler, "_tllm_runner", runner)
    setattr(runner, "_tllm_runtime", runtime)


def ensure_sampler_precompute_buffers(*, runtime: Any, runner: Any) -> None:
    provider = _resolve_provider(runtime)
    if provider is None or not provider.is_active():
        return
    ensure = getattr(provider, "ensure_runtime_buffers", None)
    if callable(ensure):
        ensure(runtime=runtime, runner=runner)


def maybe_prepare_sampler_decode_step(*, runtime: Any, runner: Any) -> None:
    provider = _resolve_provider(runtime)
    if provider is None or not provider.is_active():
        precompute = runtime.sampler_precompute
        precompute.reset_decode_step(int(getattr(runtime, "event_step_id", -1)))
        precompute.port_enabled = False
        return
    runtime.sampler_precompute.port_enabled = True
    prepare = getattr(provider, "maybe_prepare_decode_step", None)
    if callable(prepare):
        prepare(runtime=runtime, runner=runner)


def runtime_has_active_sampler_provider(runtime: Any) -> bool:
    provider = _resolve_provider(runtime)
    return bool(provider is not None and provider.is_active())


def maybe_capture_source_precompute(*, runtime: Any, runner: Any, layer_path: str) -> None:
    provider = _resolve_provider(runtime)
    if provider is None or not provider.is_active():
        return
    capture = getattr(provider, "maybe_capture_source_precompute", None)
    if callable(capture):
        capture(runtime=runtime, runner=runner, layer_path=layer_path)


def _vanilla_sample(*, sampler: Any, logits: torch.Tensor, sampling_metadata: Any) -> torch.Tensor:
    assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
    if sampling_metadata.all_random:
        greedy_sampled = None
    else:
        greedy_sampled = sampler.greedy_sample(logits)
        if sampling_metadata.all_greedy:
            return greedy_sampled

    assert sampling_metadata.temperature is not None
    logits = sampler.apply_temperature(logits, sampling_metadata.temperature)
    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        logits = processor.apply(logits)

    logits = apply_min_p(logits, getattr(sampling_metadata, "min_p", None))

    random_sampled = sampler.topk_topp_sampler(
        logits,
        sampling_metadata.generators,
        sampling_metadata.top_k,
        sampling_metadata.top_p,
    )
    if greedy_sampled is None:
        return random_sampled.to(dtype=torch.long)
    return torch.where(
        sampling_metadata.temperature < _SAMPLING_EPS,
        greedy_sampled,
        random_sampled,
        out=greedy_sampled,
    ).to(dtype=torch.long)


def _maybe_sample_precomputed_dense_fast(
    *,
    runtime: Any,
    provider: Any,
    sampler: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
    greedy_sampled: torch.Tensor | None,
) -> torch.Tensor | None:
    with _maybe_record_function("distiller.precomputed_dense_fast_path"):
        if greedy_sampled is not None:
            return None
        if bool(getattr(sampling_metadata, "generators", {})):
            return None
        cache = runtime.sampler_precompute.cache_for_step(int(getattr(runtime, "event_step_id", -2)))
        if cache is None:
            return None
        if not bool(cache.all_rows):
            return None
        dense = cache.dense_logits
        if dense is None:
            return None
        beta = float(getattr(getattr(provider, "config", None), "distiller_beta", 0.0) or 0.0)
        if beta == 0.0:
            return None
        backend = str(getattr(getattr(provider, "config", None), "distiller_sampler_backend", "") or "")
        if backend != "pre_filter_dense":
            return None
        dense = dense.to(device=logits.device, dtype=logits.dtype)
        logits.mul_(1.0 + beta)
        logits.add_(dense, alpha=-beta)
        return sampler.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        ).to(dtype=torch.long)


def wrapped_sampler_sample(*, sampler: Any, logits: torch.Tensor, sampling_metadata: Any) -> torch.Tensor:
    runtime = getattr(sampler, "_tllm_runtime", None)
    runner = getattr(sampler, "_tllm_runner", None)
    provider = _resolve_provider(runtime)
    if runtime is None or runner is None or provider is None or not provider.is_active():
        if getattr(sampling_metadata, "min_p", None) is not None:
            return _vanilla_sample(sampler=sampler, logits=logits, sampling_metadata=sampling_metadata)
        if _ORIG_VLLM_SAMPLER_SAMPLE is not None:
            return _ORIG_VLLM_SAMPLER_SAMPLE(sampler, logits, sampling_metadata)
        return _vanilla_sample(sampler=sampler, logits=logits, sampling_metadata=sampling_metadata)

    assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
    if sampling_metadata.all_random:
        greedy_sampled = None
    else:
        greedy_sampled = sampler.greedy_sample(logits)

    assert sampling_metadata.temperature is not None
    logits_for_sampling = sampler.apply_temperature(logits, sampling_metadata.temperature)
    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        logits_for_sampling = processor.apply(logits_for_sampling)

    fast_sampled = _maybe_sample_precomputed_dense_fast(
        runtime=runtime,
        provider=provider,
        sampler=sampler,
        logits=logits_for_sampling,
        sampling_metadata=sampling_metadata,
        greedy_sampled=greedy_sampled,
    )
    if fast_sampled is not None:
        return fast_sampled

    view = build_sampler_step_view(
        runtime=runtime,
        runner=runner,
        model=getattr(runner, "model", None),
        logits=logits_for_sampling,
        sampling_metadata=sampling_metadata,
    )
    sampled = sample_with_optional_modifier(
        sampler=sampler,
        logits=logits_for_sampling,
        sampling_metadata=sampling_metadata,
        view=view,
        provider=provider,
        greedy_sampled=greedy_sampled,
    )
    if sampled is not None:
        return sampled.to(dtype=torch.long)
    if _ORIG_VLLM_SAMPLER_SAMPLE is not None:
        return _ORIG_VLLM_SAMPLER_SAMPLE(sampler, logits, sampling_metadata)
    return _vanilla_sample(sampler=sampler, logits=logits, sampling_metadata=sampling_metadata)


def maybe_schedule_sampler_precompute(*, runtime: Any, runner: Any, layer_path: str) -> None:
    if str(layer_path).strip() != str(getattr(runtime, "source_resolved_path", "") or "").strip():
        return
    provider = _resolve_provider(runtime)
    if provider is None or not provider.is_active():
        return
    schedule = getattr(provider, "maybe_schedule_precompute", None)
    if callable(schedule):
        schedule(runtime=runtime, runner=runner)


def install_sampler_patch(*, core: Any) -> None:
    global _PATCH_INSTALLED, _ORIG_VLLM_SAMPLER_SAMPLE
    if _PATCH_INSTALLED:
        return
    from vllm.v1.sample.sampler import Sampler

    _ORIG_VLLM_SAMPLER_SAMPLE = Sampler.sample

    def _wrapped(self: Any, logits: torch.Tensor, sampling_metadata: Any) -> torch.Tensor:
        return wrapped_sampler_sample(sampler=self, logits=logits, sampling_metadata=sampling_metadata)

    Sampler.sample = _wrapped
    core._ORIG_VLLM_SAMPLER_SAMPLE = _ORIG_VLLM_SAMPLER_SAMPLE
    _PATCH_INSTALLED = True
