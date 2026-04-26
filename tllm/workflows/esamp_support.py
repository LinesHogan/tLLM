#!/usr/bin/env python3
"""Workflow helpers for ESamp on top of generic residual runtime."""

from __future__ import annotations

import time
from typing import Sequence

import torch

from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig
from tllm.consumers.esamp.initializers.svd import SVDModelBankInitializerConfig
from tllm.runtime import residual_runtime as runtime
from tllm.runtime.vllm_patch import capture_runner as _capture_runner


def configure_esamp_runtime(
    *,
    graph_scratch_rows: int = 0,
    tap_layer_paths: Sequence[str] | None = None,
    source_layer_path: str = "model.model.layers[0].input_layernorm",
    target_layer_path: str = "model.model.layers[-1].input_layernorm",
    enable_esamp_training: bool = True,
    distiller_hidden_dim: int = 128,
    distiller_lr: float = 1e-3,
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
    distiller_sampler_backend: str = "post_filter_exact",
    compact_capture_lane: bool = False,
) -> ESampConsumer:
    tap_layer_paths = tuple(tap_layer_paths or (source_layer_path, target_layer_path))
    config = ESampConsumerConfig(
        graph_scratch_rows=int(graph_scratch_rows),
        source_layer_path=str(source_layer_path),
        target_layer_path=str(target_layer_path),
        enable_esamp_training=bool(enable_esamp_training),
        distiller_hidden_dim=int(distiller_hidden_dim),
        distiller_lr=float(distiller_lr),
        per_request_models=bool(per_request_models),
        per_request_model_bank=bool(per_request_model_bank),
        model_bank_slots=int(model_bank_slots),
        model_bank_flush_interval=int(model_bank_flush_interval),
        model_bank_rank=int(model_bank_rank),
        model_bank_use_output_layernorm=bool(model_bank_use_output_layernorm),
        model_bank_initializer=model_bank_initializer,
        model_bank_train_cudagraph=bool(model_bank_train_cudagraph),
        model_bank_forward_backend=str(model_bank_forward_backend).strip() or "torch",
        adaptation_pipeline_slots=max(1, int(adaptation_pipeline_slots)),
        adaptation_stream_mode=str(adaptation_stream_mode).strip() or "dual",
        adaptation_stream_priority=int(adaptation_stream_priority),
        trace_per_request_losses=bool(trace_per_request_losses),
        trace_interval=max(1, int(trace_interval)),
        trace_max_points=max(0, int(trace_max_points)),
        enable_distiller_intervention=bool(enable_distiller_intervention),
        distiller_beta=float(distiller_beta),
        distiller_sampler_backend=str(distiller_sampler_backend).strip() or "post_filter_exact",
        compact_capture_lane=bool(compact_capture_lane),
    )

    current = runtime.RUNTIME.consumer
    if isinstance(current, ESampConsumer):
        consumer = current
        consumer.synchronize()
        consumer.update_config(config)
    else:
        consumer = ESampConsumer(config)
    runtime.clear_dispatch_consumers()
    runtime.set_runtime_consumer(consumer)
    runtime.configure_runtime(
        graph_scratch_rows=int(graph_scratch_rows),
        tap_layer_paths=tap_layer_paths,
        source_layer_path=str(source_layer_path),
        target_layer_path=str(target_layer_path),
        enable_esamp_training=bool(enable_esamp_training),
        distiller_hidden_dim=int(distiller_hidden_dim),
        distiller_lr=float(distiller_lr),
        per_request_models=bool(per_request_models),
        per_request_model_bank=bool(per_request_model_bank),
        model_bank_slots=int(model_bank_slots),
        model_bank_flush_interval=int(model_bank_flush_interval),
        model_bank_rank=int(model_bank_rank),
        model_bank_use_output_layernorm=bool(model_bank_use_output_layernorm),
        model_bank_initializer=model_bank_initializer,
        model_bank_train_cudagraph=bool(model_bank_train_cudagraph),
        model_bank_forward_backend=str(model_bank_forward_backend).strip() or "torch",
        adaptation_pipeline_slots=max(1, int(adaptation_pipeline_slots)),
        adaptation_stream_mode=str(adaptation_stream_mode).strip() or "dual",
        adaptation_stream_priority=int(adaptation_stream_priority),
        trace_per_request_losses=bool(trace_per_request_losses),
        trace_interval=max(1, int(trace_interval)),
        trace_max_points=max(0, int(trace_max_points)),
        enable_distiller_intervention=bool(enable_distiller_intervention),
        distiller_beta=float(distiller_beta),
        distiller_sampler_backend=str(distiller_sampler_backend).strip() or "post_filter_exact",
        compact_capture_lane=bool(compact_capture_lane),
    )
    return consumer


def run_generate_with_request_mapping(
    llm: object,
    prompts: Sequence[str],
    params: Sequence[object],
    request_prompt_indices: Sequence[int] | None = None,
    request_sample_indices: Sequence[int] | None = None,
) -> object:
    return _capture_runner.run_generate_with_request_mapping(
        runtime=runtime.RUNTIME,
        llm=llm,
        prompts=prompts,
        params=params,
        request_prompt_indices=request_prompt_indices,
        request_sample_indices=request_sample_indices,
    )


def run_esamp_throughput_case(
    *,
    llm: object,
    prompts: list[str],
    max_new_tokens: int,
    warmup_rounds: int,
    rounds: int,
    train_enabled: bool,
    ignore_eos: bool,
    log_memory: bool,
    seed_base: int = 9000,
) -> dict[str, float]:
    from tllm.util import tools as _tool_helpers

    runtime.set_esamp_training_enabled(train_enabled)
    runtime.synchronize_esamp()
    _ = runtime.read_and_reset_esamp_stats(sync=True)

    params = [
        _tool_helpers.build_greedy_params(
            max_new_tokens=max_new_tokens,
            seed=seed_base + index,
            ignore_eos=ignore_eos,
        )
        for index in range(len(prompts))
    ]

    tag = "esamp" if train_enabled else "tap_only"
    if log_memory:
        _tool_helpers.print_gpu_mem(f"[{tag}] before_warmup")

    for warmup_index in range(warmup_rounds):
        run_generate_with_request_mapping(llm, prompts, params)
        if log_memory:
            _tool_helpers.print_gpu_mem(f"[{tag}] warmup_{warmup_index + 1}/{warmup_rounds}")

    torch.cuda.synchronize()
    start_t = time.perf_counter()
    total_requests = 0
    total_output_tokens = 0

    for round_index in range(rounds):
        outputs = run_generate_with_request_mapping(llm, prompts, params)
        total_requests += len(outputs)
        total_output_tokens += _tool_helpers.sum_output_tokens(outputs)
        if log_memory:
            _tool_helpers.print_gpu_mem(f"[{tag}] round_{round_index + 1}/{rounds}")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_t
    stats = runtime.read_and_reset_esamp_stats(sync=True)

    req_per_s = float(total_requests / elapsed) if elapsed > 0 else 0.0
    out_tok_per_s = float(total_output_tokens / elapsed) if elapsed > 0 else 0.0
    return {
        "elapsed_s": float(elapsed),
        "requests": float(total_requests),
        "output_tokens": float(total_output_tokens),
        "req_per_s": req_per_s,
        "out_tok_per_s": out_tok_per_s,
        "loss_avg": float(stats.loss_avg),
        "loss_count": float(stats.loss_count),
    }


__all__ = [
    "configure_esamp_runtime",
    "run_generate_with_request_mapping",
    "run_esamp_throughput_case",
]
