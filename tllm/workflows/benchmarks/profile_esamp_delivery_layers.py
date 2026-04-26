#!/usr/bin/env python3
"""Profile ESamp delivery-layer overhead with maintained labels."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Sequence

import torch

from tllm.consumers.esamp.initializers.svd import SVDModelBankInitializerConfig
from tllm.runtime import residual_runtime as core
from tllm.util.tools import build_prompt_batch, read_prompts, shutdown_llm_instance
from tllm.workflows import esamp_support
from tllm.workflows.benchmarks import per_request_esamp_benchmark as per_request_bench
from tllm.workflows.common import (
    build_sampling_params,
    sum_all_candidate_tokens,
    sum_all_completions,
)

JSON_SUMMARY_PREFIX = "ESAMP_DELIVERY_LAYERS_JSON:"


@dataclass(frozen=True)
class LayerMode:
    name: str
    configure_enabled: bool
    per_request_models: bool
    per_request_model_bank: bool
    runtime_enabled: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", type=str, default="")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--benchmark-batch-size", type=int, default=4)
    parser.add_argument("--sampling-n", type=int, default=8)
    parser.add_argument("--benchmark-max-new-tokens", type=int, default=64)
    parser.add_argument("--benchmark-warmup-rounds", type=int, default=1)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument("--sampling-temperature", type=float, default=0.8)
    parser.add_argument("--sampling-top-p", type=float, default=0.95)
    parser.add_argument("--sampling-top-k", type=int, default=-1)
    parser.add_argument("--sampling-min-p", type=float, default=0.0)
    parser.add_argument("--graph-scratch-rows", type=int, default=0)
    parser.add_argument("--source-layer-path", type=str, default="model.model.layers[0].input_layernorm")
    parser.add_argument("--target-layer-path", type=str, default="model.model.layers[-1].input_layernorm")
    parser.add_argument("--distiller-hidden-dim", type=int, default=256)
    parser.add_argument("--distiller-lr", type=float, default=1e-3)
    parser.add_argument("--model-bank-slots", type=int, default=0)
    parser.add_argument("--model-bank-flush-interval", type=int, default=1)
    parser.add_argument("--model-bank-rank", type=int, default=64)
    parser.add_argument("--adaptation-pipeline-slots", type=int, default=4)
    parser.add_argument("--adaptation-stream-mode", type=str, default="dual", choices=["dual", "single", "serial"])
    parser.add_argument("--adaptation-stream-priority", type=int, default=0)
    parser.add_argument("--model-bank-train-cudagraph", action="store_true")
    parser.add_argument("--no-model-bank-train-cudagraph", dest="model_bank_train_cudagraph", action="store_false")
    parser.add_argument("--model-bank-initializer", type=str, default="svd", choices=["none", "svd"])
    parser.add_argument(
        "--model-bank-initializer-svd-method",
        type=str,
        default="ffn_fast_svd",
        choices=["ridge_svd", "ridge-svd", "ridge+svd", "ffn_fast_svd", "ffn-fast-svd", "ffn+svd"],
    )
    parser.add_argument("--model-bank-initializer-svd-ridge-lambda", type=float, default=1e-2)
    parser.add_argument("--model-bank-initializer-svd-min-rows", type=int, default=32)
    parser.add_argument("--model-bank-initializer-svd-max-wait-steps", type=int, default=4)
    parser.add_argument("--cooldown-s", type=float, default=1.0)
    parser.add_argument("--emit-json-summary", action="store_true")
    parser.set_defaults(model_bank_train_cudagraph=True)
    return parser.parse_args()


def _build_model_bank_initializer_config(args: argparse.Namespace) -> SVDModelBankInitializerConfig | None:
    if str(args.model_bank_initializer).strip().lower() != "svd":
        return None
    return SVDModelBankInitializerConfig(
        method=str(args.model_bank_initializer_svd_method).strip().lower(),
        ridge_lambda=float(args.model_bank_initializer_svd_ridge_lambda),
        min_rows=int(args.model_bank_initializer_svd_min_rows),
        max_wait_steps=int(args.model_bank_initializer_svd_max_wait_steps),
    )


def _configure_esamp(
    *,
    args: argparse.Namespace,
    rows: int,
    effective_batch_cap: int,
    prompt_count: int,
    enabled: bool,
    per_request_models: bool,
    per_request_model_bank: bool,
) -> None:
    slots = per_request_bench._resolve_model_bank_slots(
        int(args.model_bank_slots),
        effective_batch_cap=int(effective_batch_cap),
        prompt_count=int(prompt_count),
    )
    esamp_support.configure_esamp_runtime(
        graph_scratch_rows=int(rows),
        tap_layer_paths=[args.source_layer_path, args.target_layer_path],
        source_layer_path=str(args.source_layer_path),
        target_layer_path=str(args.target_layer_path),
        enable_esamp_training=bool(enabled),
        distiller_hidden_dim=int(args.distiller_hidden_dim),
        distiller_lr=float(args.distiller_lr),
        enable_distiller_intervention=False,
        distiller_beta=0.0,
        per_request_models=bool(per_request_models),
        per_request_model_bank=bool(per_request_model_bank),
        model_bank_slots=int(slots),
        model_bank_flush_interval=int(args.model_bank_flush_interval),
        model_bank_rank=int(args.model_bank_rank),
        model_bank_use_output_layernorm=True,
        model_bank_initializer=_build_model_bank_initializer_config(args),
        model_bank_train_cudagraph=bool(args.model_bank_train_cudagraph),
        model_bank_forward_backend="torch",
        adaptation_pipeline_slots=max(1, int(args.adaptation_pipeline_slots)),
        adaptation_stream_mode=str(args.adaptation_stream_mode),
        adaptation_stream_priority=int(args.adaptation_stream_priority),
    )


def _run_one_mode(
    *,
    mode: LayerMode,
    args: argparse.Namespace,
    rows: int,
    effective_batch_cap: int,
    prompt_count: int,
    llm,
    prompts: Sequence[str],
    params: Sequence[object],
    request_prompt_indices: Sequence[int],
    request_sample_indices: Sequence[int],
) -> dict[str, object]:
    _configure_esamp(
        args=args,
        rows=int(rows),
        effective_batch_cap=int(effective_batch_cap),
        prompt_count=int(prompt_count),
        enabled=bool(mode.configure_enabled),
        per_request_models=bool(mode.per_request_models),
        per_request_model_bank=bool(mode.per_request_model_bank),
    )
    core.set_esamp_training_enabled(bool(mode.runtime_enabled))
    core.synchronize_esamp()
    _ = core.read_and_reset_esamp_stats(sync=True)
    _ = core.read_and_reset_distiller_timing_stats(sync=True)

    def _run_generate():
        return esamp_support.run_generate_with_request_mapping(
            llm,
            list(prompts),
            list(params),
            request_prompt_indices=list(request_prompt_indices),
            request_sample_indices=list(request_sample_indices),
        )

    for _ in range(int(args.benchmark_warmup_rounds)):
        _run_generate()

    torch.cuda.synchronize()
    start = time.perf_counter()
    total_requests = 0
    total_completions = 0
    total_tokens = 0
    for _ in range(int(args.benchmark_rounds)):
        outputs = _run_generate()
        total_requests += len(outputs)
        total_completions += sum_all_completions(outputs)
        total_tokens += sum_all_candidate_tokens(outputs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    stats = core.read_and_reset_esamp_stats(sync=True)
    graph = core.read_graph_debug_stats(mode="model_bank")
    return {
        "name": mode.name,
        "elapsed_s": float(elapsed),
        "requests": int(total_requests),
        "completions": int(total_completions),
        "output_tokens": int(total_tokens),
        "req_per_s": float(total_requests / elapsed) if elapsed > 0 else 0.0,
        "completion_per_s": float(total_completions / elapsed) if elapsed > 0 else 0.0,
        "out_tok_per_s": float(total_tokens / elapsed) if elapsed > 0 else 0.0,
        "loss_avg": float(stats.loss_avg),
        "loss_count": int(stats.loss_count),
        "model_bank_graph_captured": graph.capture_state == "captured",
        "model_bank_graph_replay_attempt_count": int(graph.replay_attempt_count),
        "model_bank_graph_replay_hit_count": int(graph.replay_hit_count),
        "model_bank_graph_kernel_fallback_count": int(graph.kernel_fallback_count),
        "adaptation_stream_mode": str(args.adaptation_stream_mode),
        "adaptation_stream_priority": int(args.adaptation_stream_priority),
    }


def main() -> int:
    args = _parse_args()
    if int(args.sampling_n) <= 1:
        raise RuntimeError("--sampling-n must be > 1 for ESamp delivery-layer profiling")

    base_prompts = build_prompt_batch(read_prompts(args.prompt_file, args.prompt), int(args.benchmark_batch_size))
    prompts, prompt_indices, sample_indices = per_request_bench._expand_requests_for_effective_n(
        base_prompts,
        int(args.sampling_n),
    )
    effective_batch_cap = len(prompts)
    rows = int(args.graph_scratch_rows) if int(args.graph_scratch_rows) > 0 else max(64, effective_batch_cap)
    params = build_sampling_params(
        prompts=prompts,
        max_new_tokens=int(args.benchmark_max_new_tokens),
        sampling_n=1,
        sampling_temperature=float(args.sampling_temperature),
        sampling_top_p=float(args.sampling_top_p),
        sampling_top_k=int(args.sampling_top_k),
        sampling_min_p=float(args.sampling_min_p),
        ignore_eos=True,
        sampling_seed=None,
        sampling_per_request_seed=False,
    )

    modes = [
        LayerMode("dispatch_off_config_false", False, False, False, False),
        LayerMode("tap_only_model_bank_compact", True, True, True, False),
        LayerMode("model_bank_train", True, True, True, True),
        LayerMode("tap_only_shared_rows", True, False, False, False),
        LayerMode("shared_train", True, False, False, True),
    ]

    _configure_esamp(
        args=args,
        rows=int(rows),
        effective_batch_cap=int(effective_batch_cap),
        prompt_count=len(base_prompts),
        enabled=True,
        per_request_models=True,
        per_request_model_bank=True,
    )
    llm = core.make_llm(
        model_name=str(args.model_name),
        dtype=str(args.dtype),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        enable_prefix_caching=False,
        enforce_eager=False,
    )

    results: list[dict[str, object]] = []
    try:
        for mode in modes:
            result = _run_one_mode(
                mode=mode,
                args=args,
                rows=int(rows),
                effective_batch_cap=int(effective_batch_cap),
                prompt_count=len(base_prompts),
                llm=llm,
                prompts=prompts,
                params=params,
                request_prompt_indices=prompt_indices,
                request_sample_indices=sample_indices,
            )
            results.append(result)
            print(
                f"{mode.name}: tok/s={float(result['out_tok_per_s']):.3f} "
                f"loss_count={int(result['loss_count'])}"
            )
    finally:
        shutdown_llm_instance(llm, cooldown_s=float(args.cooldown_s))

    base = float(results[0]["out_tok_per_s"]) if results else 0.0
    for result in results:
        result["tok_ratio_vs_dispatch_off"] = (
            float(result["out_tok_per_s"]) / base if base > 0 else 0.0
        )

    payload = {"implementation": "esamp", "results": results}
    if args.emit_json_summary:
        print(JSON_SUMMARY_PREFIX + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
