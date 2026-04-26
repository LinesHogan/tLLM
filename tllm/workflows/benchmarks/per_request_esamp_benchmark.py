#!/usr/bin/env python3
"""Benchmark n>1 ESamp generation with single/per-request/model-bank variants.

Compares four modes:
1) single distiller, train off
2) single distiller, train on
3) per-request distillers, train on
4) per-request model-bank, train on

Also reports per-request loss trajectories for mode (3).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, cast

import torch
from vllm import SamplingParams

from tllm.consumers.esamp.initializers.svd import SVDModelBankInitializerConfig
from tllm.consumers.esamp.model_bank_backend import normalize_model_bank_forward_backend
from tllm.workflows.common import (
    build_sampling_params as _build_sampling_params,
    sum_all_candidate_tokens as _sum_all_candidate_tokens,
    sum_all_completions as _sum_all_completions,
)
from tllm.runtime import residual_runtime as core
from tllm.workflows import esamp_support as esamp_workflow_support
from tllm.util.tools import build_prompt_batch, read_prompts, shutdown_llm_instance
from tllm.runtime.sampler_bridge.types import SAMPLER_BACKEND_CHOICES

JSON_SUMMARY_PREFIX = "PER_REQUEST_JSON_SUMMARY:"
_TRACE_DISTILLER_TIMING = os.getenv("TLLM_TRACE_DISTILLER_TIMING", "") == "1"


def _path_hotspot_fields(stats: object) -> Dict[str, float]:
    fields: Dict[str, float] = {}
    totals = getattr(stats, "cpu_ms_total", {}) or {}
    counts = getattr(stats, "counts", {}) or {}
    for raw_name, raw_total in sorted(totals.items()):
        name = str(raw_name).strip().replace(".", "_")
        if not name:
            continue
        total = float(raw_total)
        count = float(counts.get(raw_name, 0.0) or 0.0)
        fields[f"path_hotspot_{name}_ms_total"] = total
        fields[f"path_hotspot_{name}_count"] = count
        fields[f"path_hotspot_{name}_ms_avg"] = float(total / count) if count > 0 else 0.0
    return fields


def _maybe_print_distiller_timing(case_name: str, case: Dict[str, float]) -> None:
    if not _TRACE_DISTILLER_TIMING:
        return
    pre_avg = float(case.get("distiller_precompute_ms_avg", 0.0) or 0.0)
    pre_cnt = int(case.get("distiller_precompute_count", 0.0) or 0.0)
    wait_avg = float(case.get("distiller_wait_ms_avg", 0.0) or 0.0)
    wait_cnt = int(case.get("distiller_wait_count", 0.0) or 0.0)
    print(
        f"{case_name}_distiller_timing: "
        f"port_publish_attempt_count={int(case.get('distiller_port_publish_attempt_count', 0.0) or 0.0)} "
        f"port_publish_hit_count={int(case.get('distiller_port_publish_hit_count', 0.0) or 0.0)} "
        f"precompute_ms_avg={pre_avg:.4f} precompute_count={pre_cnt} "
        f"wait_ms_avg={wait_avg:.4f} wait_count={wait_cnt} "
        f"fallback_ms_avg={float(case.get('distiller_fallback_ms_avg', 0.0) or 0.0):.4f} "
        f"fallback_count={int(case.get('distiller_fallback_count', 0.0) or 0.0)} "
        f"schedule_attempt_count={int(case.get('distiller_schedule_attempt_count', 0.0) or 0.0)} "
        f"schedule_hit_count={int(case.get('distiller_schedule_hit_count', 0.0) or 0.0)} "
        f"candidate_sample_count={int(case.get('distiller_candidate_sample_count', 0.0) or 0.0)} "
        f"candidate_avg_count={float(case.get('distiller_candidate_avg_count', 0.0) or 0.0):.2f} "
        f"candidate_avg_per_row={float(case.get('distiller_candidate_avg_per_row', 0.0) or 0.0):.2f} "
        f"candidate_max_count={int(case.get('distiller_candidate_max_count', 0.0) or 0.0)} "
        f"candidate_kernel_triton_count={int(case.get('distiller_candidate_kernel_triton_count', 0.0) or 0.0)} "
        f"candidate_kernel_torch_count={int(case.get('distiller_candidate_kernel_torch_count', 0.0) or 0.0)} "
        f"candidate_kernel_fallback_count={int(case.get('distiller_candidate_kernel_fallback_count', 0.0) or 0.0)}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", type=str, default="")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")

    parser.add_argument("--graph-scratch-rows", type=int, default=0)
    parser.add_argument("--source-layer-path", type=str, default="model.model.layers[0].input_layernorm")
    parser.add_argument("--target-layer-path", type=str, default="model.model.layers[-1].input_layernorm")
    parser.add_argument("--distiller-hidden-dim", type=int, default=256)
    parser.add_argument("--distiller-lr", type=float, default=1e-3)
    parser.add_argument("--enable-distiller-intervention", action="store_true")
    parser.add_argument("--distiller-beta", type=float, default=0.0)
    parser.add_argument(
        "--distiller-sampler-backend",
        type=str,
        default="post_filter_exact",
        choices=SAMPLER_BACKEND_CHOICES,
    )
    parser.add_argument("--emit-json-summary", action="store_true")
    parser.add_argument("--model-bank-slots", type=int, default=0)
    parser.add_argument("--model-bank-flush-interval", type=int, default=1)
    parser.add_argument("--model-bank-rank", type=int, default=64)
    parser.add_argument("--adaptation-pipeline-slots", type=int, default=4)
    parser.add_argument("--adaptation-stream-mode", type=str, default="dual", choices=["dual", "single", "serial"])
    parser.add_argument("--adaptation-stream-priority", type=int, default=0)
    parser.add_argument("--model-bank-use-output-layernorm", action="store_true")
    parser.add_argument("--no-model-bank-use-output-layernorm", dest="model_bank_use_output_layernorm", action="store_false")
    parser.add_argument("--model-bank-initializer", type=str, default="none", choices=["none", "svd"])
    parser.add_argument(
        "--model-bank-initializer-svd-method",
        type=str,
        default="ffn_fast_svd",
        choices=["ridge_svd", "ridge-svd", "ridge+svd", "ffn_fast_svd", "ffn-fast-svd", "ffn+svd"],
    )
    parser.add_argument("--model-bank-initializer-svd-ridge-lambda", type=float, default=1e-2)
    parser.add_argument("--model-bank-initializer-svd-min-rows", type=int, default=32)
    parser.add_argument("--model-bank-initializer-svd-max-wait-steps", type=int, default=4)
    parser.add_argument("--model-bank-train-cudagraph", action="store_true")
    parser.add_argument("--model-bank-compact-capture", action="store_true")
    parser.add_argument(
        "--no-model-bank-compact-capture",
        dest="model_bank_compact_capture",
        action="store_false",
    )
    parser.add_argument(
        "--model-bank-forward-backend",
        type=str,
        default="torch",
        choices=["torch", "reference", "default", "triton", "triton_grouped", "experimental_triton_grouped"],
    )
    parser.add_argument(
        "--no-model-bank-train-cudagraph",
        dest="model_bank_train_cudagraph",
        action="store_false",
    )
    parser.add_argument("--run-model-bank-case", action="store_true")
    parser.add_argument("--no-run-model-bank-case", dest="run_model_bank_case", action="store_false")

    parser.add_argument("--benchmark-batch-size", type=int, default=8)
    parser.add_argument("--benchmark-max-new-tokens", type=int, default=64)
    parser.add_argument("--benchmark-warmup-rounds", type=int, default=1)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument("--benchmark-ignore-eos", action="store_true")
    parser.add_argument("--benchmark-disable-prefix-caching", action="store_true")

    parser.add_argument("--sampling-n", type=int, default=16)
    parser.add_argument("--sampling-temperature", type=float, default=0.8)
    parser.add_argument("--sampling-top-p", type=float, default=0.95)
    parser.add_argument("--sampling-top-k", type=int, default=-1)
    parser.add_argument("--sampling-min-p", type=float, default=0.0)
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help="Shared seed for all requests. Omit to avoid per-request generators.",
    )
    parser.add_argument(
        "--sampling-per-request-seed",
        action="store_true",
        help="If set, use (sampling_seed + request_idx).",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="Deprecated alias of --sampling-seed.",
    )
    parser.add_argument("--trajectory-topk", type=int, default=8)
    parser.add_argument("--trajectory-step-interval", type=int, default=1)
    parser.add_argument("--trajectory-max-points", type=int, default=1024)
    parser.add_argument("--cooldown-s", type=float, default=1.0)
    parser.add_argument(
        "--effective-batch-cap",
        type=int,
        default=0,
        help="Chunk size for effective expanded batch in V1 n>1 emulation; <=0 means full effective batch.",
    )
    parser.add_argument(
        "--case-filter",
        type=str,
        default="all",
        choices=["all", "single_off", "single_on", "per_request_on", "model_bank_on"],
        help="Run only one benchmark case inside this process.",
    )
    parser.add_argument(
        "--skip-trajectory",
        action="store_true",
        help="Skip per-request trajectory collection.",
    )

    parser.set_defaults(enforce_eager=False)
    parser.set_defaults(benchmark_ignore_eos=True)
    parser.set_defaults(benchmark_disable_prefix_caching=True)
    parser.set_defaults(run_model_bank_case=True)
    parser.set_defaults(model_bank_use_output_layernorm=True)
    parser.set_defaults(model_bank_train_cudagraph=True)
    parser.set_defaults(model_bank_compact_capture=False)
    return parser.parse_args()


def _expand_requests_for_effective_n(
    prompts: Sequence[str],
    sampling_n: int,
) -> tuple[List[str], List[int], List[int]]:
    """Expand (prompt, n) into n independent requests for V1 compatibility."""
    n = max(1, int(sampling_n))
    expanded_prompts: List[str] = []
    prompt_indices: List[int] = []
    sample_indices: List[int] = []
    for prompt_idx, prompt in enumerate(prompts):
        for sample_idx in range(n):
            expanded_prompts.append(prompt)
            prompt_indices.append(int(prompt_idx))
            sample_indices.append(int(sample_idx))
    return expanded_prompts, prompt_indices, sample_indices


def _build_model_bank_initializer_config(args: argparse.Namespace) -> SVDModelBankInitializerConfig | None:
    if str(args.model_bank_initializer).strip().lower() != "svd":
        return None
    return SVDModelBankInitializerConfig(
        method=str(args.model_bank_initializer_svd_method).strip().lower(),
        ridge_lambda=float(args.model_bank_initializer_svd_ridge_lambda),
        min_rows=int(args.model_bank_initializer_svd_min_rows),
        max_wait_steps=int(args.model_bank_initializer_svd_max_wait_steps),
    )


def _resolve_effective_batch_cap(effective_batch: int, requested_cap: int) -> int:
    total = max(1, int(effective_batch))
    cap = int(requested_cap)
    if cap <= 0:
        return total
    return min(total, max(1, cap))


def _resolve_graph_scratch_rows(requested_rows: int, *, effective_batch_cap: int) -> int:
    rows = int(requested_rows)
    if rows > 0:
        return rows
    return max(64, int(effective_batch_cap))


def _resolve_model_bank_slots(
    requested_slots: int,
    *,
    effective_batch_cap: int,
    prompt_count: int,
) -> int:
    slots = int(requested_slots)
    if slots > 0:
        return slots
    _ = effective_batch_cap
    return max(1, int(prompt_count))


def _adaptation_pipeline_slots(args: argparse.Namespace) -> int:
    return max(1, int(getattr(args, "adaptation_pipeline_slots", 4)))


def _requires_isolated_llm_cases(*, enable_distiller_intervention: bool, distiller_beta: float = 0.0) -> bool:
    return bool(enable_distiller_intervention) and float(distiller_beta) != 0.0


def _append_bool_flag(cmd: List[str], enabled: bool, positive_flag: str, negative_flag: str | None = None) -> None:
    if enabled:
        cmd.append(positive_flag)
    elif negative_flag is not None:
        cmd.append(negative_flag)


def _build_isolated_case_subprocess_cmd(
    args: argparse.Namespace,
    *,
    case_filter: str,
    skip_trajectory: bool,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "tllm.workflows.benchmarks.per_request_esamp_benchmark",
        "--model-name",
        str(args.model_name),
        "--dtype",
        str(args.dtype),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--graph-scratch-rows",
        str(args.graph_scratch_rows),
        "--source-layer-path",
        str(args.source_layer_path),
        "--target-layer-path",
        str(args.target_layer_path),
        "--distiller-hidden-dim",
        str(args.distiller_hidden_dim),
        "--distiller-lr",
        str(args.distiller_lr),
        "--distiller-beta",
        str(getattr(args, "distiller_beta", 0.0)),
        "--distiller-sampler-backend",
        str(getattr(args, "distiller_sampler_backend", "post_filter_exact")),
        "--model-bank-slots",
        str(args.model_bank_slots),
        "--model-bank-flush-interval",
        str(args.model_bank_flush_interval),
        "--model-bank-rank",
        str(args.model_bank_rank),
        "--adaptation-pipeline-slots",
        str(_adaptation_pipeline_slots(args)),
        "--adaptation-stream-mode",
        str(getattr(args, "adaptation_stream_mode", "dual")),
        "--adaptation-stream-priority",
        str(getattr(args, "adaptation_stream_priority", 0)),
        "--model-bank-forward-backend",
        str(getattr(args, "model_bank_forward_backend", "torch")),
        "--model-bank-initializer",
        str(args.model_bank_initializer),
        "--model-bank-initializer-svd-method",
        str(args.model_bank_initializer_svd_method),
        "--model-bank-initializer-svd-ridge-lambda",
        str(args.model_bank_initializer_svd_ridge_lambda),
        "--model-bank-initializer-svd-min-rows",
        str(args.model_bank_initializer_svd_min_rows),
        "--model-bank-initializer-svd-max-wait-steps",
        str(args.model_bank_initializer_svd_max_wait_steps),
        "--benchmark-batch-size",
        str(args.benchmark_batch_size),
        "--benchmark-max-new-tokens",
        str(args.benchmark_max_new_tokens),
        "--benchmark-warmup-rounds",
        str(args.benchmark_warmup_rounds),
        "--benchmark-rounds",
        str(args.benchmark_rounds),
        "--sampling-n",
        str(args.sampling_n),
        "--sampling-temperature",
        str(args.sampling_temperature),
        "--sampling-top-p",
        str(args.sampling_top_p),
        "--sampling-top-k",
        str(args.sampling_top_k),
        "--sampling-min-p",
        str(getattr(args, "sampling_min_p", 0.0)),
        "--trajectory-topk",
        str(args.trajectory_topk),
        "--trajectory-step-interval",
        str(args.trajectory_step_interval),
        "--trajectory-max-points",
        str(args.trajectory_max_points),
        "--cooldown-s",
        str(args.cooldown_s),
        "--effective-batch-cap",
        str(args.effective_batch_cap),
        "--case-filter",
        str(case_filter),
        "--emit-json-summary",
    ]
    if skip_trajectory:
        cmd.append("--skip-trajectory")
    if getattr(args, "sampling_seed", None) is not None:
        cmd.extend(["--sampling-seed", str(args.sampling_seed)])
    if getattr(args, "seed_base", None) is not None:
        cmd.extend(["--seed-base", str(args.seed_base)])
    for prompt in getattr(args, "prompt", []):
        cmd.extend(["--prompt", str(prompt)])
    prompt_file = str(getattr(args, "prompt_file", "") or "")
    if prompt_file:
        cmd.extend(["--prompt-file", prompt_file])
    _append_bool_flag(cmd, bool(args.enforce_eager), "--enforce-eager", "--no-enforce-eager")
    _append_bool_flag(cmd, bool(getattr(args, "enable_distiller_intervention", False)), "--enable-distiller-intervention")
    _append_bool_flag(cmd, bool(args.run_model_bank_case), "--run-model-bank-case", "--no-run-model-bank-case")
    _append_bool_flag(
        cmd,
        bool(args.model_bank_use_output_layernorm),
        "--model-bank-use-output-layernorm",
        "--no-model-bank-use-output-layernorm",
    )
    _append_bool_flag(
        cmd,
        bool(args.model_bank_train_cudagraph),
        "--model-bank-train-cudagraph",
        "--no-model-bank-train-cudagraph",
    )
    _append_bool_flag(
        cmd,
        bool(getattr(args, "model_bank_compact_capture", True)),
        "--model-bank-compact-capture",
        "--no-model-bank-compact-capture",
    )
    if bool(args.benchmark_ignore_eos):
        cmd.append("--benchmark-ignore-eos")
    if bool(args.benchmark_disable_prefix_caching):
        cmd.append("--benchmark-disable-prefix-caching")
    if bool(getattr(args, "sampling_per_request_seed", False)):
        cmd.append("--sampling-per-request-seed")
    return cmd


def _extract_json_summary(stdout: str) -> Dict[str, object]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(JSON_SUMMARY_PREFIX):
            return json.loads(line[len(JSON_SUMMARY_PREFIX) :])
    raise ValueError("JSON summary marker not found in benchmark subprocess output")


def _run_isolated_case_subprocess(
    args: argparse.Namespace,
    *,
    case_filter: str,
    skip_trajectory: bool,
) -> Dict[str, object]:
    env = os.environ.copy()
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    completed = subprocess.run(
        _build_isolated_case_subprocess_cmd(args, case_filter=case_filter, skip_trajectory=skip_trajectory),
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parents[3]),
        env=env,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"isolated benchmark subprocess failed for {case_filter} with exit code {completed.returncode}")
    return _extract_json_summary(completed.stdout or "")


def _run_case(
    *,
    case_id: str,
    llm,
    graph_scratch_rows: int,
    source_layer_path: str,
    target_layer_path: str,
    distiller_hidden_dim: int,
    distiller_lr: float,
    enable_distiller_intervention: bool,
    distiller_beta: float,
    distiller_sampler_backend: str,
    per_request_models: bool,
    per_request_model_bank: bool,
    model_bank_slots: int,
    model_bank_flush_interval: int,
    model_bank_rank: int,
    model_bank_use_output_layernorm: bool,
    model_bank_initializer: SVDModelBankInitializerConfig | None,
    model_bank_train_cudagraph: bool,
    model_bank_forward_backend: str,
    adaptation_pipeline_slots: int,
    adaptation_stream_mode: str,
    adaptation_stream_priority: int,
    compact_capture_lane: bool,
    train_enabled: bool,
    prompts: Sequence[str],
    params: Sequence[SamplingParams],
    request_prompt_indices: Sequence[int] | None,
    request_sample_indices: Sequence[int] | None,
    effective_batch_cap: int,
    warmup_rounds: int,
    rounds: int,
) -> Dict[str, float]:
    esamp_workflow_support.configure_esamp_runtime(
        graph_scratch_rows=int(graph_scratch_rows),
        tap_layer_paths=[source_layer_path, target_layer_path],
        source_layer_path=source_layer_path,
        target_layer_path=target_layer_path,
        enable_esamp_training=bool(train_enabled),
        distiller_hidden_dim=int(distiller_hidden_dim),
        distiller_lr=float(distiller_lr),
        enable_distiller_intervention=bool(enable_distiller_intervention),
        distiller_beta=float(distiller_beta),
        distiller_sampler_backend=str(distiller_sampler_backend),
        per_request_models=bool(per_request_models),
        per_request_model_bank=bool(per_request_model_bank),
        model_bank_slots=int(model_bank_slots),
        model_bank_flush_interval=int(model_bank_flush_interval),
        model_bank_rank=int(model_bank_rank),
        model_bank_use_output_layernorm=bool(model_bank_use_output_layernorm),
        model_bank_initializer=model_bank_initializer,
        model_bank_train_cudagraph=bool(model_bank_train_cudagraph),
        model_bank_forward_backend=str(model_bank_forward_backend),
        adaptation_pipeline_slots=max(1, int(adaptation_pipeline_slots)),
        adaptation_stream_mode=str(adaptation_stream_mode),
        adaptation_stream_priority=int(adaptation_stream_priority),
        compact_capture_lane=bool(compact_capture_lane),
    )
    core.set_esamp_training_enabled(bool(train_enabled))
    core.synchronize_esamp()
    _ = core.read_and_reset_esamp_stats(sync=True)
    _ = core.read_and_reset_esamp_per_request_stats(sync=True)

    prompt_list = list(prompts)
    param_list = list(params)
    cap = max(1, int(effective_batch_cap))

    def _run_generate_chunked():
        outputs_all = []
        for start in range(0, len(prompt_list), cap):
            end = min(len(prompt_list), start + cap)
            pidx_slice = None
            sidx_slice = None
            if request_prompt_indices is not None:
                pidx_slice = list(request_prompt_indices[start:end])
            if request_sample_indices is not None:
                sidx_slice = list(request_sample_indices[start:end])
            outputs = esamp_workflow_support.run_generate_with_request_mapping(
                llm,
                prompt_list[start:end],
                param_list[start:end],
                request_prompt_indices=pidx_slice,
                request_sample_indices=sidx_slice,
            )
            outputs_all.extend(outputs)
        return outputs_all

    for _ in range(int(warmup_rounds)):
        _run_generate_chunked()
    _ = core.read_and_reset_path_hotspot_stats(sync=False)

    torch.cuda.synchronize()
    start = time.perf_counter()
    total_requests = 0
    total_completions = 0
    total_output_tokens = 0

    for _ in range(int(rounds)):
        outputs = _run_generate_chunked()
        total_requests += len(outputs)
        total_completions += _sum_all_completions(outputs)
        total_output_tokens += _sum_all_candidate_tokens(outputs)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    stats = core.read_and_reset_esamp_stats(sync=True)
    distiller_timing = core.read_and_reset_distiller_timing_stats(sync=True)
    path_hotspots = core.read_and_reset_path_hotspot_stats(sync=False)
    model_bank_graph = core.read_graph_debug_stats(mode="model_bank")

    req_per_s = float(total_requests / elapsed) if elapsed > 0 else 0.0
    completion_per_s = float(total_completions / elapsed) if elapsed > 0 else 0.0
    out_tok_per_s = float(total_output_tokens / elapsed) if elapsed > 0 else 0.0
    candidate_avg_count = (
        float(distiller_timing.candidate_token_count / distiller_timing.candidate_sample_count)
        if distiller_timing.candidate_sample_count > 0
        else 0.0
    )
    candidate_avg_per_row = (
        float(distiller_timing.candidate_token_count / distiller_timing.candidate_row_count)
        if distiller_timing.candidate_row_count > 0
        else 0.0
    )

    result = {
        "elapsed_s": float(elapsed),
        "requests": float(total_requests),
        "completions": float(total_completions),
        "output_tokens": float(total_output_tokens),
        "req_per_s": req_per_s,
        "completion_per_s": completion_per_s,
        "out_tok_per_s": out_tok_per_s,
        "loss_avg": float(stats.loss_avg),
        "loss_count": float(stats.loss_count),
        "train_enabled": float(1 if train_enabled else 0),
        "per_request_models": float(1 if per_request_models else 0),
        "per_request_model_bank": float(1 if per_request_model_bank else 0),
        "case_id_hash": float(abs(hash(case_id)) % (10**6)),
        "distiller_precompute_ms_avg": float(distiller_timing.precompute_ms_avg),
        "distiller_precompute_count": float(distiller_timing.precompute_count),
        "distiller_wait_ms_avg": float(distiller_timing.wait_ms_avg),
        "distiller_wait_count": float(distiller_timing.wait_count),
        "distiller_fallback_ms_avg": float(distiller_timing.fallback_ms_avg),
        "distiller_fallback_count": float(distiller_timing.fallback_count),
        "distiller_port_publish_attempt_count": float(distiller_timing.port_publish_attempt_count),
        "distiller_port_publish_hit_count": float(distiller_timing.port_publish_hit_count),
        "distiller_schedule_attempt_count": float(distiller_timing.schedule_attempt_count),
        "distiller_schedule_hit_count": float(distiller_timing.schedule_hit_count),
        "distiller_candidate_sample_count": float(distiller_timing.candidate_sample_count),
        "distiller_candidate_token_count": float(distiller_timing.candidate_token_count),
        "distiller_candidate_row_count": float(distiller_timing.candidate_row_count),
        "distiller_candidate_avg_count": float(candidate_avg_count),
        "distiller_candidate_avg_per_row": float(candidate_avg_per_row),
        "distiller_candidate_max_count": float(distiller_timing.candidate_max_count),
        "distiller_candidate_kernel_triton_count": float(distiller_timing.candidate_kernel_triton_count),
        "distiller_candidate_kernel_torch_count": float(distiller_timing.candidate_kernel_torch_count),
        "distiller_candidate_kernel_fallback_count": float(distiller_timing.candidate_kernel_fallback_count),
        "model_bank_graph_captured": float(1.0 if model_bank_graph.capture_state == "captured" else 0.0),
        "model_bank_graph_capture_attempt_count": float(model_bank_graph.capture_attempt_count),
        "model_bank_graph_skip_not_enabled_count": float(model_bank_graph.skip_not_enabled_count),
        "model_bank_graph_skip_missing_optimizer_state_count": float(model_bank_graph.skip_missing_optimizer_state_count),
        "model_bank_graph_skip_wrong_device_count": float(model_bank_graph.skip_wrong_device_count),
        "model_bank_graph_replay_attempt_count": float(model_bank_graph.replay_attempt_count),
        "model_bank_graph_replay_hit_count": float(model_bank_graph.replay_hit_count),
        "model_bank_graph_replay_stage_miss_count": float(model_bank_graph.replay_stage_miss_count),
        "model_bank_graph_kernel_fallback_count": float(model_bank_graph.kernel_fallback_count),
    }
    result.update(_path_hotspot_fields(path_hotspots))
    return result


def _configure_case_runtime(
    *,
    graph_scratch_rows: int,
    source_layer_path: str,
    target_layer_path: str,
    distiller_hidden_dim: int,
    distiller_lr: float,
    enable_distiller_intervention: bool,
    distiller_beta: float,
    distiller_sampler_backend: str,
    per_request_models: bool,
    per_request_model_bank: bool,
    model_bank_slots: int,
    model_bank_flush_interval: int,
    model_bank_rank: int,
    model_bank_use_output_layernorm: bool,
    model_bank_initializer: SVDModelBankInitializerConfig | None,
    model_bank_train_cudagraph: bool,
    model_bank_forward_backend: str,
    adaptation_pipeline_slots: int,
    adaptation_stream_mode: str,
    adaptation_stream_priority: int,
    compact_capture_lane: bool,
    train_enabled: bool,
) -> None:
    esamp_workflow_support.configure_esamp_runtime(
        graph_scratch_rows=int(graph_scratch_rows),
        tap_layer_paths=[source_layer_path, target_layer_path],
        source_layer_path=source_layer_path,
        target_layer_path=target_layer_path,
        enable_esamp_training=bool(train_enabled),
        distiller_hidden_dim=int(distiller_hidden_dim),
        distiller_lr=float(distiller_lr),
        enable_distiller_intervention=bool(enable_distiller_intervention),
        distiller_beta=float(distiller_beta),
        distiller_sampler_backend=str(distiller_sampler_backend),
        per_request_models=bool(per_request_models),
        per_request_model_bank=bool(per_request_model_bank),
        model_bank_slots=int(model_bank_slots),
        model_bank_flush_interval=int(model_bank_flush_interval),
        model_bank_rank=int(model_bank_rank),
        model_bank_use_output_layernorm=bool(model_bank_use_output_layernorm),
        model_bank_initializer=model_bank_initializer,
        model_bank_train_cudagraph=bool(model_bank_train_cudagraph),
        model_bank_forward_backend=str(model_bank_forward_backend),
        adaptation_pipeline_slots=max(1, int(adaptation_pipeline_slots)),
        adaptation_stream_mode=str(adaptation_stream_mode),
        adaptation_stream_priority=int(adaptation_stream_priority),
        compact_capture_lane=bool(compact_capture_lane),
    )


def _run_per_request_trajectory(
    *,
    args: argparse.Namespace,
    llm,
    prompts: Sequence[str],
    params: Sequence[SamplingParams],
    request_prompt_indices: Sequence[int] | None,
    request_sample_indices: Sequence[int] | None,
    effective_batch_cap: int,
    rows: int,
) -> None:
    esamp_workflow_support.configure_esamp_runtime(
        graph_scratch_rows=int(rows),
        tap_layer_paths=[args.source_layer_path, args.target_layer_path],
        source_layer_path=args.source_layer_path,
        target_layer_path=args.target_layer_path,
        enable_esamp_training=True,
        distiller_hidden_dim=int(args.distiller_hidden_dim),
        distiller_lr=float(args.distiller_lr),
        per_request_models=True,
        per_request_model_bank=False,
        model_bank_slots=0,
        model_bank_flush_interval=1,
        model_bank_rank=int(args.model_bank_rank),
        model_bank_use_output_layernorm=bool(args.model_bank_use_output_layernorm),
        model_bank_initializer=_build_model_bank_initializer_config(args),
        model_bank_train_cudagraph=bool(args.model_bank_train_cudagraph),
        model_bank_forward_backend=str(getattr(args, "model_bank_forward_backend", "torch")),
        adaptation_pipeline_slots=_adaptation_pipeline_slots(args),
        adaptation_stream_mode=str(getattr(args, "adaptation_stream_mode", "dual")),
        adaptation_stream_priority=int(getattr(args, "adaptation_stream_priority", 0)),
        trace_per_request_losses=True,
        trace_interval=max(1, int(args.trajectory_step_interval)),
        trace_max_points=max(0, int(args.trajectory_max_points)),
    )
    history: Dict[int, List[float]] = {}
    counts: Dict[int, int] = {}
    core.set_esamp_training_enabled(True)
    _ = core.read_and_reset_esamp_per_request_stats(sync=True)
    prompt_list = list(prompts)
    param_list = list(params)
    cap = max(1, int(effective_batch_cap))

    def _run_generate_chunked() -> None:
        for start in range(0, len(prompt_list), cap):
            end = min(len(prompt_list), start + cap)
            pidx_slice = None
            sidx_slice = None
            if request_prompt_indices is not None:
                pidx_slice = list(request_prompt_indices[start:end])
            if request_sample_indices is not None:
                sidx_slice = list(request_sample_indices[start:end])
            esamp_workflow_support.run_generate_with_request_mapping(
                llm,
                prompt_list[start:end],
                param_list[start:end],
                request_prompt_indices=pidx_slice,
                request_sample_indices=sidx_slice,
            )

    for _ in range(int(args.benchmark_warmup_rounds)):
        _run_generate_chunked()

    round_summaries: List[Dict[str, float]] = []
    for round_i in range(int(args.benchmark_rounds)):
        _run_generate_chunked()
        per_stats = core.read_and_reset_esamp_per_request_stats(sync=True)
        active = len(per_stats)
        total_cnt = sum(s.loss_count for s in per_stats.values())
        mean_loss = (
            sum((s.loss_avg * max(1, s.loss_count)) for s in per_stats.values()) / max(1, total_cnt)
            if per_stats
            else 0.0
        )
        print(
            f"[trajectory] round={round_i + 1} active_models={active} "
            f"loss_updates={total_cnt} weighted_loss={mean_loss:.6f}"
        )
        round_summaries.append(
            {
                "round": float(round_i + 1),
                "active_models": float(active),
                "loss_updates": float(total_cnt),
                "weighted_loss": float(mean_loss),
            }
        )
        for pidx, s in per_stats.items():
            if s.loss_count <= 0:
                continue
            trace = list(s.trace_losses) or [float(s.loss_avg)]
            history.setdefault(int(pidx), []).extend(trace)
            counts[int(pidx)] = counts.get(int(pidx), 0) + int(s.loss_count)

    topk = max(1, int(args.trajectory_topk))
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:topk]
    print(f"[trajectory] top_{topk}_models_by_update_count")
    top_summaries: List[Dict[str, float]] = []
    for pidx, cnt in ranked:
        losses = history.get(int(pidx), [])
        if not losses:
            continue
        print(
            f"  prompt_idx={pidx} updates={cnt} first={losses[0]:.6f} "
            f"last={losses[-1]:.6f} min={min(losses):.6f} max={max(losses):.6f}"
        )
        top_summaries.append(
            {
                "prompt_idx": float(pidx),
                "updates": float(cnt),
                "first": float(losses[0]),
                "last": float(losses[-1]),
                "min": float(min(losses)),
                "max": float(max(losses)),
            }
        )
    return {
        "rounds": round_summaries,
        "top_models": top_summaries,
    }


def _run_one_implementation(args: argparse.Namespace, implementation: str) -> Dict[str, object]:
    if int(args.sampling_n) <= 1:
        raise RuntimeError("--sampling-n must be > 1 for this experiment")

    prompts = read_prompts(args.prompt_file, args.prompt)
    bench_prompts = build_prompt_batch(prompts, int(args.benchmark_batch_size))
    # vLLM V1 runtime currently emits one completion per request even when
    # SamplingParams.n > 1. Expand requests explicitly so n>1 benchmark
    # semantics (and token accounting) remain correct.
    effective_prompts, request_prompt_indices, request_sample_indices = _expand_requests_for_effective_n(
        bench_prompts, int(args.sampling_n)
    )
    effective_batch_cap = _resolve_effective_batch_cap(len(effective_prompts), int(args.effective_batch_cap))
    effective_sampling_n = 1
    print(
        "[compat] vLLM V1 n>1 is emulated via prompt expansion: "
        f"base_batch={len(bench_prompts)} sampling_n={int(args.sampling_n)} "
        f"effective_batch={len(effective_prompts)} chunk_cap={effective_batch_cap}"
    )

    rows = _resolve_graph_scratch_rows(
        int(args.graph_scratch_rows),
        effective_batch_cap=effective_batch_cap,
    )

    sampling_seed = args.sampling_seed
    if sampling_seed is None and args.seed_base is not None:
        sampling_seed = int(args.seed_base)

    params = _build_sampling_params(
        prompts=effective_prompts,
        max_new_tokens=int(args.benchmark_max_new_tokens),
        sampling_n=int(effective_sampling_n),
        sampling_temperature=float(args.sampling_temperature),
        sampling_top_p=float(args.sampling_top_p),
        sampling_top_k=int(args.sampling_top_k),
        sampling_min_p=float(getattr(args, "sampling_min_p", 0.0)),
        ignore_eos=bool(args.benchmark_ignore_eos),
        sampling_seed=(None if sampling_seed is None else int(sampling_seed)),
        sampling_per_request_seed=bool(args.sampling_per_request_seed),
    )

    common = dict(
        graph_scratch_rows=int(rows),
        source_layer_path=args.source_layer_path,
        target_layer_path=args.target_layer_path,
        distiller_hidden_dim=int(args.distiller_hidden_dim),
        distiller_lr=float(args.distiller_lr),
        enable_distiller_intervention=bool(args.enable_distiller_intervention),
        distiller_beta=float(args.distiller_beta),
        distiller_sampler_backend=str(args.distiller_sampler_backend),
        model_bank_slots=_resolve_model_bank_slots(
            int(args.model_bank_slots),
            effective_batch_cap=effective_batch_cap,
            prompt_count=len(bench_prompts),
        ),
        model_bank_flush_interval=int(args.model_bank_flush_interval),
        model_bank_rank=int(args.model_bank_rank),
        model_bank_use_output_layernorm=bool(args.model_bank_use_output_layernorm),
        model_bank_initializer=_build_model_bank_initializer_config(args),
        model_bank_train_cudagraph=bool(args.model_bank_train_cudagraph),
        model_bank_forward_backend=normalize_model_bank_forward_backend(getattr(args, "model_bank_forward_backend", "torch")),
        adaptation_pipeline_slots=_adaptation_pipeline_slots(args),
        adaptation_stream_mode=str(getattr(args, "adaptation_stream_mode", "dual")),
        adaptation_stream_priority=int(getattr(args, "adaptation_stream_priority", 0)),
        prompts=effective_prompts,
        params=params,
        request_prompt_indices=request_prompt_indices,
        request_sample_indices=request_sample_indices,
        effective_batch_cap=effective_batch_cap,
        warmup_rounds=int(args.benchmark_warmup_rounds),
        rounds=int(args.benchmark_rounds),
    )
    requested_case_filter = str(getattr(args, "case_filter", "all") or "all").strip()
    if requested_case_filter == "all":
        selected_case_names = ["single_off", "single_on", "per_request_on"]
        if bool(args.run_model_bank_case):
            selected_case_names.append("model_bank_on")
    else:
        selected_case_names = [requested_case_filter]
    model_bank_case_selected = "model_bank_on" in selected_case_names
    common["compact_capture_lane"] = bool(
        getattr(args, "model_bank_compact_capture", True)
        and model_bank_case_selected
        and (not bool(args.enable_distiller_intervention))
    )

    def _make_case_llm():
        return core.make_llm(
            model_name=args.model_name,
            dtype=args.dtype,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            max_model_len=int(args.max_model_len),
            enable_prefix_caching=(not args.benchmark_disable_prefix_caching),
            enforce_eager=bool(args.enforce_eager),
        )

    isolated_cases = _requires_isolated_llm_cases(
        enable_distiller_intervention=bool(args.enable_distiller_intervention),
        distiller_beta=float(args.distiller_beta),
    )
    isolated_cooldown_s = max(float(args.cooldown_s), 3.0) if isolated_cases else float(args.cooldown_s)

    if isolated_cases and requested_case_filter == "all":
        cases: Dict[str, Dict[str, float]] = {}
        for case_name in selected_case_names:
            payload = _run_isolated_case_subprocess(
                args,
                case_filter=case_name,
                skip_trajectory=True,
            )
            cases.update(cast(Dict[str, Dict[str, float]], payload["cases"]))
        trajectory = {"rounds": [], "top_models": []}
        return {
            "implementation": str(implementation),
            "cases": cases,
            "trajectory": trajectory,
        }

    if not isolated_cases:
        # Seed the runtime with the superset of later-active hook requirements
        # before constructing the LLM, so load_model-time hook/cudagraph setup is
        # compatible with the later training cases in this same implementation run.
        esamp_workflow_support.configure_esamp_runtime(
            graph_scratch_rows=int(rows),
            tap_layer_paths=[args.source_layer_path, args.target_layer_path],
            source_layer_path=args.source_layer_path,
            target_layer_path=args.target_layer_path,
            enable_esamp_training=True,
            distiller_hidden_dim=int(args.distiller_hidden_dim),
            distiller_lr=float(args.distiller_lr),
            enable_distiller_intervention=bool(args.enable_distiller_intervention),
            distiller_beta=float(args.distiller_beta),
            distiller_sampler_backend=str(args.distiller_sampler_backend),
            per_request_models=True,
            per_request_model_bank=bool(args.run_model_bank_case),
            model_bank_slots=common["model_bank_slots"],
            model_bank_flush_interval=int(args.model_bank_flush_interval),
            model_bank_rank=int(args.model_bank_rank),
            model_bank_use_output_layernorm=bool(args.model_bank_use_output_layernorm),
            model_bank_initializer=_build_model_bank_initializer_config(args),
            model_bank_train_cudagraph=bool(args.model_bank_train_cudagraph),
            model_bank_forward_backend=str(getattr(args, "model_bank_forward_backend", "torch")),
            adaptation_pipeline_slots=_adaptation_pipeline_slots(args),
            adaptation_stream_mode=str(getattr(args, "adaptation_stream_mode", "dual")),
            adaptation_stream_priority=int(getattr(args, "adaptation_stream_priority", 0)),
            compact_capture_lane=bool(common["compact_capture_lane"]),
        )
        llm = _make_case_llm()
    else:
        llm = None

    try:
        cases: Dict[str, Dict[str, float]] = {}
        def _run_case_maybe_isolated(
            *,
            case_id: str,
            per_request_models: bool,
            per_request_model_bank: bool,
            train_enabled: bool,
        ) -> Dict[str, float]:
            if not isolated_cases:
                assert llm is not None
                return _run_case(
                    case_id=case_id,
                    llm=llm,
                    per_request_models=per_request_models,
                    per_request_model_bank=per_request_model_bank,
                    train_enabled=train_enabled,
                    **common,
                )
            _configure_case_runtime(
                graph_scratch_rows=common["graph_scratch_rows"],
                source_layer_path=common["source_layer_path"],
                target_layer_path=common["target_layer_path"],
                distiller_hidden_dim=common["distiller_hidden_dim"],
                distiller_lr=common["distiller_lr"],
                enable_distiller_intervention=common["enable_distiller_intervention"],
                distiller_beta=common["distiller_beta"],
                distiller_sampler_backend=common["distiller_sampler_backend"],
                per_request_models=per_request_models,
                per_request_model_bank=per_request_model_bank,
                model_bank_slots=common["model_bank_slots"],
                model_bank_flush_interval=common["model_bank_flush_interval"],
                model_bank_rank=common["model_bank_rank"],
                model_bank_use_output_layernorm=common["model_bank_use_output_layernorm"],
                model_bank_initializer=common["model_bank_initializer"],
                model_bank_train_cudagraph=common["model_bank_train_cudagraph"],
                model_bank_forward_backend=common["model_bank_forward_backend"],
                adaptation_pipeline_slots=common["adaptation_pipeline_slots"],
                adaptation_stream_mode=common["adaptation_stream_mode"],
                adaptation_stream_priority=common["adaptation_stream_priority"],
                compact_capture_lane=common["compact_capture_lane"],
                train_enabled=train_enabled,
            )
            case_llm = _make_case_llm()
            try:
                return _run_case(
                    case_id=case_id,
                    llm=case_llm,
                    per_request_models=per_request_models,
                    per_request_model_bank=per_request_model_bank,
                    train_enabled=train_enabled,
                    **common,
                )
            finally:
                shutdown_llm_instance(case_llm, cooldown_s=isolated_cooldown_s)

        if "single_off" in selected_case_names:
            single_off = _run_case_maybe_isolated(
                case_id="single_off",
                per_request_models=False,
                per_request_model_bank=False,
                train_enabled=False,
            )
            print(
                "single_off: "
                f"req/s={single_off['req_per_s']:.3f} comp/s={single_off['completion_per_s']:.3f} "
                f"out_tok/s={single_off['out_tok_per_s']:.3f}"
            )
            _maybe_print_distiller_timing("single_off", single_off)
            cases["single_off"] = single_off

        if "single_on" in selected_case_names:
            single_on = _run_case_maybe_isolated(
                case_id="single_on",
                per_request_models=False,
                per_request_model_bank=False,
                train_enabled=True,
            )
            print(
                "single_on: "
                f"req/s={single_on['req_per_s']:.3f} comp/s={single_on['completion_per_s']:.3f} "
                f"out_tok/s={single_on['out_tok_per_s']:.3f} "
                f"loss_avg={single_on['loss_avg']:.6f} loss_count={int(single_on['loss_count'])}"
            )
            _maybe_print_distiller_timing("single_on", single_on)
            cases["single_on"] = single_on

        if "per_request_on" in selected_case_names:
            per_req_on = _run_case_maybe_isolated(
                case_id="per_request_on",
                per_request_models=True,
                per_request_model_bank=False,
                train_enabled=True,
            )
            print(
                "per_request_on: "
                f"req/s={per_req_on['req_per_s']:.3f} comp/s={per_req_on['completion_per_s']:.3f} "
                f"out_tok/s={per_req_on['out_tok_per_s']:.3f} "
                f"loss_avg={per_req_on['loss_avg']:.6f} loss_count={int(per_req_on['loss_count'])}"
            )
            _maybe_print_distiller_timing("per_request_on", per_req_on)
            cases["per_request_on"] = per_req_on

        if "single_off" in cases:
            base_req = max(1e-12, float(cases["single_off"]["req_per_s"]))
            base_comp = max(1e-12, float(cases["single_off"]["completion_per_s"]))
            base_tok = max(1e-12, float(cases["single_off"]["out_tok_per_s"]))
            ratio_parts: list[str] = []
            if "single_on" in cases:
                ratio_parts.extend(
                    [
                        f"single_on(req/s)={cases['single_on']['req_per_s']/base_req:.4f}",
                        f"single_on(comp/s)={cases['single_on']['completion_per_s']/base_comp:.4f}",
                        f"single_on(tok/s)={cases['single_on']['out_tok_per_s']/base_tok:.4f}",
                    ]
                )
            if "per_request_on" in cases:
                ratio_parts.extend(
                    [
                        f"per_request_on(req/s)={cases['per_request_on']['req_per_s']/base_req:.4f}",
                        f"per_request_on(comp/s)={cases['per_request_on']['completion_per_s']/base_comp:.4f}",
                        f"per_request_on(tok/s)={cases['per_request_on']['out_tok_per_s']/base_tok:.4f}",
                    ]
                )
            if ratio_parts:
                print("relative_vs_single_off: " + " ".join(ratio_parts))

        if "model_bank_on" in selected_case_names:
            bank_on = _run_case_maybe_isolated(
                case_id="model_bank_on",
                per_request_models=True,
                per_request_model_bank=True,
                train_enabled=True,
            )
            print(
                "model_bank_on: "
                f"req/s={bank_on['req_per_s']:.3f} comp/s={bank_on['completion_per_s']:.3f} "
                f"out_tok/s={bank_on['out_tok_per_s']:.3f} "
                f"loss_avg={bank_on['loss_avg']:.6f} loss_count={int(bank_on['loss_count'])}"
            )
            _maybe_print_distiller_timing("model_bank_on", bank_on)
            if "single_off" in cases:
                base_req = max(1e-12, float(cases["single_off"]["req_per_s"]))
                base_comp = max(1e-12, float(cases["single_off"]["completion_per_s"]))
                base_tok = max(1e-12, float(cases["single_off"]["out_tok_per_s"]))
                print(
                    "relative_vs_single_off(model_bank): "
                    f"model_bank_on(req/s)={bank_on['req_per_s']/base_req:.4f} "
                    f"model_bank_on(comp/s)={bank_on['completion_per_s']/base_comp:.4f} "
                    f"model_bank_on(tok/s)={bank_on['out_tok_per_s']/base_tok:.4f}"
                )
            cases["model_bank_on"] = bank_on

        if bool(getattr(args, "skip_trajectory", False)) or requested_case_filter != "all":
            trajectory = {"rounds": [], "top_models": []}
        elif not isolated_cases:
            assert llm is not None
            trajectory = _run_per_request_trajectory(
                args=args,
                llm=llm,
                prompts=effective_prompts,
                params=params,
                request_prompt_indices=request_prompt_indices,
                request_sample_indices=request_sample_indices,
                effective_batch_cap=effective_batch_cap,
                rows=int(rows),
            )
        else:
            _configure_case_runtime(
                graph_scratch_rows=common["graph_scratch_rows"],
                source_layer_path=common["source_layer_path"],
                target_layer_path=common["target_layer_path"],
                distiller_hidden_dim=common["distiller_hidden_dim"],
                distiller_lr=common["distiller_lr"],
                enable_distiller_intervention=common["enable_distiller_intervention"],
                distiller_beta=common["distiller_beta"],
                distiller_sampler_backend=common["distiller_sampler_backend"],
                per_request_models=True,
                per_request_model_bank=False,
                model_bank_slots=common["model_bank_slots"],
                model_bank_flush_interval=common["model_bank_flush_interval"],
                model_bank_rank=common["model_bank_rank"],
                model_bank_use_output_layernorm=common["model_bank_use_output_layernorm"],
                model_bank_initializer=common["model_bank_initializer"],
                model_bank_train_cudagraph=common["model_bank_train_cudagraph"],
                model_bank_forward_backend=common["model_bank_forward_backend"],
                adaptation_pipeline_slots=common["adaptation_pipeline_slots"],
                adaptation_stream_mode=common["adaptation_stream_mode"],
                adaptation_stream_priority=common["adaptation_stream_priority"],
                compact_capture_lane=common["compact_capture_lane"],
                train_enabled=True,
            )
            trajectory_llm = _make_case_llm()
            try:
                trajectory = _run_per_request_trajectory(
                    args=args,
                    llm=trajectory_llm,
                    prompts=effective_prompts,
                    params=params,
                    request_prompt_indices=request_prompt_indices,
                    request_sample_indices=request_sample_indices,
                    effective_batch_cap=effective_batch_cap,
                    rows=int(rows),
                )
            finally:
                shutdown_llm_instance(trajectory_llm, cooldown_s=isolated_cooldown_s)
    finally:
        if llm is not None:
            shutdown_llm_instance(llm, cooldown_s=float(args.cooldown_s))

    return {
        "implementation": str(implementation),
        "cases": cases,
        "trajectory": trajectory,
    }


def main() -> int:
    args = _parse_args()
    payload = _run_one_implementation(args, "esamp")
    if args.emit_json_summary:
        print(JSON_SUMMARY_PREFIX + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
