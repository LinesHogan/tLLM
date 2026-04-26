#!/usr/bin/env python3
"""Profile ESamp adaptation modes and write a markdown report with evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch.profiler import ProfilerActivity, profile
from vllm import SamplingParams

from tllm.consumers.esamp.initializers.svd import SVDModelBankInitializerConfig
from tllm.workflows.common import (
    build_sampling_params as _build_sampling_params,
    sum_all_candidate_tokens as _sum_all_candidate_tokens,
    sum_all_completions as _sum_all_completions,
)
from tllm.runtime import residual_runtime as core
from tllm.runtime.sampler_bridge.types import SAMPLER_BACKEND_CHOICES
from tllm.workflows import esamp_support as esamp_workflow_support
from tllm.util.tools import shutdown_llm_instance
from tllm.util.tools import build_prompt_batch, read_prompts


@dataclass
class ModeSpec:
    name: str
    train_enabled: bool
    per_request_models: bool
    per_request_model_bank: bool


@dataclass
class ModeResult:
    name: str
    req_per_s: float
    comp_per_s: float
    out_tok_per_s: float
    loss_avg: float
    loss_count: int
    elapsed_s: float
    peak_mem_gb: float
    total_output_tokens: int
    total_requests: int
    total_completions: int
    trace_path: str
    top_ops: List[Dict[str, float]]
    esamp_cuda_ms: float
    bank_cuda_ms: float
    esamp_cpu_ms: float
    bank_cpu_ms: float
    top_metric: str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--prompt", action="append", default=[])
    p.add_argument("--prompt-file", type=str, default="")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    p.add_argument("--graph-scratch-rows", type=int, default=0)
    p.add_argument("--source-layer-path", type=str, default="model.model.layers[0].input_layernorm")
    p.add_argument("--target-layer-path", type=str, default="model.model.layers[-1].input_layernorm")
    p.add_argument("--distiller-hidden-dim", type=int, default=256)
    p.add_argument("--distiller-lr", type=float, default=1e-3)
    p.add_argument("--enable-distiller-intervention", action="store_true")
    p.add_argument("--distiller-beta", type=float, default=0.0)
    p.add_argument(
        "--distiller-sampler-backend",
        type=str,
        default="post_filter_exact",
        choices=SAMPLER_BACKEND_CHOICES,
    )
    p.add_argument("--model-bank-slots", type=int, default=0)
    p.add_argument("--model-bank-flush-interval", type=int, default=1)
    p.add_argument("--model-bank-rank", type=int, default=64)
    p.add_argument("--adaptation-pipeline-slots", type=int, default=4)
    p.add_argument("--adaptation-stream-mode", type=str, default="dual", choices=["dual", "single", "serial"])
    p.add_argument("--adaptation-stream-priority", type=int, default=0)
    p.add_argument("--model-bank-use-output-layernorm", action="store_true")
    p.add_argument("--no-model-bank-use-output-layernorm", dest="model_bank_use_output_layernorm", action="store_false")
    p.add_argument("--model-bank-initializer", type=str, default="none", choices=["none", "svd"])
    p.add_argument(
        "--model-bank-initializer-svd-method",
        type=str,
        default="ffn_fast_svd",
        choices=["ridge_svd", "ridge-svd", "ridge+svd", "ffn_fast_svd", "ffn-fast-svd", "ffn+svd"],
    )
    p.add_argument("--model-bank-initializer-svd-ridge-lambda", type=float, default=1e-2)
    p.add_argument("--model-bank-initializer-svd-min-rows", type=int, default=32)
    p.add_argument("--model-bank-initializer-svd-max-wait-steps", type=int, default=4)
    p.add_argument("--model-bank-train-cudagraph", action="store_true")
    p.add_argument(
        "--no-model-bank-train-cudagraph",
        dest="model_bank_train_cudagraph",
        action="store_false",
    )
    p.add_argument("--benchmark-batch-size", type=int, default=4)
    p.add_argument("--benchmark-max-new-tokens", type=int, default=128)
    p.add_argument("--sampling-n", type=int, default=16)
    p.add_argument("--sampling-temperature", type=float, default=0.8)
    p.add_argument("--sampling-top-p", type=float, default=0.95)
    p.add_argument("--sampling-top-k", type=int, default=-1)
    p.add_argument("--sampling-min-p", type=float, default=0.0)
    p.add_argument("--sampling-seed", type=int, default=None)
    p.add_argument("--sampling-per-request-seed", action="store_true")
    p.add_argument("--seed-base", type=int, default=None)
    p.add_argument("--warmup-rounds", type=int, default=1)
    p.add_argument("--profile-rounds", type=int, default=1)
    p.add_argument("--top-k-ops", type=int, default=20)
    p.add_argument(
        "--modes",
        type=str,
        default="single_off,single_on,per_request_on,model_bank_on",
        help="Comma separated subset of: single_off,single_on,per_request_on,model_bank_on",
    )
    p.add_argument(
        "--trace-dir",
        type=str,
        default="tLLM/outputs/profile_traces",
    )
    p.add_argument(
        "--report-md",
        type=str,
        default="tLLM/doc/PROFILE_esamp_model_bank.md",
    )
    p.add_argument(
        "--raw-json",
        type=str,
        default="tLLM/outputs/profile_traces/profile_esamp_raw.json",
    )
    p.add_argument("--cooldown-s", type=float, default=1.0)
    p.set_defaults(enforce_eager=False)
    p.set_defaults(model_bank_use_output_layernorm=True)
    p.set_defaults(model_bank_train_cudagraph=False)
    return p.parse_args()


def _build_model_bank_initializer_config(args: argparse.Namespace) -> SVDModelBankInitializerConfig | None:
    if str(args.model_bank_initializer).strip().lower() != "svd":
        return None
    return SVDModelBankInitializerConfig(
        method=str(args.model_bank_initializer_svd_method).strip().lower(),
        ridge_lambda=float(args.model_bank_initializer_svd_ridge_lambda),
        min_rows=int(args.model_bank_initializer_svd_min_rows),
        max_wait_steps=int(args.model_bank_initializer_svd_max_wait_steps),
    )

def _to_mode_spec(name: str) -> ModeSpec:
    key = name.strip()
    if key == "single_off":
        return ModeSpec(name=key, train_enabled=False, per_request_models=False, per_request_model_bank=False)
    if key == "single_on":
        return ModeSpec(name=key, train_enabled=True, per_request_models=False, per_request_model_bank=False)
    if key == "per_request_on":
        return ModeSpec(name=key, train_enabled=True, per_request_models=True, per_request_model_bank=False)
    if key == "model_bank_on":
        return ModeSpec(name=key, train_enabled=True, per_request_models=True, per_request_model_bank=True)
    raise RuntimeError(f"unknown mode: {name}")


def _collect_all_op_rows(prof) -> List[Dict[str, float]]:
    rows = []
    for e in prof.key_averages():
        rows.append(
            {
                "key": str(e.key),
                "self_cuda_ms": float(getattr(e, "self_cuda_time_total", 0.0) / 1000.0),
                "cuda_total_ms": float(getattr(e, "cuda_time_total", 0.0) / 1000.0),
                "self_cpu_ms": float(getattr(e, "self_cpu_time_total", 0.0) / 1000.0),
                "calls": int(getattr(e, "count", 0)),
            }
        )
    return rows


def _pick_top_metric(rows: List[Dict[str, float]]) -> str:
    max_cuda = max((float(r["self_cuda_ms"]) for r in rows), default=0.0)
    return "self_cuda_ms" if max_cuda > 0.0 else "self_cpu_ms"


def _collect_top_ops(rows: List[Dict[str, float]], top_k: int, metric: str) -> List[Dict[str, float]]:
    return sorted(rows, key=lambda r: float(r.get(metric, 0.0)), reverse=True)[: max(1, int(top_k))]


def _aggregate_esamp_time(rows: List[Dict[str, float]], metric: str) -> tuple[float, float]:
    side_total = 0.0
    bank_total = 0.0
    for op in rows:
        key = op["key"]
        val = float(op.get(metric, 0.0))
        if key.startswith("esamp."):
            side_total += val
        if key.startswith("esamp.bank"):
            bank_total += val
    return side_total, bank_total


def _profile_one_mode(
    *,
    mode: ModeSpec,
    llm,
    prompts: List[str],
    params: List[SamplingParams],
    args: argparse.Namespace,
    rows: int,
    trace_dir: str,
) -> ModeResult:
    esamp_workflow_support.configure_esamp_runtime(
        graph_scratch_rows=int(rows),
        tap_layer_paths=[args.source_layer_path, args.target_layer_path],
        source_layer_path=args.source_layer_path,
        target_layer_path=args.target_layer_path,
        enable_esamp_training=bool(mode.train_enabled),
        distiller_hidden_dim=int(args.distiller_hidden_dim),
        distiller_lr=float(args.distiller_lr),
        enable_distiller_intervention=bool(args.enable_distiller_intervention),
        distiller_beta=float(args.distiller_beta),
        distiller_sampler_backend=str(args.distiller_sampler_backend),
        per_request_models=bool(mode.per_request_models),
        per_request_model_bank=bool(mode.per_request_model_bank),
        model_bank_slots=(int(args.model_bank_slots) if int(args.model_bank_slots) > 0 else len(prompts)),
        model_bank_flush_interval=int(args.model_bank_flush_interval),
        model_bank_rank=int(args.model_bank_rank),
        model_bank_use_output_layernorm=bool(args.model_bank_use_output_layernorm),
        model_bank_initializer=_build_model_bank_initializer_config(args),
        model_bank_train_cudagraph=bool(args.model_bank_train_cudagraph),
        adaptation_pipeline_slots=max(1, int(args.adaptation_pipeline_slots)),
        adaptation_stream_mode=str(args.adaptation_stream_mode),
        adaptation_stream_priority=int(args.adaptation_stream_priority),
    )
    core.set_esamp_training_enabled(bool(mode.train_enabled))
    core.synchronize_esamp()
    _ = core.read_and_reset_esamp_stats(sync=True)
    _ = core.read_and_reset_esamp_per_request_stats(sync=True)

    for _ in range(int(args.warmup_rounds)):
        esamp_workflow_support.run_generate_with_request_mapping(llm, list(prompts), list(params))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    total_requests = 0
    total_completions = 0
    total_output_tokens = 0
    trace_path = os.path.join(trace_dir, f"{mode.name}.json")
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        start = time.perf_counter()
        for _ in range(int(args.profile_rounds)):
            outputs = esamp_workflow_support.run_generate_with_request_mapping(llm, list(prompts), list(params))
            total_requests += len(outputs)
            total_completions += _sum_all_completions(outputs)
            total_output_tokens += _sum_all_candidate_tokens(outputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    prof.export_chrome_trace(trace_path)

    stats = core.read_and_reset_esamp_stats(sync=True)
    all_rows = _collect_all_op_rows(prof)
    top_metric = _pick_top_metric(all_rows)
    top_ops = _collect_top_ops(all_rows, int(args.top_k_ops), top_metric)
    esamp_cuda_ms, bank_cuda_ms = _aggregate_esamp_time(all_rows, "self_cuda_ms")
    esamp_cpu_ms, bank_cpu_ms = _aggregate_esamp_time(all_rows, "self_cpu_ms")

    peak_mem_gb = 0.0
    if torch.cuda.is_available():
        peak_mem_gb = float(torch.cuda.max_memory_allocated() / (1024**3))

    req_per_s = float(total_requests / elapsed) if elapsed > 0 else 0.0
    comp_per_s = float(total_completions / elapsed) if elapsed > 0 else 0.0
    out_tok_per_s = float(total_output_tokens / elapsed) if elapsed > 0 else 0.0
    return ModeResult(
        name=mode.name,
        req_per_s=req_per_s,
        comp_per_s=comp_per_s,
        out_tok_per_s=out_tok_per_s,
        loss_avg=float(stats.loss_avg),
        loss_count=int(stats.loss_count),
        elapsed_s=float(elapsed),
        peak_mem_gb=peak_mem_gb,
        total_output_tokens=int(total_output_tokens),
        total_requests=int(total_requests),
        total_completions=int(total_completions),
        trace_path=trace_path,
        top_ops=top_ops,
        esamp_cuda_ms=esamp_cuda_ms,
        bank_cuda_ms=bank_cuda_ms,
        esamp_cpu_ms=esamp_cpu_ms,
        bank_cpu_ms=bank_cpu_ms,
        top_metric=top_metric,
    )


def _write_report(
    *,
    args: argparse.Namespace,
    results: List[ModeResult],
    report_md: str,
) -> None:
    ts = dt.datetime.now().isoformat(timespec="seconds")
    by_name = {r.name: r for r in results}
    base = by_name.get("single_off")
    lines: List[str] = []
    lines.append("# Side Train / Model Bank Profiling Report")
    lines.append("")
    lines.append(f"- Generated at: `{ts}`")
    lines.append(f"- Model: `{args.model_name}`")
    lines.append(
        f"- Config: `dtype={args.dtype}, batch={args.benchmark_batch_size}, n={args.sampling_n}, "
        f"max_new_tokens={args.benchmark_max_new_tokens}, rounds={args.profile_rounds}, "
        f"bank_slots={args.model_bank_slots if int(args.model_bank_slots) > 0 else 'auto'}, "
        f"bank_flush={args.model_bank_flush_interval}, "
        f"distiller={bool(args.enable_distiller_intervention)}, "
        f"distiller_beta={float(args.distiller_beta)}, "
        f"distiller_backend={args.distiller_sampler_backend}`"
    )
    lines.append("")
    lines.append("## Throughput & Memory")
    lines.append("")
    lines.append("| mode | req/s | comp/s | tok/s | vs single_off tok/s | peak_mem(GB) | loss_avg | loss_count |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        ratio = (r.out_tok_per_s / base.out_tok_per_s) if base and base.out_tok_per_s > 0 else 0.0
        lines.append(
            f"| `{r.name}` | {r.req_per_s:.3f} | {r.comp_per_s:.3f} | {r.out_tok_per_s:.3f} | "
            f"{ratio:.4f} | {r.peak_mem_gb:.3f} | {r.loss_avg:.6f} | {r.loss_count} |"
        )
    lines.append("")
    lines.append("## Side-Train Time Attribution (Profiler Self CUDA Time)")
    lines.append("")
    lines.append("| mode | esamp total ms | bank-only ms |")
    lines.append("| --- | ---: | ---: |")
    for r in results:
        lines.append(f"| `{r.name}` | {r.esamp_cuda_ms:.3f} | {r.bank_cuda_ms:.3f} |")
    lines.append("")
    lines.append("## Side-Train Time Attribution (Profiler Self CPU Time)")
    lines.append("")
    lines.append("| mode | esamp total ms | bank-only ms |")
    lines.append("| --- | ---: | ---: |")
    for r in results:
        lines.append(f"| `{r.name}` | {r.esamp_cpu_ms:.3f} | {r.bank_cpu_ms:.3f} |")
    lines.append("")
    lines.append("## Top Ops Per Mode")
    lines.append("")
    for r in results:
        lines.append(f"### {r.name}")
        lines.append("")
        lines.append(f"- Chrome trace: `{r.trace_path}`")
        lines.append(f"- Top list sort metric: `{r.top_metric}`")
        lines.append("")
        lines.append("| op | self_cpu_ms | self_cuda_ms | cuda_total_ms | calls |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for op in r.top_ops[:10]:
            lines.append(
                f"| `{op['key']}` | {op['self_cpu_ms']:.3f} | {op['self_cuda_ms']:.3f} | "
                f"{op['cuda_total_ms']:.3f} | {int(op['calls'])} |"
            )
        lines.append("")

    lines.append("## Preliminary Findings")
    lines.append("")
    if "model_bank_on" in by_name and "per_request_on" in by_name:
        bank = by_name["model_bank_on"]
        per = by_name["per_request_on"]
        lines.append(
            f"- `model_bank_on` tok/s is `{bank.out_tok_per_s:.3f}`, compared with "
            f"`per_request_on` `{per.out_tok_per_s:.3f}` (ratio `{(bank.out_tok_per_s / max(1e-12, per.out_tok_per_s)):.4f}`)."
        )
        lines.append(
            f"- `model_bank_on` bank-attributed self CUDA time is `{bank.bank_cuda_ms:.3f} ms` "
            f"for the sampled profile window."
        )
        lines.append(
            f"- `model_bank_on` bank-attributed self CPU time is `{bank.bank_cpu_ms:.3f} ms` "
            f"for the sampled profile window."
        )
        lines.append(
            "- If bank mode is slower while `esamp.bank_*` time is high, likely bottlenecks are "
            "`index_select`/`bmm` materialization and optimizer step overhead."
        )
    lines.append("")
    os.makedirs(os.path.dirname(report_md), exist_ok=True)
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    args = _parse_args()
    if int(args.sampling_n) <= 1:
        raise RuntimeError("--sampling-n must be > 1")
    os.makedirs(args.trace_dir, exist_ok=True)

    prompts = read_prompts(args.prompt_file, args.prompt)
    bench_prompts = build_prompt_batch(prompts, int(args.benchmark_batch_size))
    rows = int(args.graph_scratch_rows) if int(args.graph_scratch_rows) > 0 else max(
        64, len(bench_prompts) * int(args.sampling_n)
    )
    params = _build_sampling_params(
        prompts=bench_prompts,
        max_new_tokens=int(args.benchmark_max_new_tokens),
        sampling_n=int(args.sampling_n),
        sampling_temperature=float(args.sampling_temperature),
        sampling_top_p=float(args.sampling_top_p),
        sampling_top_k=int(args.sampling_top_k),
        sampling_min_p=float(args.sampling_min_p),
        ignore_eos=True,
        sampling_seed=(
            int(args.sampling_seed)
            if args.sampling_seed is not None
            else (int(args.seed_base) if args.seed_base is not None else None)
        ),
        sampling_per_request_seed=bool(args.sampling_per_request_seed),
    )

    llm = core.make_llm(
        model_name=args.model_name,
        dtype=args.dtype,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        enable_prefix_caching=False,
        enforce_eager=bool(args.enforce_eager),
    )
    mode_names = [x.strip() for x in str(args.modes).split(",") if x.strip()]
    mode_specs = [_to_mode_spec(x) for x in mode_names]

    results: List[ModeResult] = []
    try:
        for mode in mode_specs:
            print(f"[profile] mode={mode.name}")
            r = _profile_one_mode(
                mode=mode,
                llm=llm,
                prompts=bench_prompts,
                params=params,
                args=args,
                rows=int(rows),
                trace_dir=args.trace_dir,
            )
            print(
                f"[profile] done mode={mode.name} tok/s={r.out_tok_per_s:.3f} "
                f"peak_mem_gb={r.peak_mem_gb:.3f} loss_avg={r.loss_avg:.6f} loss_count={r.loss_count}"
            )
            results.append(r)
    finally:
        shutdown_llm_instance(llm, cooldown_s=float(args.cooldown_s))

    raw = {
        "args": vars(args),
        "results": [
            {
                "name": r.name,
                "req_per_s": r.req_per_s,
                "comp_per_s": r.comp_per_s,
                "out_tok_per_s": r.out_tok_per_s,
                "loss_avg": r.loss_avg,
                "loss_count": r.loss_count,
                "elapsed_s": r.elapsed_s,
                "peak_mem_gb": r.peak_mem_gb,
                "total_output_tokens": r.total_output_tokens,
                "total_requests": r.total_requests,
                "total_completions": r.total_completions,
                "trace_path": r.trace_path,
                "top_ops": r.top_ops,
                "esamp_cuda_ms": r.esamp_cuda_ms,
                "bank_cuda_ms": r.bank_cuda_ms,
                "esamp_cpu_ms": r.esamp_cpu_ms,
                "bank_cpu_ms": r.bank_cpu_ms,
                "top_metric": r.top_metric,
            }
            for r in results
        ],
    }
    os.makedirs(os.path.dirname(args.raw_json), exist_ok=True)
    with open(args.raw_json, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    _write_report(args=args, results=results, report_md=args.report_md)
    print(f"[profile] report_md={args.report_md}")
    print(f"[profile] raw_json={args.raw_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
