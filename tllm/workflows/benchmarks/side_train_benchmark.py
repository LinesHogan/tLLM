#!/usr/bin/env python3
"""Benchmark delayed side-stream dummy training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Dict, List

from tllm.runtime import residual_runtime as core
from tllm.workflows import side_train_support as esamp_workflow_support
from tllm.util.tools import build_prompt_batch, print_gpu_mem, read_prompts, shutdown_llm_instance

ESAMP_MIN_OUT_TOK_RATIO = 0.95
JSON_SUMMARY_PREFIX = "JSON_SUMMARY:"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", type=str, default="")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")

    parser.add_argument("--graph-scratch-rows", type=int, default=0)
    parser.add_argument(
        "--tap-layer-path",
        action="append",
        default=[],
        help="Can be repeated. If empty, source/target layer paths are used.",
    )
    parser.add_argument("--source-layer-path", type=str, default="model.model.layers[0].input_layernorm")
    parser.add_argument("--target-layer-path", type=str, default="model.model.layers[-1].input_layernorm")

    parser.add_argument("--side-hidden-dim", type=int, default=256)
    parser.add_argument("--side-lr", type=float, default=1e-3)
    parser.add_argument("--benchmark-batch-size", type=int, default=64)
    parser.add_argument("--benchmark-max-new-tokens", type=int, default=128)
    parser.add_argument("--benchmark-warmup-rounds", type=int, default=1)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument("--benchmark-bidirectional", action="store_true")
    parser.add_argument("--no-benchmark-bidirectional", dest="benchmark_bidirectional", action="store_false")

    parser.add_argument("--benchmark-ignore-eos", action="store_true")
    parser.add_argument("--no-benchmark-ignore-eos", dest="benchmark_ignore_eos", action="store_false")
    parser.add_argument("--benchmark-disable-prefix-caching", action="store_true")
    parser.add_argument(
        "--no-benchmark-disable-prefix-caching",
        dest="benchmark_disable_prefix_caching",
        action="store_false",
    )
    parser.add_argument("--benchmark-case-cooldown-s", type=float, default=2.0)
    parser.add_argument("--benchmark-log-memory", action="store_true")
    parser.add_argument("--emit-json-summary", action="store_true")

    parser.set_defaults(benchmark_bidirectional=True)
    parser.set_defaults(benchmark_ignore_eos=True)
    parser.set_defaults(benchmark_disable_prefix_caching=True)
    parser.set_defaults(enforce_eager=True)
    return parser.parse_args()


def _average_results(results: List[Dict[str, float]]) -> Dict[str, float]:
    if not results:
        return {
            "elapsed_s": 0.0,
            "requests": 0.0,
            "output_tokens": 0.0,
            "req_per_s": 0.0,
            "out_tok_per_s": 0.0,
            "loss_avg": 0.0,
            "loss_count": 0.0,
        }
    keys = ["elapsed_s", "requests", "output_tokens", "req_per_s", "out_tok_per_s", "loss_avg", "loss_count"]
    n = float(len(results))
    return {k: sum(r.get(k, 0.0) for r in results) / n for k in keys}


def _compute_esamp_ratio(legacy_with_train: Dict[str, float], base_consumer_with_train: Dict[str, float]) -> float:
    legacy_tok = float(legacy_with_train.get("out_tok_per_s", 0.0) or 0.0)
    if legacy_tok <= 0:
        return 0.0
    return float(base_consumer_with_train.get("out_tok_per_s", 0.0) or 0.0) / legacy_tok


def _build_comparison_summary(
    legacy_summary: Dict[str, Dict[str, float]],
    base_consumer_summary: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    ratio = _compute_esamp_ratio(legacy_summary["with_train"], base_consumer_summary["with_train"])
    return {
        "legacy": legacy_summary,
        "base_consumer": base_consumer_summary,
        "ratio": ratio,
        "min_ratio": ESAMP_MIN_OUT_TOK_RATIO,
        "passed": bool(ratio >= ESAMP_MIN_OUT_TOK_RATIO),
    }


def _append_bool_flag(cmd: List[str], enabled: bool, positive_flag: str, negative_flag: str | None = None) -> None:
    if enabled:
        cmd.append(positive_flag)
    elif negative_flag is not None:
        cmd.append(negative_flag)


def _build_subprocess_cmd(args: argparse.Namespace, implementation: str) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "tllm.workflows.benchmarks.side_train_benchmark",
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
        "--side-hidden-dim",
        str(args.side_hidden_dim),
        "--side-lr",
        str(args.side_lr),
        "--consumer-implementation",
        implementation,
        "--benchmark-batch-size",
        str(args.benchmark_batch_size),
        "--benchmark-max-new-tokens",
        str(args.benchmark_max_new_tokens),
        "--benchmark-warmup-rounds",
        str(args.benchmark_warmup_rounds),
        "--benchmark-rounds",
        str(args.benchmark_rounds),
        "--benchmark-case-cooldown-s",
        str(args.benchmark_case_cooldown_s),
        "--emit-json-summary",
    ]
    for prompt in getattr(args, "prompt", []):
        cmd.extend(["--prompt", str(prompt)])
    prompt_file = str(getattr(args, "prompt_file", "") or "")
    if prompt_file:
        cmd.extend(["--prompt-file", prompt_file])
    for path in getattr(args, "tap_layer_path", []):
        cmd.extend(["--tap-layer-path", str(path)])
    _append_bool_flag(cmd, bool(args.enforce_eager), "--enforce-eager", "--no-enforce-eager")
    _append_bool_flag(cmd, bool(args.benchmark_bidirectional), "--benchmark-bidirectional", "--no-benchmark-bidirectional")
    _append_bool_flag(cmd, bool(args.benchmark_ignore_eos), "--benchmark-ignore-eos", "--no-benchmark-ignore-eos")
    _append_bool_flag(
        cmd,
        bool(args.benchmark_disable_prefix_caching),
        "--benchmark-disable-prefix-caching",
        "--no-benchmark-disable-prefix-caching",
    )
    if bool(args.benchmark_log_memory):
        cmd.append("--benchmark-log-memory")
    return cmd


def _extract_json_summary(stdout: str) -> Dict[str, object]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(JSON_SUMMARY_PREFIX):
            return json.loads(line[len(JSON_SUMMARY_PREFIX) :])
    raise ValueError("JSON summary marker not found in benchmark subprocess output")


def _run_compare_consumer_implementations(args: argparse.Namespace) -> Dict[str, object]:
    results: Dict[str, Dict[str, float]] = {}
    for implementation in ("legacy", "base_consumer"):
        completed = subprocess.run(
            _build_subprocess_cmd(args, implementation),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"benchmark subprocess failed for {implementation} with exit code {completed.returncode}")
        payload = _extract_json_summary(completed.stdout or "")
        results[implementation] = payload["summary"]

    return _build_comparison_summary(results["legacy"], results["base_consumer"])


def _run_one_implementation(args: argparse.Namespace, implementation: str) -> Dict[str, Dict[str, float]]:
    prompts = read_prompts(args.prompt_file, args.prompt)
    bench_prompts = build_prompt_batch(prompts, int(args.benchmark_batch_size))

    graph_scratch_rows = (
        int(args.graph_scratch_rows) if int(args.graph_scratch_rows) > 0 else max(64, len(bench_prompts))
    )

    tap_paths: List[str] = [p for p in args.tap_layer_path if p.strip()]
    if not tap_paths:
        tap_paths = [args.source_layer_path, args.target_layer_path]

    esamp_workflow_support.configure_esamp_runtime(
        graph_scratch_rows=graph_scratch_rows,
        tap_layer_paths=tap_paths,
        source_layer_path=args.source_layer_path,
        target_layer_path=args.target_layer_path,
        enable_side_train=True,
        side_hidden_dim=int(args.side_hidden_dim),
        side_lr=float(args.side_lr),
    )

    if args.benchmark_log_memory:
        print_gpu_mem(f"[{implementation}] before_llm_init")

    llm = core.make_llm(
        model_name=args.model_name,
        dtype=args.dtype,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        enable_prefix_caching=(not args.benchmark_disable_prefix_caching),
        enforce_eager=bool(args.enforce_eager),
    )

    if args.benchmark_log_memory:
        print_gpu_mem(f"[{implementation}] after_llm_init")

    runs_by_mode: Dict[bool, List[Dict[str, float]]] = {False: [], True: []}
    orders = [[False, True]]
    if args.benchmark_bidirectional:
        orders.append([True, False])

    try:
        for pass_i, order in enumerate(orders):
            order_str = " -> ".join("train_on" if x else "train_off" for x in order)
            print(f"[{implementation}] Pass {pass_i + 1}/{len(orders)} order={order_str}")
            for train_enabled in order:
                result = esamp_workflow_support.run_side_train_throughput_case(
                    llm=llm,
                    prompts=bench_prompts,
                    max_new_tokens=int(args.benchmark_max_new_tokens),
                    warmup_rounds=int(args.benchmark_warmup_rounds),
                    rounds=int(args.benchmark_rounds),
                    train_enabled=bool(train_enabled),
                    ignore_eos=bool(args.benchmark_ignore_eos),
                    log_memory=bool(args.benchmark_log_memory),
                )
                runs_by_mode[bool(train_enabled)].append(result)
                label = "with_train" if train_enabled else "tap_only"
                print(
                    f"  [{implementation}] {label}: elapsed={result['elapsed_s']:.3f}s "
                    f"req/s={result['req_per_s']:.3f} out_tok/s={result['out_tok_per_s']:.3f} "
                    f"loss_avg={result['loss_avg']:.6e} loss_count={int(result['loss_count'])}"
                )
    finally:
        shutdown_llm_instance(llm, cooldown_s=float(args.benchmark_case_cooldown_s))
        if args.benchmark_log_memory:
            print_gpu_mem(f"[{implementation}] after_llm_shutdown")

    tap_only = _average_results(runs_by_mode[False])
    with_train = _average_results(runs_by_mode[True])
    req_ratio = with_train["req_per_s"] / tap_only["req_per_s"] if tap_only["req_per_s"] > 0 else 0.0
    tok_ratio = with_train["out_tok_per_s"] / tap_only["out_tok_per_s"] if tap_only["out_tok_per_s"] > 0 else 0.0

    print(f"Summary [{implementation}]: tap_only")
    print(
        f"  elapsed={tap_only['elapsed_s']:.3f}s req/s={tap_only['req_per_s']:.3f} "
        f"out_tok/s={tap_only['out_tok_per_s']:.3f} requests={int(tap_only['requests'])} "
        f"out_tokens={int(tap_only['output_tokens'])}"
    )
    print(f"Summary [{implementation}]: with_train")
    print(
        f"  elapsed={with_train['elapsed_s']:.3f}s req/s={with_train['req_per_s']:.3f} "
        f"out_tok/s={with_train['out_tok_per_s']:.3f} requests={int(with_train['requests'])} "
        f"out_tokens={int(with_train['output_tokens'])} avg_loss={with_train['loss_avg']:.6e} "
        f"loss_count={int(with_train['loss_count'])}"
    )
    print(f"Relative [{implementation}] (with_train / tap_only): req/s={req_ratio:.4f} out_tok/s={tok_ratio:.4f}")

    return {"tap_only": tap_only, "with_train": with_train}


def main() -> int:
    args = _parse_args()
    if not args.enforce_eager:
        print(
            "[warning] running side-train with enforce_eager=False. "
            "Depending on model/backend this may fail under torch.compile."
        )

    summary = _run_one_implementation(args, "esamp")
    if args.emit_json_summary:
        print(JSON_SUMMARY_PREFIX + json.dumps({"implementation": "esamp", "summary": summary}, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
