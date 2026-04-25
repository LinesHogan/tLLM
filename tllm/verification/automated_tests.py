#!/usr/bin/env python3
"""Automated GPU verification runner for the producer/consumer stack."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    project: str
    description: str
    command: List[str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv-python", type=str, default=str(DEFAULT_VENV_PYTHON))
    parser.add_argument("--list", action="store_true", help="List scenarios and exit.")
    parser.add_argument(
        "--project",
        action="append",
        default=[],
        help="Run specific project(s): unit/decode/prefill/throughput/esamp.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Run only explicit scenario id(s).",
    )
    parser.add_argument("--continue-on-fail", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=0.0, help="Per-scenario timeout. 0 means no timeout.")

    parser.add_argument(
        "--prompt-file",
        type=str,
        default="test/prompt_debug_list.txt",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--mse-tol", type=float, default=1e-5)
    capture_group = parser.add_mutually_exclusive_group()
    capture_group.add_argument(
        "--capture-layer-path",
        type=str,
        default="model.model.layers[0]",
        help=(
            "Module path to hook, e.g. `model.model.layers[0]` or "
            "`model.model.layers[5]`. Ignored when --capture-layer-index is set."
        ),
    )
    capture_group.add_argument(
        "--capture-layer-index",
        type=int,
        default=None,
        help="Layer index under model.model.layers to capture (supports negative index).",
    )

    parser.add_argument(
        "--decode-model-7b",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--decode-max-new-tokens", type=int, default=8)
    parser.add_argument("--decode-gpu-memory-utilization", type=float, default=0.8)

    parser.add_argument(
        "--throughput-model-small",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--throughput-model-large",
        type=str,
        default="Qwen/Qwen3-4B",
    )
    parser.add_argument("--throughput-gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--throughput-batch-size", type=int, default=64)
    parser.add_argument("--throughput-max-new-tokens", type=int, default=128)
    parser.add_argument("--throughput-warmup-rounds", type=int, default=1)
    parser.add_argument("--throughput-rounds", type=int, default=5)
    parser.add_argument("--throughput-cooldown-s", type=float, default=2.0)
    parser.add_argument("--throughput-ignore-eos", action="store_true")
    parser.add_argument("--no-throughput-ignore-eos", dest="throughput_ignore_eos", action="store_false")
    parser.add_argument(
        "--throughput-disable-prefix-caching",
        action="store_true",
    )
    parser.add_argument(
        "--no-throughput-disable-prefix-caching",
        dest="throughput_disable_prefix_caching",
        action="store_false",
    )
    parser.set_defaults(throughput_ignore_eos=True)
    parser.set_defaults(throughput_disable_prefix_caching=True)

    parser.add_argument(
        "--esamp-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument("--esamp-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--esamp-batch-size", type=int, default=64)
    parser.add_argument("--esamp-max-new-tokens", type=int, default=128)
    parser.add_argument("--esamp-warmup-rounds", type=int, default=1)
    parser.add_argument("--esamp-rounds", type=int, default=3)
    parser.add_argument("--esamp-hidden-dim", type=int, default=256)
    parser.add_argument("--esamp-lr", type=float, default=1e-3)
    return parser.parse_args()


def _build_scenarios(args: argparse.Namespace) -> List[Scenario]:
    py = args.venv_python

    capture_args: List[str] = []
    if args.capture_layer_index is not None:
        capture_args = ["--capture-layer-index", str(args.capture_layer_index)]
    else:
        capture_args = ["--capture-layer-path", str(args.capture_layer_path)]

    common_decode = [
        "--prompt-file",
        args.prompt_file,
        "--dtype",
        args.dtype,
        "--mse-tol",
        str(args.mse_tol),
        "--max-model-len",
        str(args.max_model_len),
        *capture_args,
    ]

    prefill_common = [
        "--prompt-file",
        args.prompt_file,
        "--dtype",
        args.dtype,
        "--mse-tol",
        str(args.mse_tol),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        *capture_args,
    ]

    throughput_common = [
        "--prompt-file",
        args.prompt_file,
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.throughput_gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--benchmark-batch-size",
        str(args.throughput_batch_size),
        "--benchmark-max-new-tokens",
        str(args.throughput_max_new_tokens),
        "--benchmark-warmup-rounds",
        str(args.throughput_warmup_rounds),
        "--benchmark-rounds",
        str(args.throughput_rounds),
        "--cooldown-s",
        str(args.throughput_cooldown_s),
        *capture_args,
    ]
    if args.throughput_ignore_eos:
        throughput_common.append("--benchmark-ignore-eos")
    else:
        throughput_common.append("--no-benchmark-ignore-eos")
    if args.throughput_disable_prefix_caching:
        throughput_common.append("--benchmark-disable-prefix-caching")
    else:
        throughput_common.append("--no-benchmark-disable-prefix-caching")

    esamp_common = [
        "--prompt-file",
        args.prompt_file,
        "--dtype",
        args.dtype,
        "--enforce-eager",
        "--gpu-memory-utilization",
        str(args.esamp_gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--benchmark-batch-size",
        str(args.esamp_batch_size),
        "--benchmark-max-new-tokens",
        str(args.esamp_max_new_tokens),
        "--benchmark-warmup-rounds",
        str(args.esamp_warmup_rounds),
        "--benchmark-rounds",
        str(args.esamp_rounds),
        "--distiller-hidden-dim",
        str(args.esamp_hidden_dim),
        "--distiller-lr",
        str(args.esamp_lr),
        "--source-layer-path",
        "model.model.layers[0].input_layernorm",
        "--target-layer-path",
        "model.model.layers[-1].input_layernorm",
    ]
    if args.throughput_ignore_eos:
        esamp_common.append("--benchmark-ignore-eos")
    else:
        esamp_common.append("--no-benchmark-ignore-eos")
    if args.throughput_disable_prefix_caching:
        esamp_common.append("--benchmark-disable-prefix-caching")
    else:
        esamp_common.append("--no-benchmark-disable-prefix-caching")

    return [
        Scenario(
            scenario_id="unit_decode_n_gt_1_localization",
            project="unit",
            description="Pure-function unit tests for decode n>1 localization.",
            command=[
                py,
                "-m",
                "pytest",
                "-q",
                "test/test_decode_localization_unit.py",
            ],
        ),
        Scenario(
            scenario_id="decode_greedy_parallel_gold_7b",
            project="decode",
            description="Greedy decode hidden MSE (parallel vs gold) on 7B model.",
            command=[
                py,
                "-m",
                "verify_v1_decode_rows_minimal",
                "--model-name",
                args.decode_model_7b,
                "--max-new-tokens",
                str(args.decode_max_new_tokens),
                "--gpu-memory-utilization",
                str(args.decode_gpu_memory_utilization),
                *common_decode,
            ],
        ),
        Scenario(
            scenario_id="prefill_parallel_gold_n_eq_1",
            project="prefill",
            description="Teacher-forcing prefill MSE for n=1 branch localization.",
            command=[
                py,
                "-m",
                "tllm.workflows.repro.repro_prefill_sampling_mse",
                "--model-name",
                args.throughput_model_small,
                "--run-phase-a",
                "--no-run-phase-b",
                *prefill_common,
            ],
        ),
        Scenario(
            scenario_id="prefill_parallel_gold_n_gt_1",
            project="prefill",
            description="Teacher-forcing prefill MSE for n>1 sample localization.",
            command=[
                py,
                "-m",
                "tllm.workflows.repro.repro_prefill_sampling_mse",
                "--model-name",
                args.throughput_model_small,
                "--no-run-phase-a",
                "--run-phase-b",
                "--sampling-n",
                "3",
                "--sampling-temperature",
                "0.8",
                "--sampling-top-p",
                "0.95",
                *prefill_common,
            ],
        ),
        Scenario(
            scenario_id="esamp_layer0_to_last_qwen2p5_0p5b",
            project="esamp",
            description="Delayed side-train benchmark: layer[0] -> layer[-1] on Qwen2.5-0.5B.",
            command=[
                py,
                "-m",
                "tllm.workflows.benchmarks.esamp_benchmark",
                "--model-name",
                args.esamp_model,
                *esamp_common,
            ],
        ),
        Scenario(
            scenario_id="esamp_loss_parity_qwen2p5_0p5b",
            project="esamp",
            description="Single-path ESamp aligned verification with historical parity record on Qwen2.5-0.5B.",
            command=[
                py,
                "-m",
                "tllm.workflows.repro.repro_esamp_loss_parity",
                "--model-name",
                args.esamp_model,
                "--prompt-file",
                args.prompt_file,
                "--dtype",
                args.dtype,
                "--gpu-memory-utilization",
                "0.5",
                "--max-model-len",
                "512",
                "--benchmark-batch-size",
                "8",
                "--benchmark-max-new-tokens",
                "256",
                "--benchmark-warmup-rounds",
                "1",
                "--benchmark-rounds",
                "1",
                "--benchmark-ignore-eos",
                "--benchmark-disable-prefix-caching",
                "--sampling-n",
                "16",
                "--sampling-temperature",
                "0.8",
                "--sampling-top-p",
                "0.95",
                "--sampling-top-k",
                "-1",
                "--sampling-seed",
                "1234",
                "--distiller-hidden-dim",
                str(args.esamp_hidden_dim),
                "--distiller-lr",
                str(args.esamp_lr),
                "--source-layer-path",
                "model.model.layers[0].input_layernorm",
                "--target-layer-path",
                "model.model.layers[-1].input_layernorm",
                "--model-bank-flush-interval",
                "1",
                "--model-bank-initializer",
                "svd",
                "--model-bank-initializer-svd-method",
                "ffn_fast_svd",
                "--trajectory-topk",
                "1",
                "--model-bank-train-cudagraph",
            ],
        ),
    ]


def _select_scenarios(args: argparse.Namespace, scenarios: Sequence[Scenario]) -> List[Scenario]:
    selected = list(scenarios)
    if args.project:
        projects = {x.strip() for x in args.project if x.strip()}
        selected = [s for s in selected if s.project in projects]
    if args.scenario:
        ids = {x.strip() for x in args.scenario if x.strip()}
        selected = [s for s in selected if s.scenario_id in ids]
    return selected


def _run_one(s: Scenario, timeout_s: float, dry_run: bool) -> tuple[bool, float]:
    command_str = " ".join(shlex.quote(x) for x in s.command)
    print(f"[{s.project}] {s.scenario_id}")
    print(f"  desc: {s.description}")
    print(f"  cmd:  {command_str}")

    if dry_run:
        return True, 0.0

    start = time.perf_counter()
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    repo_str = str(REPO_ROOT)
    env["PYTHONPATH"] = repo_str if not py_path else (repo_str + ":" + py_path)
    try:
        subprocess.run(
            s.command,
            cwd=str(REPO_ROOT),
            env=env,
            check=True,
            timeout=(None if timeout_s <= 0 else timeout_s),
        )
        elapsed = time.perf_counter() - start
        print(f"  result: PASS ({elapsed:.2f}s)")
        return True, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        print(f"  result: FAIL (timeout after {elapsed:.2f}s)")
        return False, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - start
        print(f"  result: FAIL (exit={e.returncode}, {elapsed:.2f}s)")
        return False, elapsed


def main() -> int:
    args = _parse_args()
    scenarios = _build_scenarios(args)

    if args.list:
        for s in scenarios:
            print(f"{s.scenario_id}\tproject={s.project}\t{s.description}")
        return 0

    selected = _select_scenarios(args, scenarios)
    if not selected:
        print("No scenarios selected. Use --list to inspect available ids/projects.")
        return 2

    passed = 0
    failed = 0
    elapsed_by_id: Dict[str, float] = {}
    for s in selected:
        ok, elapsed = _run_one(s, timeout_s=float(args.timeout_s), dry_run=bool(args.dry_run))
        elapsed_by_id[s.scenario_id] = elapsed
        if ok:
            passed += 1
            continue
        failed += 1
        if not args.continue_on_fail:
            break

    print("=== Summary ===")
    print(f"selected={len(selected)} passed={passed} failed={failed}")
    for sid, elapsed in elapsed_by_id.items():
        print(f"  {sid}: {elapsed:.2f}s")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
