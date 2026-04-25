#!/usr/bin/env python3
"""Main/eval logic for legacy verify_v1_decode_rows runtime."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import torch

from tllm.legacy import verify_v1_decode_rows_runtime as rt


def _compare_mse(
    gold: Dict[int, List[torch.Tensor]],
    parallel: Dict[int, List[torch.Tensor]],
    mse_tol: float,
) -> None:
    max_mse = 0.0
    max_prompt = -1
    max_step = -1
    print("Per-step MSE:")

    for prompt_idx, gold_steps in gold.items():
        par_steps = parallel.get(prompt_idx, [])
        if len(gold_steps) != len(par_steps):
            raise RuntimeError(
                f"Decode-step length mismatch at prompt {prompt_idx}: "
                f"gold={len(gold_steps)} parallel={len(par_steps)}"
            )
        for step_idx, (g, p) in enumerate(zip(gold_steps, par_steps)):
            mse = torch.mean((g - p) ** 2).item()
            print(f"prompt={prompt_idx} step={step_idx} mse={mse:.6e}")
            if mse > max_mse:
                max_mse = mse
                max_prompt = prompt_idx
                max_step = step_idx
            if mse > mse_tol:
                raise RuntimeError(
                    f"MSE too large at prompt={prompt_idx} step={step_idx}: {mse}"
                )

    print(
        f"PASS first-layer hidden MSE. max_mse={max_mse:.3e} "
        f"at prompt={max_prompt} step={max_step}"
    )


def _validate_non_empty(captured: Dict[int, List[torch.Tensor]], kind: str) -> None:
    empty = [idx for idx, steps in captured.items() if len(steps) == 0]
    if empty:
        raise RuntimeError(
            f"{kind} capture has empty decode hidden for prompts: {empty}"
        )


def _report_step_variation(captured: Dict[int, List[torch.Tensor]], kind: str) -> None:
    print(f"{kind} step variation:")
    for prompt_idx, steps in captured.items():
        if len(steps) <= 1:
            print(f"{kind} prompt={prompt_idx} steps={len(steps)} delta=n/a")
            continue
        deltas = [
            torch.norm(steps[i] - steps[i - 1]).item()
            for i in range(1, len(steps))
        ]
        fp = [float(s[:8].sum().item()) for s in steps]
        unique_fp = len({round(v, 10) for v in fp})
        print(
            f"{kind} prompt={prompt_idx} steps={len(steps)} "
            f"delta_min={min(deltas):.6e} delta_max={max(deltas):.6e} "
            f"fingerprint_unique={unique_fp}"
        )


def _validate_temporal_variation(captured: Dict[int, List[torch.Tensor]], kind: str) -> None:
    multi_step_prompts = [
        (prompt_idx, steps) for prompt_idx, steps in captured.items() if len(steps) > 1
    ]
    if not multi_step_prompts:
        return
    stale_prompts: List[int] = []
    for prompt_idx, steps in multi_step_prompts:
        deltas = [
            torch.norm(steps[i] - steps[i - 1]).item()
            for i in range(1, len(steps))
        ]
        if max(deltas) <= 1e-12:
            stale_prompts.append(prompt_idx)
    if len(stale_prompts) == len(multi_step_prompts):
        raise RuntimeError(
            f"{kind} hidden appears stale across decode steps: "
            f"all multi-step prompts have zero temporal variation. "
            "This usually means capture tap is bypassed by compiled/cudagraph path."
        )


def _print_hidden_sample(
    gold: Dict[int, List[torch.Tensor]],
    parallel: Dict[int, List[torch.Tensor]],
    num_elems: int,
) -> None:
    for prompt_idx, steps in gold.items():
        if not steps:
            continue
        g = steps[0]
        p = parallel[prompt_idx][0]
        k = max(1, min(num_elems, g.numel()))
        print(
            f"sample prompt={prompt_idx} step=0 shape={tuple(g.shape)} "
            f"gold[:{k}]={g.flatten()[:k].tolist()}"
        )
        print(
            f"sample prompt={prompt_idx} step=0 shape={tuple(p.shape)} "
            f"parallel[:{k}]={p.flatten()[:k].tolist()}"
        )
        return
    raise RuntimeError("No hidden sample available to print")


def _read_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]
        if prompts:
            return prompts

    if args.prompt:
        prompts = [p for p in args.prompt if p.strip()]
        if prompts:
            return prompts

    return [
        "Explain gravity in one sentence.",
        "What is 7*8?",
        "Write a short poem about rain.",
        "Give one programming tip.",
        "Define entropy briefly.",
        "Name a planet.",
        "Translate hello to Spanish.",
        "Summarize Hamlet in one line.",
        "List three prime numbers.",
        "Describe a cat in five words.",
        "State a fun fact.",
        "What is the capital of France?",
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--mse-tol", type=float, default=1e-5)
    parser.add_argument("--print-hidden-elems", type=int, default=8)
    parser.add_argument("--debug-step-variation", action="store_true")
    parser.add_argument(
        "--graph-scratch-rows",
        type=int,
        default=0,
        help="Row budget for graph_copy scratch buffer; 0 means auto by max_num_reqs.",
    )
    parser.add_argument(
        "--disable-compile-cache",
        action="store_true",
        help="Set VLLM_DISABLE_COMPILE_CACHE=1 before LLM init.",
    )
    parser.add_argument(
        "--capture-impl",
        type=str,
        default="graph_copy",
        choices=["graph_copy", "python_hook"],
        help="Hidden capture implementation.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Force eager mode for debugging python_hook behavior.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Use float32 if you want near-zero MSE.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rt.CAPTURE_IMPL = args.capture_impl
    if args.max_new_tokens < 2:
        raise RuntimeError(
            "max-new-tokens must be >= 2 for decode-step comparison "
            "(max_new_tokens=1 has no decode step to capture)."
        )
    prompts = _read_prompts(args)
    if args.disable_compile_cache:
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    if args.graph_scratch_rows > 0:
        rt.GRAPH_SCRATCH_ROWS_HINT = int(args.graph_scratch_rows)
    else:
        rt.GRAPH_SCRATCH_ROWS_HINT = max(32, len(prompts))

    llm = rt.LLM(
        model=args.model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.2,
        max_model_len=256,
        enforce_eager=bool(args.enforce_eager),
        dtype=args.dtype,
    )
    gold: Dict[int, List[torch.Tensor]] = {}
    for i, prompt in enumerate(prompts):
        params = [rt._build_greedy_params(args.max_new_tokens, seed=1000 + i)]
        captured = rt._run_with_capture(llm, [prompt], params)
        gold[i] = captured.get(0, [])

    parallel_params = [
        rt._build_greedy_params(args.max_new_tokens, seed=1000 + i) for i in range(len(prompts))
    ]
    parallel = rt._run_with_capture(llm, prompts, parallel_params)

    try:
        _validate_non_empty(gold, "gold")
        _validate_non_empty(parallel, "parallel")
    except RuntimeError as e:
        if rt.CAPTURE_IMPL == "python_hook" and not bool(args.enforce_eager):
            raise RuntimeError(
                f"{e} ; python_hook is typically bypassed under compiled/cudagraph v1. "
                "Use --enforce-eager for python_hook baseline."
            ) from e
        raise
    _validate_temporal_variation(gold, "gold")
    _validate_temporal_variation(parallel, "parallel")
    if args.debug_step_variation:
        _report_step_variation(gold, "gold")
        _report_step_variation(parallel, "parallel")
    _compare_mse(gold, parallel, mse_tol=args.mse_tol)
    _print_hidden_sample(gold, parallel, num_elems=args.print_hidden_elems)
    print(
        f"prompts={len(prompts)} max_new_tokens={args.max_new_tokens} "
        f"dtype={args.dtype} hook={rt.HOOK_LAYER_INFO} capture_impl={rt.CAPTURE_IMPL} "
        f"enforce_eager={bool(args.enforce_eager)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
