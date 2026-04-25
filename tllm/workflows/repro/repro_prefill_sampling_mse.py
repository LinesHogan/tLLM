#!/usr/bin/env python3
"""Repro script: teacher-forcing style prefill MSE validation.

Requested logic:
1) gold stage: generate sequences first.
2) teacher-forcing stage: use (prompt + generated tokens) as new inputs,
   then capture prefill hidden for those full sequences.
3) compare gold(one-by-one) vs parallel(batched) prefill hidden by MSE.

The script runs two phases:
- Phase A: generation with n=1 greedy.
- Phase B: generation with n>1 sampling.

Notes:
- Prefill producer in current runtime is eager-first, so we use enforce_eager=True.
- Prefix caching is disabled to avoid cache-hit empty prefill captures.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Sequence, Tuple

import torch
from vllm import SamplingParams

from tllm.workflows.repro import prefill_capture_support as capture_support


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", type=str, default="")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--max-model-len", type=int, default=256)

    parser.add_argument("--gen-max-new-tokens", type=int, default=4)
    parser.add_argument("--sampling-n", type=int, default=3)
    parser.add_argument("--sampling-temperature", type=float, default=0.8)
    parser.add_argument("--sampling-top-p", type=float, default=0.95)

    parser.add_argument(
        "--prefill-check-max-new-tokens",
        type=int,
        default=1,
        help="Decode length used during teacher-forcing prefill capture runs.",
    )
    parser.add_argument("--mse-tol", type=float, default=1e-5)
    parser.add_argument("--seed-base-greedy", type=int, default=1000)
    parser.add_argument("--seed-base-sampling", type=int, default=3000)
    parser.add_argument("--seed-base-prefill-check", type=int, default=7000)
    parser.add_argument(
        "--run-phase-a",
        action="store_true",
        help="Run phase A (n=1 greedy generation -> teacher-forcing prefill MSE).",
    )
    parser.add_argument(
        "--no-run-phase-a",
        dest="run_phase_a",
        action="store_false",
        help="Disable phase A.",
    )
    parser.add_argument(
        "--run-phase-b",
        action="store_true",
        help="Run phase B (n>1 sampling generation -> teacher-forcing prefill MSE).",
    )
    parser.add_argument(
        "--no-run-phase-b",
        dest="run_phase_b",
        action="store_false",
        help="Disable phase B.",
    )
    parser.add_argument(
        "--graph-scratch-rows",
        type=int,
        default=0,
        help="Decode scratch rows. 0 => auto=max(64, len(prompts) * sampling_n).",
    )
    capture_group = parser.add_mutually_exclusive_group()
    capture_group.add_argument(
        "--capture-layer-path",
        type=str,
        default="model.model.layers[0]",
        help="Module path to hook, e.g. `model.model.layers[0]`.",
    )
    capture_group.add_argument(
        "--capture-layer-index",
        type=int,
        default=None,
        help="Layer index under model.model.layers to capture.",
    )
    parser.set_defaults(run_phase_a=True)
    parser.set_defaults(run_phase_b=True)
    return parser.parse_args()


def _assert_non_empty_capture(captured: Dict[int, List[torch.Tensor]], name: str) -> None:
    missing = [pidx for pidx, rows in captured.items() if len(rows) == 0]
    if missing:
        raise RuntimeError(f"{name} empty prefill capture for indices: {missing}")


def _compare_prefill_capture(
    name: str,
    gold: Dict[int, List[torch.Tensor]],
    parallel: Dict[int, List[torch.Tensor]],
    mse_tol: float,
) -> None:
    max_mse = 0.0
    max_idx = -1
    max_step = -1

    for idx, gold_rows in gold.items():
        par_rows = parallel.get(idx, [])
        if len(gold_rows) != len(par_rows):
            raise RuntimeError(
                f"{name} length mismatch idx={idx}: gold={len(gold_rows)} parallel={len(par_rows)}"
            )

        for step_idx, (g, p) in enumerate(zip(gold_rows, par_rows)):
            mse = torch.mean((g - p) ** 2).item()
            if mse > max_mse:
                max_mse = mse
                max_idx = int(idx)
                max_step = int(step_idx)
            if mse > mse_tol:
                raise RuntimeError(
                    f"{name} MSE too large at idx={idx} step={step_idx}: {mse} > {mse_tol}"
                )

    print(f"{name}: PASS max_mse={max_mse:.3e} at idx={max_idx} step={max_step}")


def _encode_prompt(tokenizer, prompt: str) -> List[int]:
    # HF-style tokenizer interface.
    return [int(x) for x in tokenizer.encode(prompt, add_special_tokens=False)]


def _build_teacher_prompts_from_generated(
    llm,
    prompts: Sequence[str],
    params_per_prompt: Sequence[SamplingParams],
) -> Tuple[List[Dict[str, List[int]]], List[Tuple[int, int]]]:
    """Generate first, then return token prompts for teacher-forcing prefill.

    Returns:
    - teacher_prompts: list of {'prompt_token_ids': full_ids}
    - labels: list of (orig_prompt_idx, sample_idx)
    """
    tokenizer = llm.get_tokenizer()

    teacher_prompts: List[Dict[str, List[int]]] = []
    labels: List[Tuple[int, int]] = []

    for i, prompt in enumerate(prompts):
        prompt_token_ids = _encode_prompt(tokenizer, prompt)

        # Generate on one prompt to build canonical sequence set for this prompt.
        sample_tokens = capture_support.run_generate_with_sample_tokens(
            llm=llm,
            prompts=[prompt],
            params=[params_per_prompt[i]],
        )

        branches = sample_tokens.get(0, {})
        if not branches:
            raise RuntimeError(f"No generated tokens collected for prompt index {i}")

        for sample_idx in sorted(branches.keys()):
            gen_tokens = branches[sample_idx]
            full_ids = prompt_token_ids + [int(t) for t in gen_tokens]
            teacher_prompts.append({"prompt_token_ids": full_ids})
            labels.append((int(i), int(sample_idx)))

    return teacher_prompts, labels


def _collect_teacher_prefill_gold_parallel(
    llm,
    teacher_prompts: Sequence[Dict[str, List[int]]],
    prefill_check_max_new_tokens: int,
    seed_base: int,
) -> Tuple[Dict[int, List[torch.Tensor]], Dict[int, List[torch.Tensor]]]:
    """Capture prefill hidden for teacher prompts: gold(one-by-one) vs parallel."""
    gold: Dict[int, List[torch.Tensor]] = {}

    # Gold: one-by-one
    for i, tok_prompt in enumerate(teacher_prompts):
        params = [
                capture_support.build_greedy_params(
                    max_new_tokens=prefill_check_max_new_tokens,
                    seed=seed_base + i,
                )
            ]
        gold[i] = capture_support.run_prefill_capture(llm=llm, prompts=[tok_prompt], params=params).get(0, [])

    # Parallel: all together
    params = [
        capture_support.build_greedy_params(
            max_new_tokens=prefill_check_max_new_tokens,
            seed=seed_base + i,
        )
        for i in range(len(teacher_prompts))
    ]
    parallel = capture_support.run_prefill_capture(
        llm=llm,
        prompts=list(teacher_prompts),
        params=params,
    )

    return gold, parallel


def _print_label_counts(name: str, labels: Sequence[Tuple[int, int]]) -> None:
    by_prompt: Dict[int, int] = {}
    for pidx, _sidx in labels:
        by_prompt[pidx] = by_prompt.get(pidx, 0) + 1
    items = ", ".join([f"prompt={k}:branches={v}" for k, v in sorted(by_prompt.items())])
    print(f"{name} branch-counts: {items}")


def main() -> int:
    args = _parse_args()
    if not args.run_phase_a and not args.run_phase_b:
        raise RuntimeError("At least one phase must be enabled (--run-phase-a/--run-phase-b)")
    if args.run_phase_b and args.sampling_n <= 1:
        raise RuntimeError("--sampling-n must be > 1 when phase B is enabled")

    prompts = capture_support.read_prompts(args.prompt_file, args.prompt)
    if not prompts:
        raise RuntimeError("No prompts provided")

    rows = int(args.graph_scratch_rows) if args.graph_scratch_rows > 0 else max(64, len(prompts) * args.sampling_n)

    capture_support.reset_capture_runtime()
    capture_support.configure_capture_runtime(
        graph_scratch_rows=rows,
        capture_layer_path=args.capture_layer_path,
        capture_layer_index=args.capture_layer_index,
    )

    llm = capture_support.make_llm(
        model_name=args.model_name,
        dtype=args.dtype,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        enable_prefix_caching=False,
        enforce_eager=True,
    )

    try:
        if args.run_phase_a:
            # Phase A: generate with n=1 greedy, then teacher-forcing prefill MSE.
            phase_a_params = [
                capture_support.build_greedy_params(
                    max_new_tokens=int(args.gen_max_new_tokens),
                    seed=int(args.seed_base_greedy) + i,
                )
                for i in range(len(prompts))
            ]
            teacher_a, labels_a = _build_teacher_prompts_from_generated(
                llm=llm,
                prompts=prompts,
                params_per_prompt=phase_a_params,
            )
            _print_label_counts("phase_a(n=1)", labels_a)

            gold_a, par_a = _collect_teacher_prefill_gold_parallel(
                llm=llm,
                teacher_prompts=teacher_a,
                prefill_check_max_new_tokens=int(args.prefill_check_max_new_tokens),
                seed_base=int(args.seed_base_prefill_check),
            )
            _assert_non_empty_capture(gold_a, "phase_a_gold")
            _assert_non_empty_capture(par_a, "phase_a_parallel")
            _compare_prefill_capture("phase_a_prefill_teacher_mse", gold_a, par_a, mse_tol=float(args.mse_tol))

        if args.run_phase_b:
            # Phase B: generate with n>1 sampling, then teacher-forcing prefill MSE.
            phase_b_params = [
                SamplingParams(
                    n=int(args.sampling_n),
                    temperature=float(args.sampling_temperature),
                    top_p=float(args.sampling_top_p),
                    top_k=-1,
                    max_tokens=int(args.gen_max_new_tokens),
                    seed=int(args.seed_base_sampling) + i,
                )
                for i in range(len(prompts))
            ]
            teacher_b, labels_b = _build_teacher_prompts_from_generated(
                llm=llm,
                prompts=prompts,
                params_per_prompt=phase_b_params,
            )
            _print_label_counts(f"phase_b(n={args.sampling_n})", labels_b)

            gold_b, par_b = _collect_teacher_prefill_gold_parallel(
                llm=llm,
                teacher_prompts=teacher_b,
                prefill_check_max_new_tokens=int(args.prefill_check_max_new_tokens),
                seed_base=int(args.seed_base_prefill_check) + 100000,
            )
            _assert_non_empty_capture(gold_b, "phase_b_gold")
            _assert_non_empty_capture(par_b, "phase_b_parallel")
            _compare_prefill_capture("phase_b_prefill_teacher_mse", gold_b, par_b, mse_tol=float(args.mse_tol))

        ran_phases = []
        if args.run_phase_a:
            ran_phases.append("A(n=1)")
        if args.run_phase_b:
            ran_phases.append(f"B(n={args.sampling_n})")
        print("Done: teacher-forcing prefill MSE checks passed " + ", ".join(ran_phases) + ".")
    finally:
        capture_support.shutdown_llm_instance(llm, cooldown_s=0.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
