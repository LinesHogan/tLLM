#!/usr/bin/env python3
"""Non-core utility functions for verification and benchmarking scripts."""

from __future__ import annotations

import gc
import time
from typing import Callable, Dict, List, Optional

import torch
from vllm import LLM, SamplingParams


def build_greedy_params(max_new_tokens: int, seed: int, ignore_eos: bool = False) -> SamplingParams:
    return SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=max_new_tokens,
        min_tokens=max_new_tokens if ignore_eos else 0,
        ignore_eos=ignore_eos,
        seed=seed,
    )


def sum_output_tokens(outputs) -> int:
    total = 0
    for out in outputs:
        if not getattr(out, "outputs", None):
            continue
        total += len(out.outputs[0].token_ids)
    return total


def shutdown_llm_instance(llm: LLM, cooldown_s: float) -> None:
    try:
        engine = getattr(llm, "llm_engine", None)
        if engine is not None and hasattr(engine, "shutdown"):
            engine.shutdown()
        engine_core = getattr(engine, "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown()
    except Exception as e:
        print(f"[benchmark warning] explicit shutdown failed: {e}")
    finally:
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception as e:
                print(f"[benchmark warning] destroy_process_group failed: {e}")
        if cooldown_s > 0:
            time.sleep(cooldown_s)


def build_prompt_batch(base_prompts: List[str], batch_size: int) -> List[str]:
    if batch_size <= len(base_prompts):
        return base_prompts[:batch_size]
    prompts: List[str] = []
    for i in range(batch_size):
        base = base_prompts[i % len(base_prompts)]
        prompts.append(f"{base} [bench-{i}]")
    return prompts


def make_plain_llm(
    model_name: str,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    enable_prefix_caching: bool,
    enforce_eager: bool = False,
    seed: int | None = None,
) -> LLM:
    kwargs = {}
    if seed is not None:
        kwargs["seed"] = int(seed)
    return LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=float(gpu_memory_utilization),
        max_model_len=int(max_model_len),
        enforce_eager=bool(enforce_eager),
        dtype=dtype,
        enable_prefix_caching=bool(enable_prefix_caching),
        **kwargs,
    )


def run_throughput_case(
    llm: LLM,
    prompts: List[str],
    max_new_tokens: int,
    warmup_rounds: int,
    rounds: int,
    consumer_enabled: bool,
    ignore_eos: bool,
    log_memory: bool,
    set_consumer_enabled_fn: Callable[[bool], None],
    synchronize_runtime_fn: Callable[[], None],
    print_mem_fn: Callable[[str], None],
    seed_base: int = 2000,
) -> Dict[str, float]:
    set_consumer_enabled_fn(consumer_enabled)
    synchronize_runtime_fn()

    params = [
        build_greedy_params(max_new_tokens, seed=seed_base + i, ignore_eos=ignore_eos)
        for i in range(len(prompts))
    ]

    case_label = "with_consumer" if consumer_enabled else "baseline"
    if log_memory:
        print_mem_fn(f"[{case_label}] before_warmup")

    for warmup_i in range(warmup_rounds):
        llm.generate(prompts, params)
        if log_memory:
            print_mem_fn(f"[{case_label}] warmup_{warmup_i + 1}/{warmup_rounds}")

    torch.cuda.synchronize()
    start = time.perf_counter()
    total_requests = 0
    total_output_tokens = 0

    for round_i in range(rounds):
        outputs = llm.generate(prompts, params)
        total_requests += len(outputs)
        total_output_tokens += sum_output_tokens(outputs)
        if log_memory:
            print_mem_fn(f"[{case_label}] round_{round_i + 1}/{rounds}")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    req_per_s = float(total_requests / elapsed) if elapsed > 0 else 0.0
    out_tok_per_s = float(total_output_tokens / elapsed) if elapsed > 0 else 0.0

    return {
        "elapsed_s": float(elapsed),
        "requests": float(total_requests),
        "output_tokens": float(total_output_tokens),
        "req_per_s": req_per_s,
        "out_tok_per_s": out_tok_per_s,
    }


def average_case_results(results: List[Dict[str, float]]) -> Dict[str, float]:
    if not results:
        return {
            "elapsed_s": 0.0,
            "requests": 0.0,
            "output_tokens": 0.0,
            "req_per_s": 0.0,
            "out_tok_per_s": 0.0,
        }
    keys = ["elapsed_s", "requests", "output_tokens", "req_per_s", "out_tok_per_s"]
    n = float(len(results))
    return {k: sum(r[k] for r in results) / n for k in keys}


def validate_non_empty(captured: Dict[int, List[torch.Tensor]], name: str) -> None:
    missing = [i for i, steps in captured.items() if len(steps) == 0]
    if missing:
        raise RuntimeError(f"{name} capture empty for prompts: {missing}")


def compare_mse(
    gold: Dict[int, List[torch.Tensor]],
    parallel: Dict[int, List[torch.Tensor]],
    mse_tol: float,
    gold_tokens: Optional[Dict[int, List[int]]] = None,
    parallel_tokens: Optional[Dict[int, List[int]]] = None,
) -> None:
    max_mse = 0.0
    max_prompt = -1
    max_step = -1

    print("Per-step MSE:")
    for prompt_idx, gold_steps in gold.items():
        par_steps = parallel.get(prompt_idx, [])
        if len(gold_steps) != len(par_steps):
            raise RuntimeError(
                f"Decode-step mismatch at prompt={prompt_idx}: gold={len(gold_steps)} parallel={len(par_steps)}"
            )
        for step_idx, (g, p) in enumerate(zip(gold_steps, par_steps)):
            mse = torch.mean((g - p) ** 2).item()
            gold_tok = None
            if gold_tokens is not None:
                seq = gold_tokens.get(prompt_idx, [])
                if step_idx < len(seq):
                    gold_tok = int(seq[step_idx])
            par_tok = None
            if parallel_tokens is not None:
                seq = parallel_tokens.get(prompt_idx, [])
                if step_idx < len(seq):
                    par_tok = int(seq[step_idx])

            tok_info = ""
            if gold_tok is not None or par_tok is not None:
                tok_info = f" gold_tok={gold_tok} parallel_tok={par_tok}"
            print(f"prompt={prompt_idx} step={step_idx} mse={mse:.6e}{tok_info}")
            if mse > max_mse:
                max_mse = mse
                max_prompt = prompt_idx
                max_step = step_idx
            if mse > mse_tol:
                raise RuntimeError(
                    f"MSE too large at prompt={prompt_idx} step={step_idx}: {mse} > {mse_tol}"
                    f" gold_tok={gold_tok} parallel_tok={par_tok}"
                    f" example gold={g[:8]} parallel={p[:8]}"
                )

    print(f"PASS capture-layer hidden MSE. max_mse={max_mse:.3e} at prompt={max_prompt} step={max_step}")


def read_prompts(prompt_file: str, prompt_args: List[str]) -> List[str]:
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]
        if prompts:
            return prompts

    if prompt_args:
        prompts = [p for p in prompt_args if p.strip()]
        if prompts:
            return prompts

    return [
        "1+2=3; 5+4=",
        "4*9=36; 7*8=",
    ]


def gpu_mem_gib() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "free": 0.0, "total": 0.0}
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    gib = 1024 ** 3
    return {
        "allocated": float(allocated / gib),
        "reserved": float(reserved / gib),
        "free": float(free / gib),
        "total": float(total / gib),
    }


def print_gpu_mem(prefix: str) -> None:
    m = gpu_mem_gib()
    print(
        f"{prefix} gpu_mem_gib: allocated={m['allocated']:.2f} "
        f"reserved={m['reserved']:.2f} free={m['free']:.2f}/{m['total']:.2f}"
    )
