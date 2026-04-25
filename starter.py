#!/usr/bin/env python3
"""Minimal ESamp starter: generate 16 parallel answers with Qwen3-1.7B."""

from __future__ import annotations

import argparse
from typing import Sequence

from vllm import SamplingParams

from tllm.runtime import residual_runtime as runtime
from tllm.util.tools import make_llm, shutdown_llm_instance
from tllm.workflows import side_train_support


DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.7B"
DEFAULT_PROMPT = "用两句话介绍 tLLM，并给出一个适合它的使用场景。"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--num-answers", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--source-layer-path", type=str, default="model.model.layers[0].input_layernorm")
    parser.add_argument("--target-layer-path", type=str, default="model.model.layers[-1].input_layernorm")
    parser.add_argument("--graph-scratch-rows", type=int, default=0)
    parser.add_argument("--side-hidden-dim", type=int, default=128)
    parser.add_argument("--side-lr", type=float, default=1e-3)
    parser.add_argument("--model-bank-slots", type=int, default=0)
    parser.add_argument("--model-bank-rank", type=int, default=64)
    parser.add_argument("--model-bank-flush-interval", type=int, default=1)
    parser.add_argument("--model-bank-forward-backend", type=str, default="torch")
    parser.add_argument("--model-bank-train-cudagraph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-distiller-intervention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--distiller-beta", type=float, default=0.1)
    parser.add_argument("--distiller-sampler-backend", type=str, default="post_filter_exact")
    return parser.parse_args(argv)


def _build_parallel_requests(args: argparse.Namespace) -> tuple[list[str], list[SamplingParams], list[int], list[int]]:
    """Use explicit parallel requests because vLLM V1 may not emit all n>1 samples."""
    n = max(1, int(args.num_answers))
    prompts = [str(args.prompt)] * n
    params = [
        SamplingParams(
            n=1,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            min_p=float(args.min_p),
            max_tokens=int(args.max_new_tokens),
            seed=int(args.seed) + i,
        )
        for i in range(n)
    ]
    return prompts, params, [0] * n, list(range(n))


def _configure_esamp(args: argparse.Namespace):
    rows = int(args.graph_scratch_rows) if int(args.graph_scratch_rows) > 0 else max(64, int(args.num_answers))
    slots = int(args.model_bank_slots) if int(args.model_bank_slots) > 0 else int(args.num_answers)
    tap_layer_paths = [str(args.source_layer_path), str(args.target_layer_path)]
    return side_train_support.configure_esamp_runtime(
        graph_scratch_rows=rows,
        tap_layer_paths=tap_layer_paths,
        source_layer_path=str(args.source_layer_path),
        target_layer_path=str(args.target_layer_path),
        enable_side_train=True,
        side_hidden_dim=int(args.side_hidden_dim),
        side_lr=float(args.side_lr),
        per_request_models=False,
        per_request_model_bank=True,
        model_bank_slots=slots,
        model_bank_flush_interval=int(args.model_bank_flush_interval),
        model_bank_rank=int(args.model_bank_rank),
        model_bank_train_cudagraph=bool(args.model_bank_train_cudagraph),
        model_bank_forward_backend=str(args.model_bank_forward_backend),
        enable_distiller_intervention=bool(args.enable_distiller_intervention),
        distiller_beta=float(args.distiller_beta),
        distiller_sampler_backend=str(args.distiller_sampler_backend),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    consumer = _configure_esamp(args)
    prompts, params, prompt_indices, sample_indices = _build_parallel_requests(args)

    llm = make_llm(
        model_name=str(args.model_name),
        dtype=str(args.dtype),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        enable_prefix_caching=False,
        enforce_eager=False,
    )

    try:
        outputs = side_train_support.run_generate_with_request_mapping(
            llm,
            prompts,
            params,
            request_prompt_indices=prompt_indices,
            request_sample_indices=sample_indices,
        )
        runtime.synchronize_side_train()
        stats = runtime.read_and_reset_side_train_stats(sync=True)

        for i, out in enumerate(outputs):
            choices = getattr(out, "outputs", None) or []
            text = choices[0].text if choices else ""
            print(f"\n=== answer {i + 1}/{len(outputs)} ===")
            print(text.strip())

        print(
            "\nESamp stats: "
            f"loss_avg={float(stats.loss_avg):.6f} "
            f"loss_count={int(stats.loss_count)} "
            f"answers={len(outputs)} "
            f"distiller_enabled={bool(args.enable_distiller_intervention)} "
            f"distiller_beta={float(args.distiller_beta)}"
        )
        consumer.synchronize()
    finally:
        shutdown_llm_instance(llm, cooldown_s=0.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
