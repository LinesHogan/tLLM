#!/usr/bin/env python3
"""Smoke repro for delayed side-train: run one case and assert loss is produced."""

from __future__ import annotations

import argparse

from tllm.runtime import residual_runtime as core
from tllm.workflows import esamp_support as esamp_workflow_support
from tllm.util.tools import build_prompt_batch, read_prompts, shutdown_llm_instance


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

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.set_defaults(ignore_eos=True)
    parser.set_defaults(enforce_eager=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.enforce_eager:
        print(
            "[warning] running side-train smoke with enforce_eager=False. "
            "Depending on model/backend this may fail under torch.compile."
        )

    prompts = read_prompts(args.prompt_file, args.prompt)
    bench_prompts = build_prompt_batch(prompts, int(args.batch_size))

    rows = int(args.graph_scratch_rows) if int(args.graph_scratch_rows) > 0 else max(64, len(bench_prompts))

    esamp_workflow_support.configure_esamp_runtime(
        graph_scratch_rows=rows,
        tap_layer_paths=[args.source_layer_path, args.target_layer_path],
        source_layer_path=args.source_layer_path,
        target_layer_path=args.target_layer_path,
        enable_esamp_training=True,
        distiller_hidden_dim=int(args.distiller_hidden_dim),
        distiller_lr=float(args.distiller_lr),
    )

    llm = core.make_llm(
        model_name=args.model_name,
        dtype=args.dtype,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        enable_prefix_caching=(not args.disable_prefix_caching),
        enforce_eager=bool(args.enforce_eager),
    )

    try:
        result = esamp_workflow_support.run_esamp_throughput_case(
            llm=llm,
            prompts=bench_prompts,
            max_new_tokens=int(args.max_new_tokens),
            warmup_rounds=int(args.warmup_rounds),
            rounds=int(args.rounds),
            train_enabled=True,
            ignore_eos=bool(args.ignore_eos),
            log_memory=False,
        )
    finally:
        shutdown_llm_instance(llm, cooldown_s=1.0)

    print(
        f"esamp smoke: req/s={result['req_per_s']:.3f} out_tok/s={result['out_tok_per_s']:.3f} "
        f"loss_avg={result['loss_avg']:.6e} loss_count={int(result['loss_count'])}"
    )

    if int(result["loss_count"]) <= 0:
        raise RuntimeError("No side-train loss was produced. Check decode localization or delayed backward path.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
