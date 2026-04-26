# Run a Built-in Consumer

This guide is for users who have installed tLLM and want to run a built-in consumer during generation.

After reading it, you should understand:

1. What a consumer is.
2. How a consumer is attached to generation.
3. How to run ESamp as a concrete example.

## What Is a Consumer?

A consumer is code that receives data captured from the LLM runtime. It can read hidden states, request metadata, logits, or other ports, then perform analysis, export, training, or guidance.

tLLM ships with several consumers. The most complete one is **ESamp**, an
adaptive/guidance consumer that can train a lightweight distiller from shallow
hidden states to deeper hidden states, and can optionally use that distiller to
guide sampling.

## General Flow

The shape is the same for most consumers:

1. Create a consumer configuration.
2. Attach the consumer to the tLLM runtime.
3. Run generation.
4. Synchronize async work.
5. Read stats.

ESamp has a few extra pieces, because it also needs request mapping and sampler-guidance configuration.

## Minimal ESamp Example

```bash
python starter.py
```

This command:

1. Loads `Qwen/Qwen2.5-7B-Instruct`.
2. Configures ESamp.
3. Generates 16 answers in parallel.
4. Runs ESamp's training mechanism during generation.
5. Prints training and sampler-guidance counters such as `loss_count`, `loss_avg`, and `distiller_candidate_samples`.

For a shorter run:

```bash
python starter.py --max-new-tokens 32
```

`loss_count > 0` means ESamp's training mechanism actually ran.
`distiller_candidate_samples > 0` means the sampler-guidance path actually modified post-filter candidates.

By default, `starter.py` uses one shared seed for all explicit requests:

```bash
python starter.py --seed 2026 --seed-mode shared
```

This avoids vLLM's per-request generator path, so FlashInfer sampler acceleration can stay active when the rest of the environment supports it. For stricter per-answer reproducibility, use:

```bash
python starter.py --seed 2026 --seed-mode per-request
```

In shared mode, `seed` is passed to the LLM engine and request-level `SamplingParams.seed` is left unset. Per-request mode sets request seeds to `seed + i`. vLLM may then log `FlashInfer 0.2.3+ does not support per-request generators. Falling back to PyTorch-native implementation.` This warning is expected for that seed mode.

## Key Code Shape

The generic shape is explicit consumer registration. `starter.py` uses the ESamp workflow helper for a compact demo, but the conceptual API is still `ESampConsumer(...)` plus `register_consumer(...)`.

```python
from vllm import SamplingParams

from tllm import make_llm, register_consumer
from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig
from tllm.runtime import residual_runtime as runtime
from tllm.workflows import esamp_support

consumer = ESampConsumer(ESampConsumerConfig(
    graph_scratch_rows=64,
    source_layer_path="model.model.layers[0].input_layernorm",
    target_layer_path="model.model.layers[-1].input_layernorm",
    enable_esamp_training=True,
    distiller_hidden_dim=128,
    distiller_lr=1e-3,
    per_request_model_bank=True,
    model_bank_slots=16,
    model_bank_rank=64,
    model_bank_flush_interval=1,
    model_bank_train_cudagraph=True,
    enable_distiller_intervention=True,
    distiller_beta=0.1,
    distiller_sampler_backend="post_filter_exact",
))
register_consumer(consumer)

llm = make_llm(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.8,
    max_model_len=512,
    enable_prefix_caching=False,
    enforce_eager=False,
    seed=2026,
)

prompts = ["Introduce tLLM in two sentences."] * 16
params = [
    SamplingParams(n=1, temperature=0.8, top_p=0.95, max_tokens=32)
    for i in range(16)
]

outputs = esamp_support.run_generate_with_request_mapping(
    llm,
    prompts,
    params,
    request_prompt_indices=[0] * 16,
    request_sample_indices=list(range(16)),
)

runtime.synchronize_esamp()
stats = runtime.read_and_reset_esamp_stats(sync=True)
print(stats)
```

The explicit 16-request construction is intentional. Some vLLM V1 versions do not emit every `n>1` sample consistently through the public output path, so the starter uses separate requests with `n=1`. Keep a shared seed for the FlashInfer-friendly path, or use `--seed-mode per-request` when independent per-request generators matter more than sampler backend selection.

## Benchmark ESamp

Use the aligned benchmark when you want a meaningful throughput ratio:

```bash
VLLM_USE_FLASHINFER_SAMPLER=1 \
python -m tllm.workflows.benchmarks.per_request_esamp_benchmark \
  --emit-json-summary \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 512 \
  --benchmark-batch-size 8 \
  --benchmark-max-new-tokens 256 \
  --benchmark-warmup-rounds 1 \
  --benchmark-rounds 2 \
  --benchmark-ignore-eos \
  --benchmark-disable-prefix-caching \
  --sampling-n 16 \
  --sampling-temperature 0.8 \
  --sampling-top-p 0.95 \
  --sampling-top-k -1 \
  --distiller-lr 1e-3 \
  --model-bank-flush-interval 1 \
  --model-bank-init-method ffn_fast_svd \
  --trajectory-topk 1 \
  --model-bank-train-cudagraph \
  --run-model-bank-case
```

The key metric is:

```text
ratio = model_bank_on / single_off
```

| Metric | Meaning | How to read it |
|--------|---------|----------------|
| `single_off` | Vanilla vLLM baseline | First check that this number is reasonable |
| `model_bank_on` | Throughput with ESamp enabled | Compare it to `single_off` |
| `ratio` | Relative overhead | Depends on model size, sampler settings, intervention, and graph replay; optimized 7B min-p paths have reached the 95%+ target range |
| `loss_count` | Must be greater than zero | Zero means training did not run, regardless of throughput |
| `loss_avg` | Average training loss | Should stay in a reasonable range |

## Next Steps

- [ESamp Design](../developer-guides/esamp-design.md)
- [ESamp Usage](../reference/esamp-usage.md)
- [Consumer Delivery Modes](../developer-guides/consumer-delivery-modes.md)
- [Write Your First Consumer](write-your-first-consumer.md)
