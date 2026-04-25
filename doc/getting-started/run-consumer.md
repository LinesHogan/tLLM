# Run a Built-in Consumer

This guide is for users who have installed tLLM and want to run a built-in consumer during generation.

After reading it, you should understand:

1. What a consumer is.
2. How a consumer is attached to generation.
3. How to run ESamp as a concrete example.

## What Is a Consumer?

A consumer is code that receives data captured from the LLM runtime. It can read hidden states, request metadata, logits, or other ports, then perform analysis, export, training, or guidance.

tLLM ships with several consumers. The most complete one is **ESamp**, a side-training consumer that trains a lightweight distiller from shallow hidden states to deeper hidden states, and can optionally use that distiller to guide sampling.

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

1. Loads `Qwen/Qwen3-1.7B`.
2. Configures ESamp.
3. Generates 16 answers in parallel.
4. Runs side-training during generation.
5. Prints `loss_count` and `loss_avg`.

For a shorter run:

```bash
python starter.py --max-new-tokens 32
```

`loss_count > 0` means side-training actually happened.

## Key Code Shape

`starter.py` uses `side_train_support.configure_esamp_runtime(...)` instead of manually instantiating `ESampConsumer` and registering it. That helper keeps ESamp configuration, runtime state, sampler provider setup, and request mapping in one place.

```python
from vllm import SamplingParams

from tllm.runtime import residual_runtime as runtime
from tllm.util.tools import make_llm
from tllm.workflows import side_train_support

consumer = side_train_support.configure_esamp_runtime(
    graph_scratch_rows=64,
    tap_layer_paths=[
        "model.model.layers[0].input_layernorm",
        "model.model.layers[-1].input_layernorm",
    ],
    source_layer_path="model.model.layers[0].input_layernorm",
    target_layer_path="model.model.layers[-1].input_layernorm",
    enable_side_train=True,
    side_hidden_dim=128,
    side_lr=1e-3,
    per_request_model_bank=True,
    model_bank_slots=16,
    model_bank_rank=64,
    model_bank_flush_interval=1,
    model_bank_train_cudagraph=True,
    enable_distiller_intervention=True,
    distiller_beta=0.1,
    distiller_sampler_backend="post_filter_exact",
)

llm = make_llm(
    model_name="Qwen/Qwen3-1.7B",
    dtype="bfloat16",
    gpu_memory_utilization=0.8,
    max_model_len=512,
    enable_prefix_caching=False,
    enforce_eager=False,
)

prompts = ["Introduce tLLM in two sentences."] * 16
params = [
    SamplingParams(n=1, temperature=0.8, top_p=0.95, max_tokens=32, seed=2026 + i)
    for i in range(16)
]

outputs = side_train_support.run_generate_with_request_mapping(
    llm,
    prompts,
    params,
    request_prompt_indices=[0] * 16,
    request_sample_indices=list(range(16)),
)

runtime.synchronize_side_train()
stats = runtime.read_and_reset_side_train_stats(sync=True)
print(stats)
```

The explicit 16-request construction is intentional. Some vLLM V1 versions do not emit every `n>1` sample consistently through the public output path, so the starter uses separate requests with `n=1`.

## Benchmark ESamp

Use the aligned benchmark when you want a meaningful throughput ratio:

```bash
VLLM_USE_FLASHINFER_SAMPLER=1 \
python -m tllm.workflows.benchmarks.per_request_side_train_benchmark \
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
  --side-lr 1e-3 \
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
| `loss_avg` | Average side-training loss | Should stay in a reasonable range |

## Next Steps

- [ESamp Design](../developer-guides/esamp-design.md)
- [ESamp Usage](../reference/esamp-usage.md)
- [Write Your First Consumer](write-your-first-consumer.md)
