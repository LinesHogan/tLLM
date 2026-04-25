# tLLM

tLLM is a runtime layer for building producer/consumer extensions on top of the vLLM v1 inference engine.

It lets you capture model-internal data such as hidden states, route that data through public ports, and run custom consumers during generation without maintaining a fork of vLLM.

## Why tLLM

Most research ideas around runtime adaptation start out in a friendly stack, then become painful when moved into a high-throughput inference engine. tLLM is designed for the migration step: keep vLLM's serving performance, but add a stable producer/consumer surface for hidden-state capture, async side work, and sampler guidance.

| Task | Direct vLLM modification | With tLLM |
|------|--------------------------|-----------|
| Read hidden states during decode | Patch runner internals and keep up with vLLM changes | Declare a `ConsumerFlow` over public ports |
| Add async CPU/GPU side work | Own stream/event timing by hand | Use consumer `synchronize()` and runtime-managed bundles |
| Add sampler guidance | Patch logits/sampler code directly | Implement a sampler provider behind tLLM's bridge |
| Keep throughput measurable | Easy to accidentally benchmark a broken no-op | Standard ratio checks plus functional counters |
| Make you algorithm fast | Hard to understand how vLLM works and introduce latency | Clear ports / contracts and show be easy to write a fast pipeline |

### From vLLM Generation to tLLM Algorithms

If you already have a vLLM generation script, tLLM lets you keep the same prompts and `SamplingParams` while adding a consumer-provided algorithm before `generate`.

For example, ESamp is the algorithm proposed in *Large Language Models Explore by Latent Distilling*: it captures shallow and deep hidden states, trains a distiller during generation, and can use that distiller to modify candidate-token logits after top-k/top-p/min-p filtering.

The migration is only a few lines:

```diff
- from vllm import LLM, SamplingParams
+ from vllm import SamplingParams
+ from tllm.util.tools import make_llm
+ from tllm.workflows.esamp_support import configure_esamp_runtime

+ configure_esamp_runtime()
+ llm = make_llm(model_name="Qwen/Qwen3-1.7B", dtype="bfloat16")
  outputs = llm.generate(
      [f"Suprise me an unexpectedly story about {i} evil sorcerers and the brave hero." for i in range(2, 16)],
      SamplingParams(max_tokens=64, temperature=0.8, n=8),
  )
```

`make_llm` installs tLLM's vLLM v1 runtime hooks; `configure_esamp_runtime()` registers the ESamp consumer that actually uses those hooks. 

In the aligned 7B min-p ESamp benchmark, with RTX 4090 GPU, the optimized ESamp has measured about **96% of a vLLM baseline with modern inference optimizations enabled**. That baseline uses the vLLM V1 engine with CUDA Graph execution, FlashInfer sampling, bfloat16 weights, prefix-cache control for fair measurement, and the same sampling workload. 

Representative run:

| Model/workload | Optimized vLLM baseline | ESamp | Ratio |
|----------------|-----------------------------------------|------------------|-------|
| Qwen2.5-7B, batch=8, n=16, min-p active path | 4800.995 tok/s | 4611.270 tok/s | 0.9605 |

If you're happy, you can enable triton kernel for ESamp to have around 1-2% more throughput. But we don't guarantee this works on different model or some unusual settings.

## What You Can Build

- Activation or hidden-state export/editing pipelines.
- LLM Runtime analysis.
- Test time training algorithms that learn during generation.
- Candidate-level sampler guidance, such as ESamp distiller intervention.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install vllm
pip install -e .

python starter.py --max-new-tokens 32
```

The starter runs ESamp with `Qwen/Qwen3-1.7B`, generates 16 answers in parallel, and prints side-training statistics.

## Documentation

- English docs: [doc/README.md](doc/README.md)
- Chinese docs: [doc_zh/README.md](doc_zh/README.md)
- Write a consumer: [doc/getting-started/write-your-first-consumer.md](doc/getting-started/write-your-first-consumer.md)
- ESamp usage: [doc/reference/esamp-usage.md](doc/reference/esamp-usage.md)

## Requirements

- Python >= 3.10
- vLLM v1 engine
- PyTorch with CUDA

The current development environment is validated primarily with `vllm==0.10.x`.
