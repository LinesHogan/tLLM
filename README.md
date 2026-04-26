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
+ from tllm import make_llm, register_consumer
+ from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig

+ register_consumer(ESampConsumer(ESampConsumerConfig()))
+ llm = make_llm(model_name="Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16")
  outputs = llm.generate(
      [f"Suprise me an unexpectedly story about {i} evil sorcerers and the brave hero." for i in range(2, 16)],
      SamplingParams(max_tokens=64, temperature=0.8, n=8),
  )
```

`make_llm` installs tLLM's vLLM v1 runtime hooks; `register_consumer(...)` attaches the consumer that actually uses those hooks. ESamp workflow helpers still exist for benchmarks and one-line demos.

In the aligned 7B min-p ESamp benchmark, with RTX 4090 GPU, the optimized ESamp has measured about **98.8% of a vLLM baseline with modern inference optimizations enabled**. That baseline uses the vLLM V1 engine with CUDA Graph execution, FlashInfer sampling, bfloat16 weights, prefix-cache control for fair measurement, and the same sampling workload.

Representative run:

| Model/workload | Optimized vLLM baseline | ESamp (triton kernel) | Ratio |
|----------------|-----------------------------------------|------------------|-------|
| Qwen2.5-7B, batch=8, n=16, min-p active path | 5370.616 tok/s | 5304.855 tok/s | 0.9878 |


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

The starter runs ESamp with `Qwen/Qwen2.5-7B-Instruct`, generates 16 answers in parallel, and prints runtime adaptation statistics.
It defaults to `--seed-mode shared`, which avoids vLLM's per-request generator path and keeps FlashInfer sampler acceleration available when supported. Use `--seed-mode per-request` if independent `seed + i` request streams matter more than that sampler backend.

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

## FAQ

> What is the name "tLLM" for?

tLLM is a test-time intervention layer for vLLM. The name is a recursive-style abbreviation: **tLLM is a test-time intervention layer for vLLM**.

> I have a test time training algorithm. How can I implement it with tLLM?

You can contact us, and we will offer advice and technical support to help you achieve efficient implementation under the tLLM framework, enabling your algorithm to reach production-level throughput!
