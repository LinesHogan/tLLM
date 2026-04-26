# tLLM

tLLM is a runtime layer for inserting data capture and consumer logic into the vLLM v1 inference engine. It lets external code read, analyze, and in some cases modify model-internal state while generation is running, without maintaining a fork of vLLM.

---

## Why This Exists

Modern inference engines such as vLLM are built for throughput: continuous batching, PagedAttention, CUDA Graphs, and fused CUDA kernels. That performance comes with a cost. If you want to read hidden states during generation, train a small model alongside the LLM, or guide sampling with runtime signals, the useful hook points are deep inside the engine and strongly tied to vLLM internals.

HuggingFace Transformers is much easier to experiment with, but it is not the runtime you usually want for high-throughput serving.

tLLM sits between these two worlds. It keeps vLLM as the inference engine, but exposes a declarative consumer interface for data capture, asynchronous processing, and sampler guidance.

---

## What tLLM Does

tLLM installs runtime hooks around key vLLM v1 lifecycle points:

- `load_model`: install layer hooks after the model is loaded.
- `_prepare_inputs`: snapshot request and row-localization metadata at the start of each step.
- `execute_model`: dispatch captured data and run consumer-side step-end work.

For sampler-guidance workloads, tLLM also patches `compute_logits` and the sampler path. This lets a consumer such as ESamp schedule distiller prediction before sampling, then modify candidate logits through a generic sampler bridge.

The data path is:

```text
vLLM tensors -> Producer localization -> Runtime bundle assembly -> Consumer
```

Consumers declare their needs with `ConsumerFlow`. The runtime handles hooks, row localization, buffer management, and `PortBundle` delivery. A consumer does not need to reach into vLLM internals directly.

---

## Which Reader Are You?

### 1. You Want To Run a Built-in Consumer

Start here if you want to try ESamp or use tLLM for generation.

1. [Installation](getting-started/installation.md)
2. [Run a Consumer](getting-started/run-consumer.md)
3. [ESamp Usage](reference/esamp-usage.md)

### 2. You Want To Build Your Own Consumer

Start here if you want to read hidden states, export activations, build a runtime adaptation method, or implement a new sampler-guidance algorithm.

1. [Installation](getting-started/installation.md)
2. [Architecture](developer-guides/architecture.md)
3. [Write Your First Consumer](getting-started/write-your-first-consumer.md)
4. [Consumer Delivery Modes](developer-guides/consumer-delivery-modes.md)
5. [Validation](developer-guides/validation.md)
6. [Debugging Consumers](developer-guides/debugging.md)

### 3. You Want To Change tLLM Itself

Start here if you want to add ports, change runtime hooks, or contribute to the framework.

1. [Project Structure](reference/project-structure.md)
2. [Developer Testing Guide](development/testing-guide.md)
3. [Glossary](reference/glossary.md)

---

## Quick Start

```bash
git clone <repo-url>
cd tLLM
python -m venv .venv
source .venv/bin/activate
pip install vllm
pip install -e .

# Run the ESamp starter.
python starter.py --max-new-tokens 32
```

If the environment is healthy, you should see 16 generated answers and ESamp statistics such as `loss_count`, `loss_avg`, and sampler-guidance counters. The starter defaults to `--seed-mode shared`; use `--seed-mode per-request` when independent request seeds are more important than keeping FlashInfer sampler paths available.

---

## Requirements

- Python >= 3.10
- vLLM >= 0.7.2, v1 engine only
- PyTorch >= 2.0
- CUDA >= 12.1

The currently validated development environment uses `vllm==0.10.x`.
