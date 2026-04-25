# tLLM

tLLM is a runtime layer for building producer/consumer extensions on top of the vLLM v1 inference engine.

It lets you capture model-internal data such as hidden states, route that data through public ports, and run custom consumers during generation without maintaining a fork of vLLM.

## What You Can Build

- Activation or hidden-state export pipelines.
- Runtime analysis consumers.
- Side-training algorithms that learn during generation.
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
