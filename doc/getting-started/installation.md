# Installation

This guide walks through a clean local installation of tLLM.

## 1. Clone the Repository

```bash
git clone <your-repo-url>
cd tLLM
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

tLLM expects Python >= 3.10.

## 3. Install vLLM

```bash
pip install vllm
```

tLLM uses vLLM as the underlying inference engine.

Version notes:

- Minimum supported version: `vllm >= 0.7.2`
- Only the vLLM v1 engine is supported.
- The main development and verification target is currently `vllm==0.10.x`.

Check the installed version:

```bash
python -c "import vllm; print(vllm.__version__)"
```

## 4. Install tLLM

```bash
pip install -e .
```

Editable install is recommended during development because code changes take effect immediately.

## 5. Check CUDA

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

If CUDA is not available, first check that your PyTorch installation matches your CUDA runtime.

## 6. Verify the Installation

For most users, the simplest verification is the ESamp starter:

```bash
python starter.py --max-new-tokens 32
```

This loads `Qwen/Qwen2.5-7B-Instruct`, generates 16 answers in parallel, and runs
ESamp's training mechanism alongside generation. A healthy run ends with
statistics similar to:

```text
ESamp stats: loss_avg=... loss_count=... answers=16 ...
```

`loss_count > 0` means the consumer received hidden states and ESamp's training
mechanism actually ran.

If you see `ModuleNotFoundError`, check that the virtual environment is active. If you hit OOM, reduce `--gpu-memory-utilization` or temporarily switch to `Qwen/Qwen2.5-0.5B-Instruct`.

Developers who need to verify hidden-row localization numerically can run `verify_v1_decode_rows_minimal.py`. That is a framework correctness check, not a required installation step.

## Environment Variables

These variables are set automatically by the tLLM runtime entrypoints:

| Variable | Value | Purpose |
|----------|-------|---------|
| `VLLM_USE_V1` | `1` | Force the vLLM v1 engine |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `0` | Disable multiprocessing so runtime state stays in-process |

For throughput experiments, FlashInfer is usually worth enabling:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=1
```

If FlashInfer is unavailable or fails to compile, disable it while checking basic functionality:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

## Common Issues

**FlashInfer fails to compile.**

FlashInfer is an optional vLLM acceleration backend. If it fails, disable it first and verify the base path. Common causes include CUDA/PyTorch mismatch and an unsupported compiler. Prefer a prebuilt `flashinfer` package when available.

**`VLLM_DISABLE_COMPILE_CACHE=1` causes `FileNotFoundError`.**

This is a known issue on some older vLLM setups. Run:

```bash
unset VLLM_DISABLE_COMPILE_CACHE
```

## Next Steps

- [Run a Consumer](run-consumer.md)
- [Write Your First Consumer](write-your-first-consumer.md)
