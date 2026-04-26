# FAQ

## Environment

### FlashInfer fails to compile. What should I do?

FlashInfer is an optional vLLM acceleration backend. Disable it first and verify the base path:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

Common causes include CUDA/PyTorch mismatch and an unsupported compiler. Prefer a prebuilt `flashinfer` package when one is available.

### Which vLLM version is required?

tLLM requires `vllm >= 0.7.2` and only supports the vLLM v1 engine.

The repo-local development environment currently uses `vllm==0.10.1.1`, so day-to-day validation should prioritize 0.10.x behavior.

### What do the vLLM environment variables mean?

| Variable | Meaning | Set automatically? |
|----------|---------|--------------------|
| `VLLM_USE_V1=1` | Use the vLLM v1 engine | Yes |
| `VLLM_ENABLE_V1_MULTIPROCESSING=0` | Keep runtime state in one process | Yes |
| `VLLM_USE_FLASHINFER_SAMPLER=1` | Enable FlashInfer sampler | No |
| `VLLM_DISABLE_COMPILE_CACHE` | Disable compile cache | Usually leave unset |

tLLM disables vLLM multiprocessing because Producer/Consumer state is stored in-process.

## Running and Debugging

### MSE validation failed. What now?

MSE failure means the gold path and batched path produced different hidden rows.

Check:

1. Model, dtype, max model length, and GPU memory settings.
2. Whether sampling randomness caused token divergence.
3. Whether a looser tolerance such as `1e-3` passes.
4. `test/test_decode_localization_unit.py` for pure localization logic.
5. Decode first, then prefill, to narrow the failing phase.

### How do I debug OOM?

Common cases:

- Model load OOM: reduce `--gpu-memory-utilization`.
- High `sampling_n`: increase `--max-model-len` or lower `n`.
- ESamp training OOM: lower `--distiller-hidden-dim` or `--model-bank-rank`.

### CUDA Graph capture errors?

Typical causes:

1. Non-graph-safe logic inside layer hooks. Keep hooks light; use them for capture and staging.
2. `VLLM_DISABLE_COMPILE_CACHE=1` on older vLLM setups. Run `unset VLLM_DISABLE_COMPILE_CACHE`.
3. Very high sampling pressure causing sampler CUDA asserts. Increase `--max-model-len`.

Debug with eager mode when needed, but production/benchmark paths should try to preserve vLLM graph/compile optimizations. ESamp's `--model-bank-train-cudagraph` is separate from vLLM's main inference graph.

## Architecture

### Why monkey-patch vLLM?

vLLM does not expose all hook points tLLM needs. tLLM installs patches at a few key boundaries:

- `load_model`: install layer hooks.
- `_prepare_inputs`: snapshot request and row-localization metadata.
- `execute_model`: dispatch bundles and async consumer work.

It also uses layer hooks, `compute_logits`, and sampler patches for hidden capture and sampler intervention. The three `GPUModelRunner` methods are the main lifecycle boundaries, not the only hook points.

### What is the relationship between `ConsumerFlow` and ports?

`ConsumerFlow` is how a consumer declares:

- Which ports it reads.
- Which ports it writes.
- Which execution window it uses.
- How frames are grouped into bundles.

Consumers should prefer this public surface over raw runtime events.

### What is DummyConsumer for?

DummyConsumer is an async hidden-read/export demo, not a production algorithm. It shows how to read `residual_stream + request_meta`, copy hidden rows to CPU without blocking, and drain work safely.

## ESamp Training Mechanism

### Does ESamp training require `enforce_eager`?

No. `--enforce-eager` disables vLLM CUDA Graph and torch.compile paths, which is useful for debugging. It should not be treated as a requirement for ESamp training.

The preferred production shape is:

- Keep layer hooks lightweight.
- Schedule distiller prediction at `compute_logits`.
- Run ESamp distiller updates in the `out_of_band` window on a side stream.
- Preserve vLLM graph/compile optimizations where possible.

### When should I use model-bank mode?

Use model-bank when many active requests each need ESamp training state. It assigns requests to fixed slots and batches training work, reducing launch overhead and enabling CUDA Graph replay for the ESamp training path.

Use `single` for quick checks. Use `per-request` mainly for debugging or small fixed request counts.
