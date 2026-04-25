# Producer/Consumer Contract

This document describes the data contract between Producer and Consumer.

The Producer locates rows inside vLLM's packed tensors. The Consumer receives those rows and performs analysis, training, export, or feedback. Runtime connects the two by installing hooks, maintaining localization metadata, and assembling bundles.

## Producer Output

Producer extracts:

- `hidden`: selected rows from a captured layer.
- metadata:
  - `phase`
  - `prompt_idx`
  - `sample_idx`
  - optional token offsets for prefill.

Capture storage currently separates:

- decode: `captured_decode[prompt_idx] -> List[Tensor]`
- prefill: `captured_prefill[prompt_idx] -> List[Tensor]`

## Consumer Input

Modern consumers receive `PortBundle` objects through:

```python
consume_bundle(bundle, ctx)
```

Typical entries include:

- localized hidden rows, shaped `[rows, hidden_size]`
- request metadata
- optional sampler or export data

Invalid padded rows are masked out through runtime-managed masks.

## Decode Localization

Inputs:

- request ids
- decode request flags
- `logits_indices`
- actual token counts

High-level steps:

1. Select active decode requests.
2. Read row indices from `logits_indices`.
3. Write them into a fixed GPU buffer.
4. Mark valid rows.
5. Gather hidden rows in the tap-layer hook.

Fixed buffers keep the decode path compatible with CUDA Graph replay.

## Prefill Localization

For each request:

```text
prefill_len = clamp(prompt_len - computed, 0, scheduled)
```

If the request occupies `[row_base, row_base + scheduled)`, then its prefill rows are `[row_base, row_base + prefill_len)`.

Prefill currently uses an eager-first path.

## Validation

Decode correctness:

```bash
python -m verify_v1_decode_rows_minimal \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "hello" \
  --max-new-tokens 8 \
  --mse-tol 1e-4
```

Prefill correctness:

```bash
python -m tllm.workflows.repro.repro_prefill_sampling_mse \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --prompt-file test/prompt_debug_list.txt \
  --gen-max-new-tokens 4 \
  --sampling-n 3 \
  --mse-tol 1e-5 \
  --gpu-memory-utilization 0.3 \
  --max-model-len 256
```

Automated verification scenarios:

```bash
python -m tllm.verification.automated_tests --list
python -m tllm.verification.automated_tests --scenario esamp_loss_parity_qwen2p5_0p5b
```

## Related Docs

- [Validation](../developer-guides/validation.md)
- [Port Catalog](port-catalog.md)
- [Architecture](../developer-guides/architecture.md)
