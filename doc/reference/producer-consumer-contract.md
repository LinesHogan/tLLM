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

Current decode capture uses runtime tap buffers keyed by resolved layer path:

- `tap_decode_hidden[resolved_path] -> Tensor[rows, hidden_size]`

The hook writes localized decode rows into these fixed buffers. Older producer
helpers may still use prompt-indexed storage for repro or prefill workflows, but
modern consumer delivery should be understood through ports and bundles rather
than through that internal storage shape.

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

`ConsumerFlow` delivery metadata defaults to `delivery="bundle"` and
`ownership="borrowed"`, which is the standard bundle dispatch path.
Consumers can opt into `delivery="device_lease"` with
`ownership="runtime_lease"` when they are prepared to consume runtime-leased
device tensors directly. ESamp is one example of an adaptive/guidance
consumer that may use this opt-in path; it should not be treated as a synonym
for ESamp's training mechanism.

In the current implementation, a device tensor lease describes runtime-owned
tensor entries and active row count. The entries are valid for the
`consume_bundle()` call and must be treated as read-only. The lease advertises
this as `lifetime="consume_call"`. It does not yet carry ready events or
durable-buffer lifetime guarantees; consumers that need to retain data beyond
the call must copy it into their own buffers.

Current `device_lease` delivery is limited to decode step bundles with
`bundle_key=("engine_step_id", "phase")`; it supports `residual_stream` reads
and optional `request_meta`. Broader port coverage and durable staged-buffer
leases should be added as new contract revisions rather than inferred from this
first implementation.

Flows may also declare row shaping with `row_compaction`. The default is
`row_compaction="none"`, which preserves decode-row order and cardinality.
`row_compaction="first_per_prompt"` asks runtime to deliver only the first row
for each prompt in the current decode step. When request metadata is delivered
as `RowBatchMeta`, `row_ids` records the original decode-row positions.
Metadata cardinality matches the delivered live rows. This is a generic
delivery contract for per-prompt GPU consumers; ESamp model-bank uses it, but
runtime does not special-case ESamp.

For a teaching-oriented comparison of the two modes, read
[Consumer Delivery Modes](../developer-guides/consumer-delivery-modes.md).

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
