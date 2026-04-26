# Architecture

This guide explains how data leaves vLLM, passes through tLLM, and reaches a consumer.

If you only want to write a simple consumer, you do not need to memorize every detail. Read [Write Your First Consumer](../getting-started/write-your-first-consumer.md) first, then come back here when a concept becomes relevant.

tLLM separates two concerns:

- **Producer**: find and extract the correct rows from vLLM's packed tensors.
- **Consumer**: decide what to do with those rows.
- **Runtime**: install hooks, assemble bundles, and schedule consumer work.

## The Problem

Suppose you want to capture hidden states during vLLM generation and optionally feed something back into the generation process. Three things make this hard.

### 1. vLLM Packs Many Requests Into One Tensor

With multiple active requests, vLLM uses tensors shaped like `[total_tokens, hidden_size]`. You need to know which row belongs to which request and phase. tLLM calls this **localization**.

### 2. Useful Hook Points Are Version-Sensitive

`_prepare_inputs`, `execute_model`, and model-layer forward calls can change across vLLM versions. Putting algorithm code directly into those places creates a brittle fork.

### 3. Training Must Not Block Inference

Running ESamp distiller backward directly inside model forward would block the main inference stream. The work needs to be asynchronous, but still synchronized correctly with the data it consumes.

## tLLM's Split

tLLM keeps the boundaries narrow:

- Runtime handles capture, localization, and bundle assembly.
- Consumer code receives `PortBundle` objects.
- Algorithm internals stay inside the consumer.

ESamp follows this split:

- `ESampConsumer` reads bundles and owns consumer-facing state.
- `ESampTrainEngine` owns the training pipeline and model-bank mechanics.
- Runtime does not know ESamp's training details.

## Port: The Public Data Interface

Consumers do not manipulate vLLM internals directly. They declare ports:

| Port | Meaning | Example |
|------|---------|---------|
| `residual_stream` | Hidden states at a layer/site | `ResidualStream.read(layer=0, site="block_output")` |
| `request_meta` | Request identity metadata | `RequestMeta.read()` |
| `cpu_export` | Asynchronous CPU export | `CpuExport.write(channel="debug")` |
| `logits` | Pre-sampling logits | `Logits.read()` |
| `kv_cache` | KV cache state | `KVCache.read(layer=12)` |

Example flow:

```python
ConsumerFlow(
    reads=(
        ResidualStream.read(layer=0, site="block_output", phase="decode"),
        RequestMeta.read(),
    ),
    writes=(CpuExport.write(channel="debug"),),
    window="background",
)
```

The runtime installs hooks, gathers rows, groups frames by `bundle_key`, and calls `consume_bundle(bundle, ctx)`.

## A Decode Step

From a consumer's point of view, one decode step looks like this:

1. `_prepare_inputs` runs.
2. Runtime snapshots request order and decode localization metadata.
3. vLLM runs model forward.
4. Runtime layer hooks gather the requested hidden rows into fixed buffers.
5. `execute_model` returns.
6. Runtime assembles `PortBundle` objects and dispatches consumers.
7. Step-end feedback or async training is scheduled if the consumer needs it.

For sampler-guidance consumers, tLLM also uses the `compute_logits` and sampler boundaries so prediction can happen before candidate logits are modified.

## Localization

### Decode Localization

During decode, each active request contributes one row: the token being generated now.

The producer:

1. Reads `logits_indices`.
2. Selects requests marked as decode.
3. Writes row indices into a fixed GPU buffer.
4. Uses graph-safe gather in the layer hook.
5. Applies a valid mask for active rows.

Fixed buffers matter because CUDA Graph replay cannot allocate new tensors every step.

### Prefill Localization

During prefill, each request may occupy a contiguous span of rows.

The producer computes:

```text
prefill_len = min(scheduled, prompt_len - computed)
```

Then it records the `[start, start + prefill_len)` range for each request. Prefill currently uses an eager-first path, separate from decode's graph-safe path.

## Writing a Consumer

The simplest consumer implements:

```python
class MyConsumer(BaseConsumer):
    @property
    def consumer_id(self):
        return "my_consumer"

    def flows(self):
        return [
            ConsumerFlow(
                reads=(ResidualStream.read(layer=0, site="block_output", phase="decode"),),
                window="background",
            )
        ]

    def consume_bundle(self, bundle, ctx):
        hidden = bundle.entries["hidden"]
        # Your logic here.
```

If you need step-end work, implement `on_step_end(ctx)` as an optional compatibility hook.

## Related Docs

- [Glossary](../reference/glossary.md)
- [Project Structure](../reference/project-structure.md)
- [Write Your First Consumer](../getting-started/write-your-first-consumer.md)
