# Port Catalog

This page lists tLLM's public data interfaces.

## What Is a Port?

A port is how a consumer declares the data it wants from runtime. Consumers do not directly read vLLM internal objects. They declare ports through `ConsumerFlow`, and runtime captures the data and delivers a `PortBundle`.

| Port | Direction | Phases | Source |
|------|-----------|--------|--------|
| `request_meta` | Read-only | prefill, decode | Request mapping and identity |
| `residual_stream` | Read/write | prefill, decode | Hidden states at tap layers |
| `cpu_export` | Write-only | prefill, decode | Async CPU sink |
| `logits` | Read-only | decode | Pre-sampling logits |
| `kv_cache` | Read/write | decode | Layer-local KV cache |
| `token_target` | Read-only | prefill, decode | Training or guidance target token |
| `sampler` | Read/provider modifier | decode | Sampler logits, metadata, candidates |

## `request_meta`

Request identity metadata: request id, prompt index, sample index, phase, and step-related fields.

Use it to pair source/target rows, label CPU exports, or maintain per-request state.

## `residual_stream`

Hidden states at a logical layer/site/phase locator.

Supported sites:

- `block_input`
- `attn_input`
- `attn_output`
- `mlp_input`
- `block_output`

Typical uses: ESamp, activation export, residual write-back experiments.

## `cpu_export`

A write-only port for asynchronous CPU export. It keeps file formats, databases, or external sinks out of the public runtime contract.

Locator fields:

- `channel`
- `format`
- optional `schema`

## `logits`

Pre-sampling logits for decode. Useful for classifier guidance or logits analysis.

## `kv_cache`

Layer-local KV cache state during decode. Useful for next-step guidance or KV debugging.

## `token_target`

Token-level targets for training or guidance.

## `sampler`

The sampler port exposes the decode-step sampler view and modifier provider contract.

Runtime responsibilities:

- Install sampler patch.
- Build generic sampler views.
- Call candidate modifier providers.

Consumer responsibilities:

- Implement algorithm-specific providers.
- Return candidate-level logits deltas with the expected shape.

ESamp's provider lives in `tllm/consumers/esamp/sampler_provider.py`, but the runtime layer does not depend on ESamp.
