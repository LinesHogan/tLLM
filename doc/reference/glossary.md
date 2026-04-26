# Glossary

This is a quick reference. If you are new to tLLM, start with [Architecture](../developer-guides/architecture.md).

## Consumer

The code that receives `PortBundle` objects and performs analysis, export, training, or feedback.

Core methods:

- `consumer_id`
- `flows()`
- `consume_bundle(bundle, ctx)`
- `synchronize()`

## ConsumerFlow

A declarative description of what a consumer needs:

- `reads`: ports to read.
- `writes`: ports to write.
- `window`: when the flow runs.
- `bundle_key`: how runtime groups frames into a bundle.

## Decode Localization

The process of finding the row for each active decode request inside vLLM's packed hidden tensor. Decode contributes one row per active request.

## HiddenBatch

A legacy/event-path unit of hidden data. Important fields include `step_id`, `phase`, `layer_path`, `rows_hidden`, `row_idx`, `valid_mask`, `prompt_idx`, and `sample_idx`.

## Localization

The general process of mapping packed tensor rows back to prompt, sample, and phase. It includes decode and prefill localization.

## Model Bank

An ESamp parameter layout where active requests are assigned to fixed slots. It reduces launch overhead and supports CUDA Graph acceleration for ESamp's distiller update path.

## Phase

The generation stage:

- `prefill`: prompt processing.
- `decode`: token-by-token generation.

## Port

The public runtime data interface. Consumers use ports instead of vLLM internal objects.

| Port | Meaning | Direction |
|------|---------|-----------|
| `residual_stream` | Layer hidden states | Read/write |
| `request_meta` | Request identity metadata | Read-only |
| `cpu_export` | Async CPU export | Write-only |
| `logits` | Pre-sampling logits | Read-only |
| `kv_cache` | KV cache state | Read/write |
| `token_target` | Training or guidance target token | Read-only |
| `sampler` | Candidate sampler view | Read/provider modifier |

## PortBundle

The data package delivered to `consume_bundle(bundle, ctx)`. It contains a key and a dictionary of entries requested by the consumer flow.

## Prefill Localization

The process of finding each request's span of prompt rows inside a packed prefill tensor.

## Producer

The component that locates and extracts the correct hidden rows. It does not decide how those rows are used.

## Prompt Index / Sample Index

`prompt_idx` identifies the input prompt. `sample_idx` identifies the sample within a multi-sample request.

## Request ID

vLLM's request identifier. tLLM parses it to recover prompt/sample mapping for parallel sampling.

## Residual Stream

The hidden-state stream between model blocks. Exposed through the `residual_stream` port. Supported sites include `block_input`, `attn_input`, `attn_output`, `mlp_input`, and `block_output`.

## RuntimeContext

Context passed to consumers. It can include `runner`, `model`, `device`, `main_stream`, `is_compiling`, `uses_cudagraph`, and `event_name`.

## Runtime Adaptation

A pattern where captured runtime data is used to adapt auxiliary state during generation, often on a side CUDA stream. ESamp's distiller training mechanism is one example.

## Source Layer / Target Layer

In ESamp, the source layer provides input hidden states; the target layer provides supervision.

## Tap Layer

The model layer where runtime installs a forward hook.

## Window

The declared timing of a `ConsumerFlow`:

- `background`: async background work.
- `same_step`: must affect the current step.
- `next_step`: result affects a later step.
- `out_of_band`: step-end async side-stream work.

## Related Docs

- [Architecture](../developer-guides/architecture.md)
- [Port Catalog](port-catalog.md)
- [Write Your First Consumer](../getting-started/write-your-first-consumer.md)
