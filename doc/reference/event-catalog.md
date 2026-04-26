# Runtime Event Catalog

This page describes runtime hook order and event semantics. Consumer authors usually do not need to model their code around raw event names. Prefer `ConsumerFlow` and public ports.

## Event Overview

The runtime currently emits these internal events:

- `load_model.pre`
- `load_model.post`
- `prepare_inputs.pre`
- `prepare_inputs.post`
- `layer.pre`
- `layer.post`
- `block.end`
- `stack.end`
- `execute_model.pre`
- `execute_model.post`

`layer.post` may carry a `HiddenBatch`; most other events are lifecycle events without payload.

| Event | Trigger | Default phase | HiddenBatch? | Typical use |
|-------|---------|---------------|--------------|-------------|
| `load_model.pre` | Before `GPUModelRunner.load_model` | None | No | Preflight checks |
| `load_model.post` | After model load and hook setup | None | No | Lazy initialization using model metadata |
| `prepare_inputs.pre` | Before `_prepare_inputs` | `decode` | No | Step bookkeeping |
| `prepare_inputs.post` | After `_prepare_inputs` and localization | `decode` | No | Inspect row metadata |
| `layer.pre` | After tap-layer forward, before gather | `decode` | No | Profiling or pre-work |
| `layer.post` | After localized hidden rows are ready | `decode` | Yes | Main hidden-consumption entry |
| `block.end` | Immediately after `layer.post` | `decode` | No | Flush/checkpoint boundary |
| `stack.end` | When the target tap layer is reached | `decode` | No | Source/target aggregation |
| `execute_model.pre` | Before `execute_model` | `decode` | No | Step coordination |
| `execute_model.post` | After `execute_model` | `decode` | No | Async drain and feedback |

The main event path is decode-focused. Prefill export currently uses capture/export paths rather than the `HiddenBatch` event path.

## Key Events

### `load_model.post`

Use this after model structure and device information are available. Consumers can lazily allocate resources based on hidden size or layer paths.

### `prepare_inputs.post`

At this point, runtime has row-localization metadata but not hidden rows. Hidden rows arrive only after the tap-layer forward.

### `layer.post`

This is the main hidden-data event. `HiddenBatch` includes:

- `step_id`
- `phase`
- `layer_path`
- `rows_hidden`
- `row_idx`
- `valid_mask`
- `prompt_idx`
- `sample_idx`
- `metadata`

### `execute_model.post`

This is the step-end boundary. It is where async queues, feedback hooks, and delayed training work are commonly scheduled.

## Dispatch Order

For legacy event-style consumers, runtime may call:

1. `consume(batch, ctx)` when a `HiddenBatch` exists.
2. `on_tick(event_name, ctx)` for lifecycle events.
3. `on_step_end(ctx)` at `execute_model.post`.

Flow-based consumers should generally use `consume_bundle(bundle, ctx)` instead.

## Related Docs

- [Architecture](../developer-guides/architecture.md)
- [Glossary](glossary.md)
- [Project Structure](project-structure.md)
- [Write Your First Consumer](../getting-started/write-your-first-consumer.md)
