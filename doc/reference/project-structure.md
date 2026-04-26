# Project Structure

This page answers two practical questions:

1. How is the repository organized?
2. Where should you look before changing a behavior?

## Top-Level Directories

| Directory | Purpose |
|-----------|---------|
| `doc/` | English user/developer documentation |
| `doc_zh/` | Chinese documentation |
| `test/` | Pytest unit tests |
| `tllm/` | Core implementation |
| `outputs/` | Runtime outputs, if produced locally |

## `tllm/` Layers

### `tllm/common/`

Shared state and helpers.

- `state.py`: global state and layer resolution.
- `runtime_step_state.py`: step metadata from `_prepare_inputs`.
- `path_resolution.py`: model-path compatibility helpers.

Look here when hidden alignment or layer path resolution is wrong.

### `tllm/contracts/`

Shared data structures between runtime and consumers.

- `hidden_batch.py`
- `runtime_context.py`
- `port_bundle.py`

Look here when writing a consumer and checking what data structures contain.

### `tllm/ports/`

Public port declarations.

- `base.py`: `PortKind`, `ConsumerFlow`, `Window`.
- `residual_stream.py`: residual hidden locators.
- `request_meta.py`: request metadata port.
- `cpu_export.py`: CPU export port.

Look here when declaring a new flow or adding a new public port.

### `tllm/producer/`

Packed-tensor localization algorithms.

- `decode.py`: decode localization and graph-safe gather.
- `prefill.py`: prefill localization.

Look here when MSE validation or row alignment fails.

### `tllm/consumers/`

Consumer implementations.

- `base.py`: common consumer interface.
- `dummy/`: minimal teaching consumer.
- `esamp/`: built-in adaptive/guidance consumer.

ESamp key files:

- `consumer.py`: `ESampConsumer`, the framework-facing wrapper.
- `engine.py`: `ESampTrainEngine`, training state, and model-bank scheduling.
- `model.py`: distiller model definitions used by ESamp's training mechanism.
- `template.py`: model-bank initialization helpers.

### `tllm/runtime/`

The bridge between vLLM and tLLM.

- `vllm_patch/port_runtime_hooks.py`: vLLM hook lifecycle.
- `vllm_patch/common_hooks.py`: context construction and event dispatch helpers.
- `vllm_patch/sampler_patch.py`: sampler bridge and distiller precompute scheduling.
- `vllm_patch/adapters/`: version-specific `_prepare_inputs` normalization.
- `dispatch_plan.py`: static routing from flows to runtime targets.
- `ports/`: internal frame and bundle assembly logic.
- `residual_runtime.py`: generic runtime host.

Consumer authors usually do not need to change this layer.

### `tllm/workflows/`

Manual benchmark, repro, and experiment entrypoints.

### `tllm/verification/`

GPU validation harnesses with pass/fail semantics.

## Where To Start

### Decode hidden is wrong

1. `tllm/producer/decode.py`
2. `tllm/common/runtime_step_state.py`
3. `test/test_decode_localization_unit.py`

### You want to write a new consumer

1. `tllm/consumers/base.py`
2. `tllm/consumers/dummy/consumer.py`
3. `tllm/ports/base.py`
4. `tllm/ports/residual_stream.py`

### Consumer receives no bundle

1. Check `flows()`.
2. Read `tllm/runtime/dispatch_plan.py`.
3. Read `tllm/runtime/ports/residual_bundle_dispatch.py`.

## Related Docs

- [Architecture](../developer-guides/architecture.md)
- [Glossary](glossary.md)
