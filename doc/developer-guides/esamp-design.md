# Case Study: ESamp Consumer Design

This is a case study, not required reading.

If you only want to run ESamp, read [Run a Consumer](../getting-started/run-consumer.md). This page is for developers who want to understand how a complex consumer can be built from tLLM's generic mechanisms.

## What ESamp Does

ESamp is a runtime adaptation and sampler-guidance consumer. At each decode step, it wants to:

1. Read a shallow hidden state.
2. Read a deeper hidden state as a supervision target.
3. Adapt a lightweight distiller from the former to the latter.
4. Optionally use the distiller prediction to guide token sampling.

That involves capture, asynchronous training, stateful per-request/model-bank management, and sampler intervention.

## Data Capture: Declarative Ports

ESamp declares its data needs in `flows()`:

```python
ConsumerFlow(
    reads=(
        ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
        ResidualStream.read(layer=-1, site="block_output", phase="decode", role="target"),
        RequestMeta.read(),
    ),
    window="out_of_band",
    delivery="device_lease",
    ownership="runtime_lease",
    bundle_key=("engine_step_id", "phase"),
)
```

Runtime installs hooks, localizes packed rows, and assembles bundles. In the
default bundle path a consumer reads direct entries such as
`bundle.entries["source"]`. ESamp uses the device-lease path, so runtime
normally delivers those tensors through `bundle.entries["device_lease"]`.

Most consumers should stop at the ordinary bundle contract. ESamp opts into the
advanced delivery contract because its tensors stay on GPU and feed a step-level
engine:

```python
ConsumerFlow(
    ...,
    window="out_of_band",
    delivery="device_lease",
    ownership="runtime_lease",
    row_compaction="first_per_prompt",  # model-bank path
    bundle_key=("engine_step_id", "phase"),
)
```

With `device_lease`, runtime can deliver a `DeviceTensorLease` under
`bundle.entries["device_lease"]`. ESamp still accepts direct tensor entries as a
small test/manual integration fallback. On the model-bank path,
`first_per_prompt` compaction is a tLLM delivery feature: runtime shapes the
bundle to the per-prompt rows that ESamp actually trains, without changing the
full decode-row state used by sampler guidance.

## Asynchronous Adaptation: `out_of_band`

The training mechanism includes matrix multiplications and optimizer steps. It
must not run synchronously on the main inference path.

ESamp uses `window="out_of_band"` to make the intent explicit: this flow is
step-end async adaptation work on a side stream. Runtime captures and stages
data; ESamp's engine owns the distiller update pipeline.

## Sampler Intervention: Provider, Not Patch

ESamp wants to modify candidate-token logits. It does not patch vLLM's sampler itself. Instead, it implements a sampler modifier provider and hands it to tLLM's generic sampler bridge.

The split is:

- tLLM runtime patches `compute_logits` and the sampler boundary.
- Runtime builds a candidate-token view.
- The ESamp provider returns a candidate-level logits delta.
- Runtime applies the delta before final sampling.

This keeps ESamp-specific code out of the generic runtime and lets future consumers reuse the same sampler bridge.

## State Management

ESamp supports several state layouts:

| Mode | State strategy | When to use it |
|------|----------------|----------------|
| Single | One shared parameter set | Quick checks and small experiments |
| Per-request | One parameter set per request | Clear semantics, small fixed request count |
| Model-bank | Fixed slots assigned to active requests | Higher concurrency and recommended path |

Model-bank reduces launch overhead and works well with CUDA Graph replay for the training path.

## Component Split

| Component | Responsibility | Runtime-facing? |
|-----------|----------------|-----------------|
| `ESampConsumer` | Declares ports, consumes bundles, coordinates training | Yes |
| `ESampTrainEngine` | Owns parameters, model-bank, forward/backward | No |
| `ESampSamplerModifierProvider` | Converts distiller output into logits deltas | Yes, through sampler bridge |

Runtime only sees the consumer interfaces and sampler provider. It does not know the internals of `ESampTrainEngine`.

## Decode-Step Timing

At a high level:

1. Layer hooks capture source hidden rows.
2. Layer hooks capture target hidden rows.
3. At the `compute_logits` boundary, tLLM schedules distiller no-grad precompute.
4. The sampler bridge calls the ESamp provider on filtered candidate tokens.
5. vLLM samples from modified candidate logits.
6. At `execute_model.post` / `out_of_band`, ESamp schedules delayed backward and model-bank flushes.

Avoid running distiller forward directly inside PyTorch layer hooks. Hook-time computation looks early, but it is fragile under vLLM compile / CUDA Graph paths. The current design uses `compute_logits` as a safer boundary that is still early enough for same-step sampling.

## Performance Principles

These lessons apply beyond ESamp:

- `loss_count == 0` is a failure signal, not a success.
- Avoid `.item()`, `.tolist()`, and large `.cpu()` calls in hot paths.
- Do not drain CPU workers inside `consume_bundle()`.
- Vectorize per-row work whenever possible.
- State layout decisions often matter more than small code-level optimizations.
- Treat copy count as an implementation detail, not an API promise. A
  `device_lease` may be a direct view, a strided view, or a staged tensor as the
  runtime evolves.
- A long Python section does not force CUDA to synchronize, but it can delay the
  next decode enqueue. Side-stream work can also compete with vLLM kernels for
  SMs, L2, and memory bandwidth. When tap-only throughput is already below the
  baseline, look at hook operations and stream scheduling before blaming
  distiller FLOPs.
- ESamp exposes `adaptation_stream_mode` as an implementation tuning knob for
  its own engine. `dual` is the normal overlapped path, `single` coalesces
  ESamp's adaptation staging/training work onto one auxiliary stream, and
  `serial` runs adaptation work on the current stream as a diagnostic baseline.
  On CUDA devices where vLLM uses default-priority streams, lowering ESamp
  priority may not be possible; the practical fixes are usually less hot-path
  capture/delivery work, fewer graph launches, or a budgeted adaptation queue.

## Building Your Own Consumer

Most consumers do not need ESamp's complexity. Start small:

1. Declare one `ConsumerFlow`.
2. Read data in `consume_bundle()`.
3. Add stats and verify nonzero processing.
4. Add async workers, state, or sampler guidance only when the simple path works.

ESamp is a reference for what the framework can support, not a template to copy line by line.

For a side-by-side explanation of the ordinary DummyConsumer path and ESamp's
advanced path, read [Consumer Delivery Modes](consumer-delivery-modes.md).
