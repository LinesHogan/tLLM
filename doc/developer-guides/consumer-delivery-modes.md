# Consumer Delivery Modes

Most consumers should use the ordinary bundle path. It is stable, easy to test,
and keeps your code independent from runtime buffer management.

The device-lease path exists for a narrower case: a consumer does per-step GPU
work, wants to overlap that work with generation, and can follow stricter buffer
lifetime rules. ESamp uses this path because it is an adaptive/guidance consumer
with a training mechanism in the decode loop. That does not make ESamp a synonym
for training; it is one example of a more demanding consumer.

## The Ordinary Bundle Path

This is the default:

```python
ConsumerFlow(
    reads=(...),
    writes=(),
    window="background",
    bundle_key=("engine_step_id", "phase"),
)
```

Leaving `delivery` and `ownership` unset means:

```python
delivery="bundle"
ownership="borrowed"
```

Runtime assembles a `PortBundle`, and each requested port appears directly in
`bundle.entries`. If your flow reads a residual stream with `role="hidden"`, you
read it as:

```python
hidden = bundle.entries.get("hidden")
```

Treat tensors as borrowed views. Read them during `consume_bundle()`, or clone
what you need to keep. This is the interface third-party consumers should start
from.

DummyConsumer is the maintained example for this path. It reads hidden rows,
copies a small slice to CPU asynchronously, and drains its worker at the end.
That makes it a good template for analysis, export, logging, and CPU-side
experiments.

## The Device-Lease Path

The opt-in form is:

```python
ConsumerFlow(
    reads=(...),
    writes=(),
    window="out_of_band",
    delivery="device_lease",
    ownership="runtime_lease",
    bundle_key=("engine_step_id", "phase"),
)
```

`out_of_band` is the neutral step-end async window for GPU work that should not
block the main inference stream.

The current `device_lease` implementation is intentionally narrow: it supports
step-scope decode delivery with `bundle_key=("engine_step_id", "phase")` for
`residual_stream` reads and optional `request_meta`. It is a contract boundary
for a fuller GPU consumer lane, not a promise that every port type already has
durable staged buffers.

Runtime may place the requested GPU tensors under a lease object:

```python
lease = bundle.entries.get("device_lease")
source = lease.entries.get("source")
target = lease.entries.get("target")
```

In the current implementation, `ownership="runtime_lease"` means the tensor
entries are runtime-owned and must be treated as read-only. They are valid for
the `consume_bundle()` call. The lease advertises this as
`lifetime="consume_call"`. A consumer that needs to keep data across later
feedback or synchronization boundaries must first copy it into its own buffers,
unless a future lease contract explicitly provides durable buffers and readiness
fences.

Use this mode only when all of these are true:

- The consumer works on GPU tensors every decode step or almost every decode
  step.
- Avoiding extra staging copies matters for throughput.
- The consumer has clear synchronization points such as `on_step_end()` or
  `synchronize()`.
- Unit tests cover both the flow declaration and the lease-shaped bundle input.

Advanced flows can also request row shaping:

```python
ConsumerFlow(
    ...,
    delivery="device_lease",
    row_compaction="first_per_prompt",
)
```

This keeps the global decode-step state unchanged, but shapes the bundle
delivered to that flow. `first_per_prompt` is useful when a consumer's work is
per prompt rather than per expanded sample. Runtime also includes
`row_ids` when request metadata is delivered as `RowBatchMeta`, so the consumer
can relate compact rows back to the full decode step. Metadata cardinality
matches the delivered live rows; spare tensor capacity belongs only to the
lease, not to metadata.

When compact row ids form a regular stride, runtime may deliver a strided
runtime-owned view instead of materializing an `index_select` result. This is an
implementation optimization under the same `device_lease` lifetime rules. Do not
write code that depends on a particular copy count.

## How DummyConsumer and ESamp Differ

DummyConsumer teaches the common interface. It uses `delivery="bundle"` and
`ownership="borrowed"` because it is meant to show the safest default contract.

ESamp teaches the advanced interface. It reads source and target hidden rows,
schedules runtime adaptation work, and can provide sampler guidance. Its
training mechanism is one part of that design. Because the tensors stay on GPU
and are consumed by a step-level engine, ESamp opts into `device_lease` delivery.
When ESamp uses model-bank state, it also requests `first_per_prompt` row
compaction because the model-bank update is per prompt, not per expanded sample.

The important boundary is this:

- The runtime exposes a generic delivery contract.
- ESamp chooses the advanced contract because its workload needs it.
- Other consumers keep the default unless they have the same kind of per-step
  GPU pressure.

## Compatibility

Adding a device-lease flow does not change existing consumers. If a flow does not
opt in, runtime still delivers ordinary bundle entries. A consumer can also keep
a fallback path that accepts direct tensor entries; ESamp does this so tests and
manual integrations can construct simple bundles without a runtime lease.

When in doubt, start with DummyConsumer and the ordinary bundle path. Move to
`device_lease` only after profiling shows bundle assembly or staging copies are a
real part of the cost.

For ESamp, recent profiling showed exactly that shape: a tap-only configuration
was already measurably below the dispatch-off baseline, before training work was
enabled. That evidence belongs to ESamp's optimization history, not to the
generic contract. The generic lesson is narrower: use `device_lease` only when
profiling shows that borrowed bundle delivery or consumer-local staging is part
of the bottleneck.

Profiling can also show the opposite. Long Python sections do not synchronize
CUDA by themselves, but they can delay the next vLLM enqueue and look like lost
overlap. Extra side-stream kernels can also compete with the main decode kernels
for SMs, L2, and memory bandwidth. In that case, reducing one copy is useful but
not enough; the consumer needs fewer hot-path hook operations, less per-step
Python scheduling, or a different stream schedule.

## Related Docs

- [Write Your First Consumer](../getting-started/write-your-first-consumer.md)
- [ESamp Consumer Design](esamp-design.md)
- [Producer/Consumer Contract](../reference/producer-consumer-contract.md)
- [Benchmarking](benchmarking.md)
