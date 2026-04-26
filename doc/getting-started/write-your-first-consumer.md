# Write Your First Consumer

This guide is for developers who want to insert their own logic into generation.

You will learn:

1. How `ConsumerFlow` declares data needs.
2. What arrives in `consume_bundle()`.
3. How to export data to CPU without blocking generation.
4. How to flush async work at the end.

## Start With DummyConsumer

`tllm/consumers/dummy/` is the smallest useful example. It:

- Reads `residual_stream` and `request_meta`.
- Copies hidden rows to CPU asynchronously.
- Runs a tiny CPU-side demo worker.
- Drops work instead of blocking when its queue is full.

It is not a production algorithm. It is a teaching example for a healthy async pattern.

The main implementation lives in `tllm/consumers/dummy/consumer.py`.

## 1. Copy the Template

```bash
cp -r tllm/consumers/dummy tllm/consumers/my_consumer
```

Rename `DummyConsumer` and `DummyConsumerConfig`, then choose your own `consumer_id`.

## 2. Declare the Data You Need

The core method is `flows()`. It tells the runtime which ports to read and when to deliver them.

```python
from tllm.ports.base import ConsumerFlow
from tllm.ports.residual_stream import ResidualStream
from tllm.ports.request_meta import RequestMeta
from tllm.ports.cpu_export import CpuExport

def flows(self):
    return [
        ConsumerFlow(
            reads=(
                ResidualStream.read(
                    layer=0,
                    site="block_output",
                    phase="decode",
                    role="hidden",
                ),
                RequestMeta.read(),
            ),
            writes=(
                CpuExport.write(channel="my_consumer", format="row_batch"),
            ),
            window="background",
            bundle_key=("engine_step_id", "phase"),
        )
    ]
```

Important fields:

- `role` becomes the key in `bundle.entries`.
- `window="background"` means asynchronous processing.
- `bundle_key` tells the runtime how to group frames into one complete bundle.
- The default delivery mode is `delivery="bundle"` with borrowed entries. Keep
  that default unless profiling shows that your consumer needs device-lease
  delivery.

## 3. Consume the Bundle

```python
def consume_bundle(self, bundle, ctx):
    hidden = bundle.entries.get("hidden")
    if hidden is None:
        return
    self.num_rows += int(hidden.shape[0])
```

Remember:

- Hidden tensors are usually views into runtime buffers.
- Clone if you need to keep data across steps.
- Avoid `.item()`, `.tolist()`, large `.cpu()`, or worker draining in the hot path.

## 4. Export to CPU Asynchronously

If your logic mostly runs on CPU, use a staged non-blocking copy:

```python
def consume_bundle(self, bundle, ctx):
    hidden = bundle.entries.get("hidden")
    if hidden is None:
        return
    if self.worker.pending() >= self.config.max_queue_size:
        self.dropped += 1
        return

    def submit():
        hidden_cpu = hidden.detach().to("cpu", non_blocking=True)
        self.worker.enqueue(hidden_cpu)

    self.stream_runtime.run(submit)

def synchronize(self):
    self.stream_runtime.synchronize()
    self.worker.drain()
```

The rule of thumb: enqueue in `consume_bundle()`, process later.

## 5. Register the Consumer

For simple consumers, direct registration is fine:

```python
from tllm import register_consumer
from tllm.consumers.my_consumer import MyConsumer, MyConsumerConfig

consumer = MyConsumer(MyConsumerConfig())
register_consumer(consumer)

outputs = llm.generate(prompts, sampling_params)

consumer.synchronize()
stats = consumer.read_stats()
```

Workflow helpers can still wrap this pattern for benchmarks and demos, but third-party consumers should start with explicit registration.

## Validate It

Start with the DummyConsumer tests:

```bash
python -m pytest -q test/test_dummy_consumer_unit.py
```

Then add tests for your own consumer:

- `flows()` returns the expected `ConsumerFlow`.
- `consume_bundle()` handles expected input.
- `synchronize()` drains resources safely.

## Common Mistakes

| Symptom | Likely cause |
|---------|--------------|
| `consume_bundle()` never runs | No flow, phase mismatch, or consumer not registered |
| Bundle fields look wrong | You kept a view across steps without cloning |
| Throughput collapses | Hot path has `.item()`, `.tolist()`, `.cpu()`, sync, or logging |
| Queue grows forever | Worker is not drained or backpressure blocks the hot path |

## Next Steps

- [Consumer Delivery Modes](../developer-guides/consumer-delivery-modes.md)
- [Benchmarking](../developer-guides/benchmarking.md)
- [Validation](../developer-guides/validation.md)
- [Debugging](../developer-guides/debugging.md)
