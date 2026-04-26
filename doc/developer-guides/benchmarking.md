# Benchmarking

This guide explains how to measure the throughput impact of a consumer.

The central rule is simple: compare ratios, not isolated absolute throughput. Absolute token/s numbers vary across GPUs, drivers, CUDA versions, model sizes, and sampling shapes.

## The Core Metric

Every benchmark needs at least two cases:

- `off`: vanilla vLLM, no consumer.
- `on`: same workload, consumer enabled.

The useful number is:

```text
ratio = on_tok_per_s / off_tok_per_s
```

If the consumer trains or modifies sampling, also check functional counters such as `loss_count`, processed rows, or candidate counts. `loss_count == 0` is not a fast success; it means training did not happen.

## Generic Consumer Benchmark Shape

```python
from tllm.runtime import residual_runtime as runtime
from tllm.consumers.my_consumer import MyConsumer, MyConsumerConfig

consumer = MyConsumer(MyConsumerConfig())
runtime.register_dispatch_consumer(consumer)

outputs = llm.generate(prompts, sampling_params)

consumer.synchronize()
stats = consumer.read_stats()
```

Keep `prompts`, sampling params, model, batch size, dtype, prefix caching, and max model length identical between `off` and `on`.

## DummyConsumer Baseline

DummyConsumer is configured to be intentionally light by default:

- `dispatch_every_n_steps=256`
- `max_bundle_rows=1`
- `export_max_cols=16`

This makes it a useful lower bound for framework overhead. In one aligned RTX 4090 / Qwen2.5-0.5B / batch=8 / n=16 / max_new_tokens=256 run, vanilla vLLM reached about 27691 tok/s and DummyConsumer reached about 26934 tok/s, or roughly 0.97.

Always check that `processed_batches > 0` and `processed_rows > 0`; otherwise a high ratio may simply mean the consumer did no work.

## First Performance Triage

If throughput drops sharply, check in this order:

1. Search for hot-path synchronization:

   ```bash
   rg -n "\\.item\\(|\\.tolist\\(|\\.cpu\\(|synchronize\\(|print\\(" tllm
   ```

2. Check whether `consume_bundle()` does heavy CPU work.
3. Check whether every step copies large tensors across devices.
4. Check whether workers are drained too frequently.
5. Confirm the `off` and `on` workloads are truly identical.

## Next Steps

- [ESamp Usage](../reference/esamp-usage.md)
- [Consumer Delivery Modes](consumer-delivery-modes.md)
- [Validation](validation.md)
- [Debugging](debugging.md)
