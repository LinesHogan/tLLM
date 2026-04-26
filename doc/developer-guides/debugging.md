# Debugging Consumers

Debugging tLLM consumers works best if you first identify the boundary where the problem lives:

| Boundary | Responsibility | Typical failures |
|----------|----------------|------------------|
| Producer | Locate rows in vLLM packed tensors | Hidden rows are wrong, MSE fails |
| Runtime | Hooks, frames, bundles, dispatch | Consumer is not called, fields are missing |
| Consumer | Your algorithm | Stats are wrong, queues leak, sync slows everything down |

## Consumer Is Never Called

Check:

1. Does `flows()` return at least one `ConsumerFlow`?
2. Does the requested phase match the current stage (`prefill` or `decode`)?
3. Does the `role` match the key you read from `bundle.entries`?
4. Does `bundle_key` group the data you expect into the same bundle?
5. Was the consumer registered before generation?
6. Was the consumer disabled?

Add a cheap counter at the beginning of `consume_bundle()` and read it after generation.

## Bundle Fields Look Wrong

`residual_stream` tensors are usually views into runtime buffers.

- Read-only within the current call: use directly.
- Keep across steps: clone.
- Send to CPU: use non-blocking staging.
- Modify in-place only if you are explicitly implementing a write-back port.

If you suspect row localization is wrong, run the MSE checks in [Validation](validation.md).

## CPU/GPU Synchronization Problems

Start by searching for common sync traps:

```bash
rg -n "\\.item\\(|\\.tolist\\(|\\.cpu\\(|synchronize\\(|print\\(" tllm
```

Common causes:

- Reading a GPU scalar with `.item()` every step.
- Calling `.tolist()` on a GPU tensor.
- Copying a full tensor to CPU in the hot path.
- Draining a CPU worker inside `consume_bundle()`.
- Printing every step.

Debug logging is fine temporarily. Do not leave hot-path logging in committed code.

## Async CPU Worker Does Not Drain

For DummyConsumer-style async workers:

- `consume_bundle()` should enqueue work only.
- `on_step_end()` may drain at a controlled interval.
- `synchronize()` must drain leftovers at the end.
- When the queue is full, prefer dropping work or applying non-hot-path backpressure.

## Sampler / Guidance Does Not Take Effect

Split the problem:

1. Is the provider active?
2. Is the sampler patch calling the provider?
3. Is the returned logits delta shaped like the candidate view?

For ESamp-specific guidance, read [ESamp Usage](../reference/esamp-usage.md).

## Throughput Regressed

Do not guess first. Compare against a matching `off` baseline and then inspect:

- CPU sync keywords.
- Python loops over per-row or per-token work.
- Branch-heavy GPU-adjacent code that could be vectorized.
- Whether vLLM is being forced to wait for side-stream work.
- Whether `loss_count` or processed-row counters prove real work happened.

## Related Docs

- [Benchmarking](benchmarking.md)
- [Validation](validation.md)
- [Architecture](architecture.md)
