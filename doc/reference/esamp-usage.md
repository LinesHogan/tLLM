# ESamp Usage

This is a command reference for ESamp. It focuses on how to run it, what the parameters mean, and how to read the output.

For design background, read [ESamp Design](../developer-guides/esamp-design.md).

Programmatic integrations create `ESampConsumer`, register it with `tllm.register_consumer(...)`, then call `llm.generate(...)`. `configure_esamp_runtime()` remains a workflow helper for benchmarks and short demos.

## Functional Check

Before benchmarking, first confirm that ESamp training mechanism runs and loss appears:

```bash
python -m tllm.workflows.repro.repro_esamp_loss \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --prompt-file test/prompt_debug_list.txt \
  --source-layer-path model.model.layers[0] \
  --target-layer-path model.model.layers[-1]
```

Check that `loss_count > 0` and that the loss is finite. If loss is zero or NaN, fix functionality before measuring throughput.

## Throughput Benchmark

```bash
VLLM_USE_FLASHINFER_SAMPLER=1 \
python -m tllm.workflows.benchmarks.per_request_esamp_benchmark \
  --emit-json-summary \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 512 \
  --benchmark-batch-size 8 \
  --benchmark-max-new-tokens 256 \
  --benchmark-warmup-rounds 1 \
  --benchmark-rounds 2 \
  --benchmark-ignore-eos \
  --benchmark-disable-prefix-caching \
  --sampling-n 16 \
  --sampling-temperature 0.8 \
  --sampling-top-p 0.95 \
  --sampling-top-k -1 \
  --distiller-lr 1e-3 \
  --model-bank-flush-interval 1 \
  --model-bank-init-method ffn_fast_svd \
  --trajectory-topk 1 \
  --model-bank-train-cudagraph \
  --run-model-bank-case
```

Read the ratio:

```text
ratio = model_bank_on / single_off
```

| Metric | Good sign | If it looks wrong |
|--------|-----------|-------------------|
| `loss_count` | Must be > 0 | Consumer or training path did not activate |
| `loss_avg` | Finite and reasonable | Check LR, initialization, or workload |
| `single_off` | Close to naked vLLM | Environment or workload may be off |
| `model_bank_on` | Lower than baseline but not dramatically lower | Look for CPU sync or hot-path blocking |
| `ratio` | Compare within the same environment | Optimized 7B min-p paths have reached 95%+; small models and unoptimized paths may be lower |

## Distiller Sampling Intervention

Enable distiller guidance with:

```bash
  --enable-distiller-intervention \
  --distiller-beta 0.1 \
  --distiller-sampler-backend post_filter_exact
```

The intervention formula is:

```text
new_logit = (1 + beta) * llm_logit - beta * distiller_logit
```

`post_filter_exact` modifies only the candidates kept by the LLM sampler after filtering. This avoids a full-vocabulary projection on the hot sampler path.

## Optional Triton Grouped Backend

The default model-bank prediction backend is `torch`. CUDA/Qwen throughput experiments can try:

```bash
  --model-bank-forward-backend triton_grouped
```

This affects the no-grad model-bank prediction / sampling fast path. Training and autograd still use torch.

Use it after the torch backend already works. If it fails or regresses, return to the default.

## Parameter Quick Reference

| Parameter | Meaning | Default or suggestion |
|-----------|---------|-----------------------|
| `--source-layer-path` | Distiller input hidden layer | Early layer |
| `--target-layer-path` | Training target hidden layer | Late layer |
| `--distiller-hidden-dim` | Side model hidden width | Tune for quality/cost |
| `--distiller-lr` | ESamp distiller learning rate | `1e-3` |
| `--model-bank-rank` | Low-rank model-bank rank | `64` |
| `--model-bank-flush-interval` | Optimizer flush interval | `1` for strict training |
| `--model-bank-init-method` | Initialization method | `ffn_fast_svd` for Qwen paths |
| `--model-bank-train-cudagraph` | Capture the ESamp distiller update graph | Recommended for benchmark paths |
| `--enable-distiller-intervention` | Enable sampler guidance | Off unless needed |
| `--distiller-beta` | Guidance strength | Start with `0.1` |
| `--distiller-sampler-backend` | Guidance backend | Keep `post_filter_exact` unless experimenting |
