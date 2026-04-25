# Validation

After changing a consumer, producer, or runtime hook, do not rely on one test. tLLM uses three validation layers:

| Layer | Coverage | When to use it |
|-------|----------|----------------|
| Unit tests | Pure logic, contracts, regressions | After any code change |
| GPU correctness | Hidden alignment, MSE, generation consistency | After producer/localization/runtime changes |
| Functional activation | Training or guidance actually happens | After consumer training or sampler changes |

## Ordinary Consumer Changes

If you changed only consumer logic:

```bash
python -m pytest -q test/test_dummy_consumer_unit.py
python -m pytest -q test/test_consumer_dispatch_contracts_unit.py
```

These tests check that `flows()` declarations, bundle assembly, and basic consumer behavior still match the public contract.

Add your own tests for your consumer's behavior and stats.

## Producer or Runtime Changes

If you changed a port, bundle assembly, hook timing, or localization:

```bash
python -m pytest -q \
  test/test_port_catalog_unit.py \
  test/test_port_bundle_assembler_unit.py \
  test/test_runtime_port_bridge_unit.py
```

If hidden-row localization is involved, also run a GPU MSE check.

## Decode MSE

Decode MSE verifies that tLLM reconstructs the right rows from vLLM's packed decode tensor.

```bash
python -m verify_v1_decode_rows_minimal \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "hello" \
  --max-new-tokens 8 \
  --mse-tol 1e-4
```

The check compares:

1. A gold path that runs requests independently.
2. A batched vLLM path where rows must be localized.

If localization is correct, corresponding hidden rows have very small MSE. If the row index or phase logic is wrong, MSE will exceed the threshold.

## Prefill Validation

Prefill is different because one request may occupy many consecutive rows.

```bash
python -m tllm.workflows.repro.repro_prefill_sampling_mse \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --prompt-file test/prompt_debug_list.txt \
  --gen-max-new-tokens 4 \
  --sampling-n 3 \
  --mse-tol 1e-5 \
  --gpu-memory-utilization 0.3 \
  --max-model-len 256
```

Use this after changing `tllm/producer/prefill.py` or any runtime metadata that affects prefill spans.

## Sampler / Guidance Validation

If a consumer modifies logits or sampling:

```bash
python -m pytest -q \
  test/test_sampler_port_unit.py \
  test/test_sampler_patch_unit.py
```

For min-p candidate intervention:

```bash
python -m pytest -q test/test_sampler_bridge_minp_unit.py
```

These checks make sure the provider receives the intended candidate set and returns deltas with the right shape.

## ESamp Correctness

For ESamp, unit tests are not enough. You also need an end-to-end run where `loss_count > 0`.

```bash
python -m pytest -q \
  test/test_esamp_per_request_unit.py \
  test/test_esamp_distiller_sampling_integration_unit.py \
  test/test_esamp_model_bank_backend_unit.py
```

Then run an aligned GPU validation or benchmark:

```bash
VLLM_USE_FLASHINFER_SAMPLER=1 \
python -m tllm.verification.automated_tests \
  --scenario esamp_loss_parity_qwen2p5_0p5b
```

The key check is not only throughput. Verify that `single_on`, `per_request_on`, and `model_bank_on` report nonzero loss counts when those paths are expected to train.

## Third-Party Extension Checklist

Before submitting a new consumer or runtime extension:

- The consumer faces `tllm.ports` and `PortBundle`, not vLLM internals.
- Async resources are released in `synchronize()`.
- Hot paths avoid CPU sync.
- Unit tests cover the public contract.
- GPU validation is run when localization, hooks, or sampler behavior changes.
