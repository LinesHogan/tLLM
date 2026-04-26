# Developer Testing Guide

This guide explains which tests to run after changing tLLM.

tLLM uses three layers:

| Layer | Location | Purpose | Command style |
|-------|----------|---------|---------------|
| Unit tests | `test/` | CPU logic, contracts, regressions | `pytest -q` |
| Verification harness | `tllm/verification/` | GPU pass/fail scenarios | `python -m tllm.verification...` |
| Manual workflows | `tllm/workflows/` | Benchmarks, repros, experiments | `python -m tllm.workflows...` |

Rules:

- Keep pytest assertions in `test/`.
- Put GPU pass/fail validation in `tllm/verification/`.
- Use `tllm/workflows/` for manual benchmarks and repros.

## TDD Flow

1. Write or update a failing test.
2. Run it and confirm it fails.
3. Implement the smallest fix.
4. Run the regression set.
5. Refactor and update docs if needed.

Do not call a throughput improvement successful if `loss_count == 0`.

## Unit Tests

```bash
python -m pytest -q
```

Useful focused groups:

| Test | Focus | Run when |
|------|-------|----------|
| `test_decode_localization_unit.py` | Decode row localization | Producer/localization changes |
| `test_consumer_dispatch_contracts_unit.py` | ConsumerFlow and bundles | Port or bundle changes |
| `test_dummy_consumer_unit.py` | Minimal consumer behavior | Consumer base/public API changes |
| `test_esamp_per_request_unit.py` | ESamp training behavior | Training path changes |
| `test_esamp_model_bank_backend_unit.py` | Model-bank path | Model-bank state or scheduling changes |

Minimal logic-only check:

```bash
python -m pytest -q \
  test/test_decode_localization_unit.py \
  test/test_consumer_dispatch_contracts_unit.py
```

## Producer or Runtime Changes

Run unit tests plus decode MSE:

```bash
python -m pytest -q
python -m verify_v1_decode_rows_minimal \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "hello" \
  --max-new-tokens 8 \
  --mse-tol 1e-4
```

Unit tests cover logic. MSE validates that the real vLLM GPU path still maps hidden rows correctly.

## ESamp or Training Changes

Run unit tests, MSE, and an aligned ESamp benchmark:

```bash
python -m pytest -q
python -m verify_v1_decode_rows_minimal \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "hello" \
  --max-new-tokens 8 \
  --mse-tol 1e-4

VLLM_USE_FLASHINFER_SAMPLER=1 \
python -m tllm.workflows.benchmarks.per_request_esamp_benchmark \
  --emit-json-summary \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --benchmark-batch-size 8 \
  --benchmark-max-new-tokens 256 \
  --distiller-lr 1e-3 \
  --model-bank-train-cudagraph \
  --run-model-bank-case
```

Check both throughput ratio and `loss_count`. A fast run with zero loss is a bug.

## One-Line Rule

Use `test/` to prove logic, `tllm/verification/` to prove GPU behavior, and `tllm/workflows/` to measure or reproduce experiments.
