# vLLM Compatibility

This page explains how tLLM handles vLLM version differences.

tLLM installs monkey patches at key `GPUModelRunner` lifecycle boundaries. vLLM changes internal interfaces across versions, especially the return structure of `_prepare_inputs`. tLLM isolates those differences behind versioned adapters, so the runtime can consume a normalized view.

## Supported Version Families

- `0.7.x`
- `0.10.x`
- `0.11+`

The current development environment uses `vllm==0.10.1.1`, so 0.10.x is the primary day-to-day validation target. 0.7.x remains a historical compatibility baseline.

## What the Adapter Normalizes

The normalized view contains:

- `attn_metadata`
- `logits_indices`
- `spec_decode_common`
- `num_scheduled_tokens_np`

## Current Parsing Rules

For a 6-tuple `_prepare_inputs` result:

- `attn_metadata = out[0]`
- `logits_indices = out[1]`
- `num_scheduled_tokens_np = out[3]`
- `spec_decode_common = out[4]`

For a 2-tuple result:

- `attn_metadata = out[0]`
- `logits_indices = out[1]`
- `spec_decode_common = None`
- `num_scheduled_tokens_np` is derived from `scheduler_output.num_scheduled_tokens`

## Adapter Family

- `V072PrepareInputsAdapter`: vLLM 0.7.x
- `V010PrepareInputsAdapter`: vLLM 0.10.x
- `V011PlusPrepareInputsAdapter`: vLLM 0.11+

The implementation shares parsing helpers but keeps version-family boundaries explicit.

## Who Uses It?

Developer verification and repro paths:

- `verify_v1_decode_rows_minimal.py`
- `tllm/workflows/repro/prefill_capture_support.py`
- `tllm/runtime/residual_runtime.py`

Runtime paths:

- `tllm/runtime/vllm_patch/port_runtime_hooks.py`
- `tllm/runtime/ports/`

Both groups consume the same normalized adapter view before doing localization or bundle assembly.
