#!/usr/bin/env python3
"""Verify vLLM v1 decode-row localization using first-layer hidden MSE.

Gold standard:
- Run prompts one-by-one (greedy)
- Capture first transformer layer output hidden for decode tokens only

Candidate:
- Run the same prompts in one parallel batch (greedy)
- Use v1 logits_indices to select rows from first-layer hidden

Pass condition:
- Per prompt, per decode step MSE(gold, parallel) <= mse_tol
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional


def _ensure_v1_env() -> None:
    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


_ensure_v1_env()

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # noqa: E402


# Runtime capture state.
CAPTURE_ACTIVE = False
CURRENT_REQ_IDS: List[str] = []
# Per request in CURRENT_REQ_IDS: whether this request is in true decode stage.
CURRENT_IS_DECODE_REQ: List[bool] = []
CURRENT_LOGITS_INDICES: Optional[torch.Tensor] = None
# Number of real tokens in this step (excluding cudagraph padding rows).
CURRENT_NUM_ACTUAL_TOKENS = 0
CURRENT_H1_SCRATCH: Optional[torch.Tensor] = None
REQID_TO_PROMPTIDX: Dict[str, int] = {}
CAPTURED: Dict[int, List[torch.Tensor]] = {}
HOOK_LAYER_INFO = "unknown"
DEBUG_VERIFY = os.environ.get("VERIFY_V1_DEBUG", "0") == "1"
DEBUG_MAX_STEPS = 32
DEBUG_STEP = 0
TAP_CALL_COUNT = 0
TAP_LAST_SUM = float("nan")
CAPTURE_IMPL = "graph_copy"  # graph_copy | python_hook
GRAPH_SCRATCH_ROWS_HINT = 0


_ORIG_PREPARE_INPUTS = GPUModelRunner._prepare_inputs
_ORIG_EXECUTE_MODEL = GPUModelRunner.execute_model
_ORIG_LOAD_MODEL = GPUModelRunner.load_model


def _pick_common_attn_metadata(attn_metadata: Any, fallback: Any) -> Any:
    # v1 returns per-layer metadata dict; query_start_loc/num_actual_tokens are
    # shared semantically, so taking the first entry is sufficient.
    if fallback is not None:
        return fallback
    if isinstance(attn_metadata, dict):
        for meta in attn_metadata.values():
            return meta
    if hasattr(attn_metadata, "query_start_loc"):
        return attn_metadata
    return None


def _dbg(msg: str) -> None:
    if DEBUG_VERIFY:
        print(f"[verify_v1_debug] {msg}")


def _ensure_global_scratch(rows: int, hidden_dim: int, device: torch.device) -> torch.Tensor:
    global CURRENT_H1_SCRATCH
    need_new = (
        CURRENT_H1_SCRATCH is None
        or CURRENT_H1_SCRATCH.device != device
        or CURRENT_H1_SCRATCH.dtype != torch.float32
        or CURRENT_H1_SCRATCH.shape[1] != hidden_dim
        or CURRENT_H1_SCRATCH.shape[0] < rows
    )
    if need_new:
        CURRENT_H1_SCRATCH = torch.empty(
            (rows, hidden_dim), device=device, dtype=torch.float32
        )
    return CURRENT_H1_SCRATCH


def _copy_tensor_to_global_scratch(tensor: torch.Tensor) -> None:
    global TAP_CALL_COUNT, TAP_LAST_SUM
    rows = int(min(CURRENT_NUM_ACTUAL_TOKENS, tensor.shape[0]))
    if rows <= 0:
        return
    hidden_dim = int(tensor.shape[-1])
    scratch = _ensure_global_scratch(rows, hidden_dim, tensor.device)
    scratch[:rows, :hidden_dim].copy_(tensor[:rows, :hidden_dim].to(torch.float32))
    TAP_CALL_COUNT += 1
    TAP_LAST_SUM = float(scratch[0, : min(8, hidden_dim)].sum().item())


def _collect_from_scratch() -> None:
    if not CAPTURE_ACTIVE:
        return
    if CURRENT_LOGITS_INDICES is None:
        return
    if CURRENT_H1_SCRATCH is None:
        return
    rows = int(min(CURRENT_NUM_ACTUAL_TOKENS, CURRENT_H1_SCRATCH.shape[0]))
    if rows <= 0:
        return
    hidden = CURRENT_H1_SCRATCH[:rows]

    row_idx_all = CURRENT_LOGITS_INDICES
    if row_idx_all.numel() == 0:
        return

    # Only gather rows for decode requests. This avoids requiring scratch to
    # cover prefill-only rows in mixed prefill/decode steps.
    decode_positions = [
        i for i, is_decode in enumerate(CURRENT_IS_DECODE_REQ)
        if is_decode and i < int(row_idx_all.numel()) and i < len(CURRENT_REQ_IDS)
    ]
    if not decode_positions:
        return

    decode_pos_cpu = torch.tensor(decode_positions, dtype=torch.long, device=row_idx_all.device)
    row_idx = row_idx_all.index_select(0, decode_pos_cpu)
    if row_idx.device != hidden.device:
        row_idx = row_idx.to(device=hidden.device)

    min_idx = int(row_idx.min().item())
    max_idx = int(row_idx.max().item())
    if min_idx < 0 or max_idx >= rows:
        raise RuntimeError(
            f"logits_indices out of range: min={min_idx} max={max_idx} rows={rows}"
        )

    selected = hidden.index_select(0, row_idx)
    global DEBUG_STEP
    if DEBUG_VERIFY and DEBUG_STEP < DEBUG_MAX_STEPS:
        _dbg(
            "collect "
            f"step={DEBUG_STEP} rows={rows} selected={selected.shape[0]} "
            f"reqs={len(CURRENT_REQ_IDS)} decode_true={sum(1 for x in CURRENT_IS_DECODE_REQ if x)} "
            f"logits_min={int(row_idx.min().item())} logits_max={int(row_idx.max().item())}"
        )
        DEBUG_STEP += 1
    if not bool(torch.isfinite(selected).all().item()):
        nan = int(torch.isnan(selected).sum().item())
        neginf = int(torch.isneginf(selected).sum().item())
        posinf = int(torch.isposinf(selected).sum().item())
        raise RuntimeError(
            f"Non-finite first-layer hidden detected: nan={nan} neginf={neginf} posinf={posinf}"
        )

    for selected_i, req_pos in enumerate(decode_positions):
        if selected_i >= selected.shape[0]:
            break
        req_id = CURRENT_REQ_IDS[req_pos]
        prompt_idx = REQID_TO_PROMPTIDX.get(req_id)
        if prompt_idx is None:
            continue
        if prompt_idx not in CAPTURED:
            continue
        CAPTURED[prompt_idx].append(
            selected[selected_i].detach().to(device="cpu", dtype=torch.float32)
        )


def _find_first_layer(model) -> Optional[torch.nn.Module]:
    # Common vLLM HF causal LM layout (e.g. Qwen2ForCausalLM):
    # model.model.layers[0]
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is not None and len(layers) > 0:
        return layers[0]

    # Minimal fallback for other model wrappers.
    roots = [model, inner, getattr(model, "module", None)]
    for root in roots:
        if root is None:
            continue
        decoder = getattr(root, "decoder", None)
        if decoder is not None:
            dec_layers = getattr(decoder, "layers", None)
            if dec_layers is not None and len(dec_layers) > 0:
                return dec_layers[0]
        transformer = getattr(root, "transformer", None)
        if transformer is not None:
            blocks = getattr(transformer, "h", None)
            if blocks is not None and len(blocks) > 0:
                return blocks[0]
    return None


def _infer_hidden_size(model, layer) -> int:
    hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
    if hidden_size > 0:
        return hidden_size
    ln = getattr(layer, "input_layernorm", None)
    weight = getattr(ln, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.numel() > 0:
        return int(weight.numel())
    raise RuntimeError("Could not infer hidden size for first-layer scratch buffer")


def _ensure_first_layer_hook(runner) -> None:
    global HOOK_LAYER_INFO, CURRENT_H1_SCRATCH
    model = runner.model
    if getattr(model, "_first_hidden_mse_hook_installed", False):
        return
    layer = _find_first_layer(model)
    if layer is None:
        raise RuntimeError("Could not find first transformer layer for hook registration")

    # For Qwen-like model layout, force exact first decoder layer.
    direct_layers = getattr(getattr(model, "model", None), "layers", None)
    if direct_layers is not None and len(direct_layers) > 0:
        expected = direct_layers[0]
        if layer is not expected:
            raise RuntimeError("Hook layer mismatch: expected model.model.layers[0]")
        HOOK_LAYER_INFO = "model.model.layers[0]"
    else:
        HOOK_LAYER_INFO = type(layer).__name__

    if CAPTURE_IMPL == "python_hook":
        if getattr(layer, "_first_hidden_mse_python_hook_installed", False):
            model._first_hidden_mse_hook_installed = True
            return

        def _python_hook(_module, _inputs, output):
            if not CAPTURE_ACTIVE:
                return
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(tensor, torch.Tensor):
                _copy_tensor_to_global_scratch(tensor)

        handle = layer.register_forward_hook(_python_hook)
        layer._first_hidden_mse_python_hook_installed = True
        model._first_hidden_mse_hook_installed = True
        model._first_hidden_mse_hook_handle = handle
        return

    # graph_copy path: inject copy op into layer forward.
    # Prefer decode-oriented row budget over max_num_tokens persistent buffer.
    # This keeps scratch tied to parallel sequence count.
    if GRAPH_SCRATCH_ROWS_HINT > 0:
        max_rows = int(GRAPH_SCRATCH_ROWS_HINT)
    else:
        max_rows = int(getattr(runner, "max_num_reqs", 0) or 0)
    if max_rows <= 0:
        raise RuntimeError(
            "Could not determine graph scratch rows. "
            "Please pass --graph-scratch-rows explicitly."
        )
    max_rows = max(1, max_rows)
    hidden_size = _infer_hidden_size(model, layer)
    scratch_name = "_verify_h1_scratch"
    scratch = getattr(layer, scratch_name, None)
    need_new_scratch = (
        not isinstance(scratch, torch.Tensor)
        or scratch.device != runner.device
        or scratch.dtype != torch.float32
        or tuple(scratch.shape) != (max_rows, hidden_size)
    )
    if need_new_scratch:
        new_scratch = torch.empty(
            (max_rows, hidden_size),
            device=runner.device,
            dtype=torch.float32,
        )
        if not hasattr(layer, scratch_name):
            layer.register_buffer(scratch_name, new_scratch, persistent=False)
        else:
            setattr(layer, scratch_name, new_scratch)
        scratch = getattr(layer, scratch_name)
    CURRENT_H1_SCRATCH = scratch

    if not getattr(layer, "_first_hidden_mse_forward_patched", False):
        orig_forward = layer.forward

        def _forward_with_tap(*args, **kwargs):
            out = orig_forward(*args, **kwargs)
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(tensor, torch.Tensor):
                scratch_buf = getattr(layer, scratch_name, None)
                if isinstance(scratch_buf, torch.Tensor):
                    rows = min(int(tensor.shape[0]), int(scratch_buf.shape[0]))
                    cols = min(int(tensor.shape[-1]), int(scratch_buf.shape[1]))
                    scratch_buf[:rows, :cols].copy_(tensor[:rows, :cols].to(torch.float32))
            return out

        layer.forward = _forward_with_tap  # type: ignore[method-assign]
        layer._first_hidden_mse_forward_patched = True

    model._first_hidden_mse_hook_installed = True
    model._first_hidden_mse_hook_handle = None


def _wrapped_execute_model(self, *args, **kwargs):
    global TAP_CALL_COUNT
    _ensure_first_layer_hook(self)
    tap_before = TAP_CALL_COUNT
    out = _ORIG_EXECUTE_MODEL(self, *args, **kwargs)
    tap_after = TAP_CALL_COUNT
    if DEBUG_VERIFY and CAPTURE_ACTIVE and DEBUG_STEP < DEBUG_MAX_STEPS:
        _dbg(
            f"exec tap_delta={tap_after - tap_before} "
            f"tap_total={tap_after} tap_last_sum={TAP_LAST_SUM:.6e}"
        )
    _collect_from_scratch()
    return out


def _wrapped_load_model(self, *args, **kwargs):
    # Install patch right after model load and before warmup/cudagraph capture.
    out = _ORIG_LOAD_MODEL(self, *args, **kwargs)
    _ensure_first_layer_hook(self)
    return out


def _wrapped_prepare_inputs(self, scheduler_output):
    out = _ORIG_PREPARE_INPUTS(self, scheduler_output)
    if not isinstance(out, tuple) or len(out) < 6:
        return out

    # Current vLLM v1 signature:
    # (
    #   attn_metadata,
    #   logits_indices,
    #   spec_decode_metadata,
    #   num_scheduled_tokens_np,
    #   spec_decode_common_attn_metadata,
    #   max_query_len,
    # )
    (
        attn_metadata,
        logits_indices,
        _spec_decode_metadata,
        _num_scheduled_tokens_np,
        spec_decode_common,
        _max_query_len,
    ) = out

    common = _pick_common_attn_metadata(attn_metadata, spec_decode_common)

    req_ids = [
        rid for rid in self.input_batch.req_ids[: self.input_batch.num_reqs] if rid is not None
    ]
    # Use input_batch arrays instead of self.requests[*] to avoid stale/async
    # state under cudagraph mode.
    req_id_to_index = self.input_batch.req_id_to_index
    num_prompt_tokens = self.input_batch.num_prompt_tokens
    num_computed_tokens = self.input_batch.num_computed_tokens_cpu
    is_decode_req: List[bool] = []
    for req_id in req_ids:
        req_idx = req_id_to_index.get(req_id)
        if req_idx is None:
            is_decode_req.append(False)
            continue
        prompt_len = int(num_prompt_tokens[req_idx])
        num_computed = int(num_computed_tokens[req_idx])
        # Decode starts after all prompt tokens are already computed.
        is_decode_req.append(num_computed >= prompt_len)

    global DEBUG_STEP
    if DEBUG_VERIFY and DEBUG_STEP < DEBUG_MAX_STEPS:
        preview = []
        for rid in req_ids[:4]:
            idx = req_id_to_index.get(rid)
            if idx is None:
                preview.append(f"{rid}:idx=None")
            else:
                preview.append(
                    f"{rid}:comp={int(num_computed_tokens[idx])}/prompt={int(num_prompt_tokens[idx])}"
                )
        _dbg(
            "prepare "
            f"step={DEBUG_STEP} reqs={len(req_ids)} decode_true={sum(1 for x in is_decode_req if x)} "
            f"num_actual={int(getattr(common, 'num_actual_tokens', 0) or 0) if common else -1} "
            f"logits_n={int(logits_indices.numel()) if isinstance(logits_indices, torch.Tensor) else -1} "
            f"preview={preview}"
        )

    global CURRENT_REQ_IDS, CURRENT_IS_DECODE_REQ, CURRENT_LOGITS_INDICES, CURRENT_NUM_ACTUAL_TOKENS
    CURRENT_REQ_IDS = req_ids
    CURRENT_IS_DECODE_REQ = is_decode_req
    CURRENT_LOGITS_INDICES = logits_indices.detach()
    CURRENT_NUM_ACTUAL_TOKENS = int(getattr(common, "num_actual_tokens", 0) or 0) if common else 0

    return out


GPUModelRunner.load_model = _wrapped_load_model
GPUModelRunner.execute_model = _wrapped_execute_model
GPUModelRunner._prepare_inputs = _wrapped_prepare_inputs


def _build_greedy_params(max_new_tokens: int, seed: int) -> SamplingParams:
    return SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_tokens=max_new_tokens,
        seed=seed,
    )


def _run_with_capture(llm: LLM, prompts: List[str], params: List[SamplingParams]) -> Dict[int, List[torch.Tensor]]:
    global CAPTURE_ACTIVE, REQID_TO_PROMPTIDX, CAPTURED
    global DEBUG_STEP, TAP_CALL_COUNT, TAP_LAST_SUM

    REQID_TO_PROMPTIDX = {}
    CAPTURED = {i: [] for i in range(len(prompts))}
    DEBUG_STEP = 0
    TAP_CALL_COUNT = 0
    TAP_LAST_SUM = float("nan")

    orig_add_request = llm.llm_engine.add_request
    counter = {"i": 0}

    # Keep insertion order even if request_id is not contiguous.
    def _wrapped_add_request(request_id, prompt, p, *args, **kwargs):
        REQID_TO_PROMPTIDX[request_id] = counter["i"]
        counter["i"] += 1
        return orig_add_request(request_id, prompt, p, *args, **kwargs)

    CAPTURE_ACTIVE = True
    llm.llm_engine.add_request = _wrapped_add_request  # type: ignore[assignment]
    try:
        llm.generate(prompts, params)
    finally:
        CAPTURE_ACTIVE = False
        llm.llm_engine.add_request = orig_add_request  # type: ignore[assignment]

    return CAPTURED
