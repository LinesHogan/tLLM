#!/usr/bin/env python3
"""Per-step runtime snapshot helpers shared by producer pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class RuntimeStepState:
    req_ids: List[str] = field(default_factory=list)
    req_id_to_index: Dict[str, int] = field(default_factory=dict)
    is_decode_req: List[bool] = field(default_factory=list)
    logits_indices: Optional[torch.Tensor] = None
    num_actual_tokens: int = 0
    num_prompt_tokens: Any = None
    num_computed_tokens: Any = None
    num_scheduled_tokens_np: Any = None

    decode_count: int = 0
    decode_prompt_idxs: List[int] = field(default_factory=list)
    decode_sample_idxs: List[int] = field(default_factory=list)

    prefill_count: int = 0
    prefill_row_indices: List[int] = field(default_factory=list)
    prefill_prompt_idxs: List[int] = field(default_factory=list)
    prefill_sample_idxs: List[int] = field(default_factory=list)
    prefill_token_offsets: List[int] = field(default_factory=list)

    def reset_localization_outputs(self) -> None:
        self.decode_count = 0
        self.decode_prompt_idxs = []
        self.decode_sample_idxs = []
        self.prefill_count = 0
        self.prefill_row_indices = []
        self.prefill_prompt_idxs = []
        self.prefill_sample_idxs = []
        self.prefill_token_offsets = []


def snapshot_step_common(
    *,
    step: RuntimeStepState,
    runner: Any,
    common_attn_metadata: Any,
    logits_indices: torch.Tensor,
    num_scheduled_tokens_np: Any,
) -> None:
    """Capture request-order and packed-row metadata from `_prepare_inputs`."""
    req_ids = [rid for rid in runner.input_batch.req_ids[: runner.input_batch.num_reqs] if rid is not None]
    req_id_to_index = dict(runner.input_batch.req_id_to_index)
    num_prompt_tokens = runner.input_batch.num_prompt_tokens
    num_computed_tokens = runner.input_batch.num_computed_tokens_cpu

    is_decode_req: List[bool] = []
    for req_id in req_ids:
        req_idx = req_id_to_index.get(req_id)
        if req_idx is None:
            is_decode_req.append(False)
            continue
        is_decode_req.append(int(num_computed_tokens[req_idx]) >= int(num_prompt_tokens[req_idx]))

    step.req_ids = req_ids
    step.req_id_to_index = req_id_to_index
    step.is_decode_req = is_decode_req
    step.logits_indices = logits_indices.detach() if logits_indices is not None else None
    step.num_actual_tokens = int(getattr(common_attn_metadata, "num_actual_tokens", 0) or 0) if common_attn_metadata else 0
    step.num_prompt_tokens = num_prompt_tokens
    step.num_computed_tokens = num_computed_tokens
    step.num_scheduled_tokens_np = num_scheduled_tokens_np
    step.reset_localization_outputs()
