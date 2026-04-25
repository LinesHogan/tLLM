#!/usr/bin/env python3
"""Versioned vLLM prepare-inputs adapter helpers."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PrepareInputsView:
    attn_metadata: Any
    logits_indices: torch.Tensor
    spec_decode_common: Any
    num_scheduled_tokens_np: Any


class BasePrepareInputsAdapter(ABC):
    family_name = "base"

    def unpack_prepare_inputs_output(
        self,
        *,
        runner: Any,
        scheduler_output: Any,
        out: tuple,
    ) -> PrepareInputsView:
        if not isinstance(out, tuple):
            raise TypeError(f"prepare_inputs output must be a tuple, got {type(out).__name__}")

        if len(out) >= 6:
            return PrepareInputsView(
                attn_metadata=out[0],
                logits_indices=out[1],
                spec_decode_common=out[4],
                num_scheduled_tokens_np=out[3],
            )

        if len(out) >= 2:
            attn_metadata = out[0]
            logits_indices = out[1]
            spec_decode_common = None
            num_scheduled_map = getattr(scheduler_output, "num_scheduled_tokens", None)
            if isinstance(num_scheduled_map, dict):
                req_ids = [rid for rid in runner.input_batch.req_ids[: runner.input_batch.num_reqs] if rid is not None]
                num_scheduled_tokens_np = [int(num_scheduled_map.get(rid, 0)) for rid in req_ids]
            else:
                num_scheduled_tokens_np = num_scheduled_map
            return PrepareInputsView(
                attn_metadata=attn_metadata,
                logits_indices=logits_indices,
                spec_decode_common=spec_decode_common,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
            )

        raise ValueError(f"prepare_inputs output is too short: len={len(out)}")
