#!/usr/bin/env python3
"""Helpers for extracting per-prompt and per-sample token maps from vLLM outputs."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple


def build_token_maps_from_outputs(
    outputs: Any,
    prompt_count: int,
    resolve_prompt_sample_fn: Callable[[str], tuple[int, int]],
) -> Tuple[Dict[int, List[int]], Dict[int, Dict[int, List[int]]]]:
    token_ids_by_prompt: Dict[int, List[int]] = {i: [] for i in range(int(prompt_count))}
    token_ids_by_prompt_sample: Dict[int, Dict[int, List[int]]] = {
        i: {} for i in range(int(prompt_count))
    }
    if outputs is None:
        return token_ids_by_prompt, token_ids_by_prompt_sample

    for out in outputs:
        req_id = getattr(out, "request_id", None)
        if req_id is None:
            continue
        prompt_idx, sample_idx_base = resolve_prompt_sample_fn(req_id)
        if prompt_idx < 0:
            continue

        pidx = int(prompt_idx)
        out_list = getattr(out, "outputs", None)
        if out_list:
            for j, cand in enumerate(out_list):
                token_ids = getattr(cand, "token_ids", None)
                if token_ids is None:
                    continue
                seq = [int(x) for x in token_ids]

                cand_index = getattr(cand, "index", None)
                if cand_index is None:
                    # child-request path usually has one candidate only
                    # and sample index encoded in request_id.
                    sidx = int(sample_idx_base if len(out_list) == 1 else j)
                else:
                    sidx = int(cand_index)

                token_ids_by_prompt_sample[pidx][sidx] = seq
                if sidx == 0:
                    token_ids_by_prompt[pidx] = seq
        else:
            token_ids_by_prompt[pidx] = []

    return token_ids_by_prompt, token_ids_by_prompt_sample
