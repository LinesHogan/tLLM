#!/usr/bin/env python3
"""Shared helpers for workflow scripts."""

from __future__ import annotations

from typing import List, Sequence

try:
    from vllm import SamplingParams as _SamplingParams
except Exception:  # pragma: no cover - fallback for unit-only environments
    class _SamplingParams:  # type: ignore[override]
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


def sum_all_candidate_tokens(outputs) -> int:
    total = 0
    for out in outputs:
        for cand in (getattr(out, "outputs", None) or []):
            token_ids = getattr(cand, "token_ids", None)
            if token_ids is None:
                continue
            total += len(token_ids)
    return int(total)


def sum_all_completions(outputs) -> int:
    total = 0
    for out in outputs:
        total += len(getattr(out, "outputs", None) or [])
    return int(total)


def build_sampling_params(
    *,
    prompts: Sequence[str],
    max_new_tokens: int,
    sampling_n: int,
    sampling_temperature: float,
    sampling_top_p: float,
    sampling_top_k: int,
    sampling_min_p: float = 0.0,
    ignore_eos: bool,
    sampling_seed: int | None,
    sampling_per_request_seed: bool,
) -> List[_SamplingParams]:
    common_kwargs = dict(
        n=int(sampling_n),
        temperature=float(sampling_temperature),
        top_p=float(sampling_top_p),
        top_k=int(sampling_top_k),
        min_p=float(sampling_min_p),
        max_tokens=int(max_new_tokens),
        min_tokens=(int(max_new_tokens) if ignore_eos else 0),
        ignore_eos=bool(ignore_eos),
    )
    if not bool(sampling_per_request_seed):
        seed_val = None if sampling_seed is None else int(sampling_seed)
        return [_SamplingParams(seed=seed_val, **common_kwargs) for _ in range(len(prompts))]

    params: List[_SamplingParams] = []
    for i in range(len(prompts)):
        seed_i = None if sampling_seed is None else int(sampling_seed) + i
        params.append(_SamplingParams(seed=seed_i, **common_kwargs))
    return params
