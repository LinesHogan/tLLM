#!/usr/bin/env python3
"""Shared helpers for validating active decode metadata slices."""

from __future__ import annotations

from typing import Protocol


class DecodeMetadataRuntime(Protocol):
    decode_request_ids: list[str]
    decode_prompt_idxs: list[int]
    decode_sample_idxs: list[int]


def active_request_prompt_sample_metadata(
    runtime: DecodeMetadataRuntime,
    active: int,
) -> tuple[list[str], list[int], list[int]]:
    request_ids = list(runtime.decode_request_ids[:active])
    prompt_idxs = list(runtime.decode_prompt_idxs[:active])
    sample_idxs = list(runtime.decode_sample_idxs[:active])
    if len(request_ids) != active or len(prompt_idxs) != active or len(sample_idxs) != active:
        raise RuntimeError(
            "decode runtime metadata is inconsistent: "
            f"active={active} request_ids={len(request_ids)} prompt_idxs={len(prompt_idxs)} sample_idxs={len(sample_idxs)}"
        )
    return request_ids, prompt_idxs, sample_idxs
