#!/usr/bin/env python3
"""Typed row metadata for high-frequency bundle delivery."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, init=False)
class RowBatchMeta:
    request_ids: tuple[str, ...]
    prompt_idxs: tuple[int, ...]
    sample_idxs: tuple[int, ...]
    phase: str
    engine_step_id: int
    row_compaction: str = "none"
    row_ids: tuple[int, ...] = ()

    def __init__(
        self,
        *,
        request_ids: tuple[str, ...],
        prompt_idxs: tuple[int, ...],
        sample_idxs: tuple[int, ...],
        phase: str,
        engine_step_id: int,
        row_compaction: str = "none",
        row_ids: tuple[int, ...] = (),
    ) -> None:
        object.__setattr__(self, "request_ids", tuple(request_ids))
        object.__setattr__(self, "prompt_idxs", tuple(int(v) for v in prompt_idxs))
        object.__setattr__(self, "sample_idxs", tuple(int(v) for v in sample_idxs))
        object.__setattr__(self, "phase", str(phase))
        object.__setattr__(self, "engine_step_id", int(engine_step_id))
        object.__setattr__(self, "row_compaction", str(row_compaction))
        object.__setattr__(self, "row_ids", tuple(int(v) for v in row_ids))
        self.__post_init__()

    def __post_init__(self) -> None:
        sizes = {len(self.request_ids), len(self.prompt_idxs), len(self.sample_idxs)}
        if len(sizes) != 1:
            raise ValueError("RowBatchMeta request_ids, prompt_idxs, and sample_idxs must have the same length")
        if self.row_ids and len(self.row_ids) != len(self.request_ids):
            raise ValueError("RowBatchMeta row_ids must have the same length as request metadata")

    def as_legacy_dicts(self) -> list[dict[str, object]]:
        return [
            {
                "request_id": request_id,
                "prompt_idx": int(prompt_idx),
                "sample_idx": int(sample_idx),
                "phase": self.phase,
                "engine_step_id": int(self.engine_step_id),
            }
            for request_id, prompt_idx, sample_idx in zip(self.request_ids, self.prompt_idxs, self.sample_idxs)
        ]
