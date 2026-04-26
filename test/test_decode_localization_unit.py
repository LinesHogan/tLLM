#!/usr/bin/env python3
"""Unit tests for pure decode-localization logic."""

from __future__ import annotations

import unittest
from typing import Dict, Tuple

import torch

from tllm.producer.decode import (
    DECODE_HIDDEN_ROWS_BUFFER,
    compute_decode_localization,
)


def _make_resolver(base_map: Dict[str, int]):
    def _resolver(req_id: str) -> Tuple[int, int]:
        prompt_idx = base_map.get(req_id)
        if prompt_idx is not None:
            return int(prompt_idx), 0
        if "_" in req_id:
            maybe_sample_idx, parent_req_id = req_id.split("_", 1)
            if maybe_sample_idx.isdigit():
                parent_prompt_idx = base_map.get(parent_req_id)
                if parent_prompt_idx is not None:
                    return int(parent_prompt_idx), int(maybe_sample_idx)
        return -1, -1

    return _resolver


class DecodeLocalizationUnitTest(unittest.TestCase):
    def test_decode_hidden_rows_buffer_name_is_descriptive(self) -> None:
        self.assertEqual(DECODE_HIDDEN_ROWS_BUFFER, "decode_hidden_rows_buffer")

    def test_n_greater_than_one_decode_mapping(self) -> None:
        req_ids = ["reqA", "1_reqA", "2_reqA", "reqB", "1_reqB"]
        is_decode_req = [False, True, True, True, False]
        logits_indices = torch.tensor([5, 9, 12, 21, 25], dtype=torch.long)
        resolver = _make_resolver({"reqA": 0, "reqB": 1})

        row_idx, prompt_idxs, sample_idxs, decode_positions = compute_decode_localization(
            req_ids=req_ids,
            is_decode_req=is_decode_req,
            logits_indices=logits_indices,
            num_actual_tokens=32,
            resolve_prompt_sample_fn=resolver,
        )

        self.assertEqual(decode_positions, [1, 2, 3])
        self.assertEqual(row_idx.tolist(), [9, 12, 21])
        self.assertEqual(prompt_idxs, [0, 0, 1])
        self.assertEqual(sample_idxs, [1, 2, 0])

    def test_out_of_range_logits_index_raises(self) -> None:
        req_ids = ["reqA"]
        is_decode_req = [True]
        logits_indices = torch.tensor([99], dtype=torch.long)
        resolver = _make_resolver({"reqA": 0})

        with self.assertRaisesRegex(RuntimeError, "out of range"):
            compute_decode_localization(
                req_ids=req_ids,
                is_decode_req=is_decode_req,
                logits_indices=logits_indices,
                num_actual_tokens=50,
                resolve_prompt_sample_fn=resolver,
            )

    def test_empty_decode_returns_empty_outputs(self) -> None:
        req_ids = ["reqA", "reqB"]
        is_decode_req = [False, False]
        logits_indices = torch.tensor([3, 7], dtype=torch.long)
        resolver = _make_resolver({"reqA": 0, "reqB": 1})

        row_idx, prompt_idxs, sample_idxs, decode_positions = compute_decode_localization(
            req_ids=req_ids,
            is_decode_req=is_decode_req,
            logits_indices=logits_indices,
            num_actual_tokens=8,
            resolve_prompt_sample_fn=resolver,
        )

        self.assertEqual(row_idx.numel(), 0)
        self.assertEqual(prompt_idxs, [])
        self.assertEqual(sample_idxs, [])
        self.assertEqual(decode_positions, [])

    def test_decode_mask_longer_than_req_ids_is_safely_bounded(self) -> None:
        req_ids = ["reqA", "1_reqA"]
        is_decode_req = [True, True, True]
        logits_indices = torch.tensor([4, 8], dtype=torch.long)
        resolver = _make_resolver({"reqA": 3})

        row_idx, prompt_idxs, sample_idxs, decode_positions = compute_decode_localization(
            req_ids=req_ids,
            is_decode_req=is_decode_req,
            logits_indices=logits_indices,
            num_actual_tokens=16,
            resolve_prompt_sample_fn=resolver,
        )

        self.assertEqual(decode_positions, [0, 1])
        self.assertEqual(row_idx.tolist(), [4, 8])
        self.assertEqual(prompt_idxs, [3, 3])
        self.assertEqual(sample_idxs, [0, 1])

    def test_contiguous_decode_positions_return_logits_indices_view(self) -> None:
        req_ids = ["prefill", "reqA", "1_reqA", "reqB"]
        is_decode_req = [False, True, True, True]
        logits_indices = torch.tensor([3, 5, 9, 12], dtype=torch.long)
        resolver = _make_resolver({"reqA": 0, "reqB": 1})

        row_idx, prompt_idxs, sample_idxs, decode_positions = compute_decode_localization(
            req_ids=req_ids,
            is_decode_req=is_decode_req,
            logits_indices=logits_indices,
            num_actual_tokens=16,
            resolve_prompt_sample_fn=resolver,
        )

        self.assertEqual(decode_positions, [1, 2, 3])
        self.assertEqual(row_idx.tolist(), [5, 9, 12])
        self.assertEqual(prompt_idxs, [0, 0, 1])
        self.assertEqual(sample_idxs, [0, 1, 0])
        self.assertEqual(row_idx.untyped_storage().data_ptr(), logits_indices.untyped_storage().data_ptr())
        self.assertEqual(row_idx.storage_offset(), 1)

    def test_max_decode_rows_stops_before_resolving_unused_rows(self) -> None:
        req_ids = ["reqA", "reqB", "reqC"]
        is_decode_req = [True, True, True]
        logits_indices = torch.tensor([4, 8, 12], dtype=torch.long)
        resolved: list[str] = []

        def _resolver(req_id: str) -> Tuple[int, int]:
            resolved.append(req_id)
            return len(resolved) - 1, 0

        row_idx, prompt_idxs, sample_idxs, decode_positions = compute_decode_localization(
            req_ids=req_ids,
            is_decode_req=is_decode_req,
            logits_indices=logits_indices,
            num_actual_tokens=16,
            resolve_prompt_sample_fn=_resolver,
            max_decode_rows=1,
        )

        self.assertEqual(decode_positions, [0])
        self.assertEqual(row_idx.tolist(), [4])
        self.assertEqual(prompt_idxs, [0])
        self.assertEqual(sample_idxs, [0])
        self.assertEqual(resolved, ["reqA"])


if __name__ == "__main__":
    unittest.main()
