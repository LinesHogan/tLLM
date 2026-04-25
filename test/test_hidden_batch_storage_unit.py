#!/usr/bin/env python3
"""Unit tests for HiddenBatch storage semantics."""

from __future__ import annotations

import unittest

import torch

from tllm.contracts.hidden_batch import HiddenBatch


class HiddenBatchStorageUnitTest(unittest.TestCase):
    def test_hidden_batch_docstring_mentions_non_cloning_contract(self) -> None:
        doc = HiddenBatch.__doc__ or ""
        self.assertIn("does not clone", doc)

    def test_decode_handoff_is_view_on_scratch_after_gather_copy(self) -> None:
        packed_hidden = torch.tensor(
            [
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
            ],
            dtype=torch.float32,
        )
        decode_row_idx = torch.tensor([2, 0, 1], dtype=torch.long)
        decode_valid_mask = torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float32)
        decode_buf = torch.empty((3, 2), dtype=torch.float32)

        torch.index_select(packed_hidden, 0, decode_row_idx, out=decode_buf)
        decode_buf.mul_(decode_valid_mask)

        batch = HiddenBatch(
            step_id=1,
            phase="decode",
            layer_path="layer",
            rows_hidden=decode_buf[:2],
            row_idx=decode_row_idx[:2],
            valid_mask=decode_valid_mask[:2, 0],
            prompt_idx=torch.tensor([7, 8], dtype=torch.long),
            sample_idx=torch.tensor([0, 0], dtype=torch.long),
            metadata={},
        )

        self.assertEqual(batch.rows_hidden.data_ptr(), decode_buf.data_ptr())
        self.assertNotEqual(batch.rows_hidden.data_ptr(), packed_hidden.data_ptr())
        self.assertTrue(torch.equal(batch.rows_hidden, torch.tensor([[30.0, 31.0], [10.0, 11.0]])))


if __name__ == "__main__":
    unittest.main()
