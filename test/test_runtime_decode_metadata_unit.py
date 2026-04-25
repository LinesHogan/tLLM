#!/usr/bin/env python3
"""Unit tests for decode runtime metadata helpers."""

from __future__ import annotations

import unittest

from tllm.runtime import decode_runtime_metadata


class RuntimeDecodeMetadataUnitTest(unittest.TestCase):
    def test_active_request_prompt_sample_metadata_returns_aligned_lists(self) -> None:
        runtime = type(
            "Runtime",
            (),
            {
                "decode_request_ids": ["reqA", "reqB", "reqC"],
                "decode_prompt_idxs": [1, 2, 3],
                "decode_sample_idxs": [0, 1, 2],
            },
        )()

        request_ids, prompt_idxs, sample_idxs = decode_runtime_metadata.active_request_prompt_sample_metadata(runtime, 2)

        self.assertEqual(request_ids, ["reqA", "reqB"])
        self.assertEqual(prompt_idxs, [1, 2])
        self.assertEqual(sample_idxs, [0, 1])

    def test_active_request_prompt_sample_metadata_rejects_inconsistent_lengths(self) -> None:
        runtime = type(
            "Runtime",
            (),
            {
                "decode_request_ids": ["reqA"],
                "decode_prompt_idxs": [1, 2],
                "decode_sample_idxs": [0, 1],
            },
        )()

        with self.assertRaisesRegex(RuntimeError, "decode runtime metadata is inconsistent"):
            decode_runtime_metadata.active_request_prompt_sample_metadata(runtime, 2)


if __name__ == "__main__":
    unittest.main()
