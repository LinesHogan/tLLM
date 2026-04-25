#!/usr/bin/env python3
"""Unit tests for runtime token map extraction helpers."""

from __future__ import annotations

import unittest

from tllm.runtime.token_map import build_token_maps_from_outputs


class _Cand:
    def __init__(self, token_ids, index=None):
        self.token_ids = token_ids
        self.index = index


class _Out:
    def __init__(self, request_id, outputs):
        self.request_id = request_id
        self.outputs = outputs


class RuntimeTokenMapUnitTest(unittest.TestCase):
    def test_build_token_maps_supports_n_greater_than_one(self) -> None:
        outputs = [
            _Out("reqA", [_Cand([1, 2], index=0), _Cand([3], index=1)]),
            _Out("2_reqA", [_Cand([7, 8])]),
            _Out("reqB", [_Cand([9], index=0)]),
        ]

        def _resolver(req_id: str) -> tuple[int, int]:
            if req_id == "reqA":
                return 0, 0
            if req_id == "2_reqA":
                return 0, 2
            if req_id == "reqB":
                return 1, 0
            return -1, -1

        by_prompt, by_prompt_sample = build_token_maps_from_outputs(outputs, 2, _resolver)
        self.assertEqual(by_prompt[0], [1, 2])
        self.assertEqual(by_prompt[1], [9])
        self.assertEqual(by_prompt_sample[0][1], [3])
        self.assertEqual(by_prompt_sample[0][2], [7, 8])


if __name__ == "__main__":
    unittest.main()
