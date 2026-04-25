#!/usr/bin/env python3
"""Unit tests for shared workflow helper functions."""

from __future__ import annotations

import unittest

from tllm.workflows.common import (
    build_sampling_params,
    sum_all_candidate_tokens,
    sum_all_completions,
)


class _Cand:
    def __init__(self, token_ids):
        self.token_ids = token_ids


class _Out:
    def __init__(self, outputs):
        self.outputs = outputs


class RunnerCommonUnitTest(unittest.TestCase):
    def test_sum_helpers(self) -> None:
        outputs = [_Out([_Cand([1, 2]), _Cand([3])]), _Out([_Cand([4, 5, 6])])]
        self.assertEqual(sum_all_candidate_tokens(outputs), 6)
        self.assertEqual(sum_all_completions(outputs), 3)

    def test_build_sampling_params_with_per_request_seed(self) -> None:
        params = build_sampling_params(
            prompts=["a", "b", "c"],
            max_new_tokens=4,
            sampling_n=2,
            sampling_temperature=0.8,
            sampling_top_p=0.95,
            sampling_top_k=-1,
            sampling_min_p=0.0,
            ignore_eos=True,
            sampling_seed=100,
            sampling_per_request_seed=True,
        )
        seeds = [getattr(p, "seed", None) for p in params]
        self.assertEqual(seeds, [100, 101, 102])

    def test_build_sampling_params_records_min_p(self) -> None:
        params = build_sampling_params(
            prompts=["a"],
            max_new_tokens=4,
            sampling_n=1,
            sampling_temperature=0.8,
            sampling_top_p=1.0,
            sampling_top_k=-1,
            sampling_min_p=0.12,
            ignore_eos=True,
            sampling_seed=None,
            sampling_per_request_seed=False,
        )

        self.assertAlmostEqual(float(getattr(params[0], "min_p", 0.0)), 0.12)


if __name__ == "__main__":
    unittest.main()
