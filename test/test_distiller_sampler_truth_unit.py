#!/usr/bin/env python3
"""Unit tests for distiller sampler truth helpers."""

from __future__ import annotations

import unittest

import torch

from tllm.runtime.sampler_bridge.truth import (
    apply_candidate_intervention,
    project_candidate_logits,
    select_candidate_pairs,
)


class DistillerSamplerTruthUnitTest(unittest.TestCase):
    def test_candidate_only_formula_rewrites_only_surviving_tokens(self) -> None:
        logits = torch.tensor(
            [
                [0.1, float("-inf"), 0.4, float("-inf")],
                [float("-inf"), 0.2, float("-inf"), 0.8],
            ],
            dtype=torch.float32,
        )
        pred_hidden = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=torch.float32,
        )
        lm_head_weight = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ],
            dtype=torch.float32,
        )
        lm_head_bias = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float32)

        row_ids, token_ids = select_candidate_pairs(logits)
        candidate_logits = project_candidate_logits(
            pred_hidden=pred_hidden,
            row_ids=row_ids,
            token_ids=token_ids,
            lm_head_weight=lm_head_weight,
            lm_head_bias=lm_head_bias,
        )
        out = apply_candidate_intervention(
            logits=logits,
            row_ids=row_ids,
            token_ids=token_ids,
            distiller_candidate_logits=candidate_logits,
            beta=0.5,
        )

        expected = logits.clone()
        expected[0, 0] = (1.5 * 0.1) - 0.5 * 1.0
        expected[0, 2] = (1.5 * 0.4) - 0.5 * 3.2
        expected[1, 1] = (1.5 * 0.2) - 0.5 * 4.1
        expected[1, 3] = (1.5 * 0.8) - 0.5 * 10.3

        self.assertTrue(torch.equal(torch.isfinite(out), torch.isfinite(logits)))
        self.assertTrue(torch.allclose(out, expected))

    def test_unaffected_rows_are_left_unchanged(self) -> None:
        logits = torch.tensor(
            [
                [0.1, float("-inf"), 0.4],
                [0.2, 0.3, float("-inf")],
            ],
            dtype=torch.float32,
        )
        pred_hidden = torch.tensor([[5.0, 6.0]], dtype=torch.float32)
        lm_head_weight = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        )

        row_ids = torch.tensor([0, 0], dtype=torch.long)
        token_ids = torch.tensor([0, 2], dtype=torch.long)
        candidate_logits = project_candidate_logits(
            pred_hidden=pred_hidden,
            row_ids=row_ids,
            token_ids=token_ids,
            lm_head_weight=lm_head_weight,
            lm_head_bias=None,
        )
        out = apply_candidate_intervention(
            logits=logits,
            row_ids=row_ids,
            token_ids=token_ids,
            distiller_candidate_logits=candidate_logits,
            beta=0.25,
        )

        self.assertTrue(torch.allclose(out[1], logits[1], equal_nan=True))

    def test_beta_zero_is_exact_noop(self) -> None:
        logits = torch.tensor([[0.1, float("-inf"), 0.4]], dtype=torch.float32)
        row_ids = torch.tensor([0, 0], dtype=torch.long)
        token_ids = torch.tensor([0, 2], dtype=torch.long)
        distiller_candidate_logits = torch.tensor([9.0, 8.0], dtype=torch.float32)

        out = apply_candidate_intervention(
            logits=logits,
            row_ids=row_ids,
            token_ids=token_ids,
            distiller_candidate_logits=distiller_candidate_logits,
            beta=0.0,
        )

        self.assertTrue(torch.equal(out, logits))


if __name__ == "__main__":
    unittest.main()
