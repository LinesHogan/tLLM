#!/usr/bin/env python3
"""Unit tests for ESamp loss parity gate helpers."""

from __future__ import annotations

import unittest

from tllm.workflows.benchmarks import per_request_side_train_benchmark as bench


class ESampLossParityContractUnitTest(unittest.TestCase):
    def test_loss_parity_gate_requires_non_zero_updates_and_small_deltas(self) -> None:
        summary = {
            "loss_deltas": {"single_on": 0.01, "per_request_on": -0.01},
            "loss_count_deltas": {"single_on": 0.0, "per_request_on": 0.0},
            "legacy": {
                "single_on": {"loss_count": 10},
                "per_request_on": {"loss_count": 20},
            },
            "base_consumer": {
                "single_on": {"loss_count": 10},
                "per_request_on": {"loss_count": 20},
            },
        }

        self.assertTrue(bench._loss_parity_passes(summary, abs_loss_tol=0.05))

    def test_loss_parity_gate_rejects_zero_training_updates(self) -> None:
        summary = {
            "loss_deltas": {"single_on": 0.0},
            "loss_count_deltas": {"single_on": 0.0},
            "legacy": {"single_on": {"loss_count": 10}},
            "base_consumer": {"single_on": {"loss_count": 0}},
        }

        self.assertFalse(bench._loss_parity_passes(summary, abs_loss_tol=0.05))

    def test_loss_parity_gate_rejects_mismatched_update_counts(self) -> None:
        summary = {
            "loss_deltas": {"single_on": 0.0},
            "loss_count_deltas": {"single_on": -1.0},
            "legacy": {"single_on": {"loss_count": 10}},
            "base_consumer": {"single_on": {"loss_count": 9}},
        }

        self.assertFalse(bench._loss_parity_passes(summary, abs_loss_tol=0.05))


if __name__ == "__main__":
    unittest.main()
