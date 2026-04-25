#!/usr/bin/env python3
"""Unit tests for extracted runtime step snapshot helper."""

from __future__ import annotations

import unittest

import torch

from tllm.common.runtime_step_state import (
    RuntimeStepState,
    snapshot_step_common,
)


class _InputBatch:
    def __init__(self) -> None:
        self.req_ids = ["reqA", "reqB", None]
        self.num_reqs = 2
        self.req_id_to_index = {"reqA": 0, "reqB": 1}
        self.num_prompt_tokens = [4, 6]
        self.num_computed_tokens_cpu = [4, 1]


class _Runner:
    def __init__(self) -> None:
        self.input_batch = _InputBatch()


class _Common:
    num_actual_tokens = 10


class RuntimeStepStateUnitTest(unittest.TestCase):
    def test_snapshot_step_common_extracts_decode_flags_and_ids(self) -> None:
        step = RuntimeStepState()
        runner = _Runner()
        logits = torch.tensor([1, 2], dtype=torch.long)

        snapshot_step_common(
            step=step,
            runner=runner,
            common_attn_metadata=_Common(),
            logits_indices=logits,
            num_scheduled_tokens_np=[1, 1],
        )

        self.assertEqual(step.req_ids, ["reqA", "reqB"])
        self.assertEqual(step.is_decode_req, [True, False])
        self.assertEqual(step.num_actual_tokens, 10)
        self.assertTrue(torch.equal(step.logits_indices, logits))


if __name__ == "__main__":
    unittest.main()
