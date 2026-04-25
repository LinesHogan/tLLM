#!/usr/bin/env python3
"""Unit tests for vLLM prepare-inputs adapter selection and parsing."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch


class _Runner:
    def __init__(self) -> None:
        self.input_batch = SimpleNamespace(
            req_ids=["reqA", "reqB", None],
            num_reqs=2,
            req_id_to_index={"reqA": 0, "reqB": 1},
            num_prompt_tokens=[4, 5],
            num_computed_tokens_cpu=[4, 3],
        )


class VllmAdapterSelectionUnitTest(unittest.TestCase):
    def test_selects_expected_adapter_family(self) -> None:
        from tllm.runtime.vllm_patch.adapters import select_prepare_inputs_adapter

        cases = [
            ("0.7.2", "V072PrepareInputsAdapter"),
            ("0.7.9", "V072PrepareInputsAdapter"),
            ("0.10.2", "V010PrepareInputsAdapter"),
            ("0.11.0", "V011PlusPrepareInputsAdapter"),
            ("0.12.0", "V011PlusPrepareInputsAdapter"),
        ]

        for version, expected in cases:
            adapter = select_prepare_inputs_adapter(version)
            self.assertEqual(type(adapter).__name__, expected, msg=version)

    def test_adapter_parses_two_tuple_output(self) -> None:
        from tllm.runtime.vllm_patch.adapters import select_prepare_inputs_adapter

        adapter = select_prepare_inputs_adapter("0.7.2")
        runner = _Runner()
        attn_metadata = SimpleNamespace(num_actual_tokens=7)
        logits_indices = torch.tensor([1, 3], dtype=torch.long)
        out = (attn_metadata, logits_indices)

        view = adapter.unpack_prepare_inputs_output(
            runner=runner,
            scheduler_output=SimpleNamespace(num_scheduled_tokens={"reqA": 1, "reqB": 2}),
            out=out,
        )

        self.assertIs(view.attn_metadata, attn_metadata)
        self.assertTrue(torch.equal(view.logits_indices, logits_indices))
        self.assertIsNone(view.spec_decode_common)
        self.assertEqual(view.num_scheduled_tokens_np, [1, 2])


if __name__ == "__main__":
    unittest.main()
