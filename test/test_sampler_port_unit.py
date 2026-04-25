#!/usr/bin/env python3
"""Unit tests for the sampler port surface."""

from __future__ import annotations

import unittest

import torch

from tllm.ports.sampler import CandidateModifierProvider, Sampler, SamplerLocator
from tllm.runtime.sampler_bridge.types import CandidateModifierRequest, CandidateModifierState, SamplerFilterSpec


class SamplerPortUnitTest(unittest.TestCase):
    def test_sampler_port_is_public_and_decode_scoped(self) -> None:
        self.assertTrue(Sampler.READABLE)
        self.assertFalse(Sampler.WRITABLE)
        self.assertEqual(Sampler.KIND.value, "sampler")
        self.assertEqual(Sampler.SUPPORTED_PHASES, ("decode",))
        self.assertIn("same-step", Sampler.BACKING_VLLM_STRUCT)

    def test_sampler_read_builds_typed_locator(self) -> None:
        read_spec = Sampler.read(step_scope="current")

        self.assertIsInstance(read_spec.locator, SamplerLocator)
        self.assertEqual(read_spec.locator.step_scope, "current")

    def test_candidate_modifier_provider_is_runtime_checkable_public_contract(self) -> None:
        class Provider:
            def is_active(self) -> bool:
                return True

            def prepare_candidate_state(
                self,
                request: CandidateModifierRequest,
            ) -> CandidateModifierState | None:
                return CandidateModifierState(
                    beta=0.1,
                    backend="post_filter_exact",
                    affected_row_ids=request.affected_row_ids,
                    pred_hidden=torch.ones((1, 2), dtype=torch.float32),
                    lm_head_weight=torch.ones((4, 2), dtype=torch.float32),
                    lm_head_bias=None,
                )

        request = CandidateModifierRequest(
            logits=torch.zeros((1, 4), dtype=torch.float32),
            affected_row_ids=torch.tensor([0], dtype=torch.long),
            filter_spec=SamplerFilterSpec(
                top_k=None,
                top_p=None,
                min_p=torch.tensor([0.05], dtype=torch.float32),
                temperature_mode="all_random",
                has_generators=False,
            ),
            sampling_mode="random",
        )

        provider = Provider()

        self.assertIsInstance(provider, CandidateModifierProvider)
        self.assertIsNotNone(provider.prepare_candidate_state(request))


if __name__ == "__main__":
    unittest.main()
