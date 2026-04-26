#!/usr/bin/env python3
"""Integration-style tests for ESamp distiller sampling intervention."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig
from tllm.runtime.residual_runtime import SamplerPrecomputeState
from tllm.runtime.vllm_patch import sampler_patch


class _FakeTopKTopPSampler:
    def __call__(self, logits, generators, k, p):
        _ = (generators, k, p)
        return logits.argmax(dim=-1)

    def forward_native(self, logits, generators, k, p):
        return self(logits, generators, k, p)


class _FakeSampler:
    def __init__(self) -> None:
        self.topk_topp_sampler = _FakeTopKTopPSampler()

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)

    def apply_temperature(self, logits: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        return logits.div(temp.unsqueeze(dim=1))


class ESampDistillerSamplingIntegrationUnitTest(unittest.TestCase):
    def tearDown(self) -> None:
        sampler_patch._ORIG_VLLM_SAMPLER_SAMPLE = None

    def test_sampler_patch_uses_runtime_aligned_source_rows_for_esamp_provider(self) -> None:
        engine = mock.Mock()
        engine.predict_hidden_for_sampling.return_value = (
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[3_000_000.0, 0.0]], dtype=torch.float32),
        )
        consumer = ESampConsumer(
            ESampConsumerConfig(
                enable_esamp_training=True,
                enable_distiller_intervention=True,
                distiller_beta=1.0,
            ),
            engine=engine,
        )
        runtime = SimpleNamespace(
            consumer=consumer,
            event_step_id=12,
            sampler_precompute=SamplerPrecomputeState(port_publish_step_id=12),
            decode_count=2,
            decode_request_ids=["reqA", "reqB"],
            decode_prompt_idxs=[7, 8],
            decode_sample_idxs=[0, 1],
            source_resolved_path="layers.0",
            tap_decode_hidden={
                "layers.0": torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32)
            },
        )
        model = SimpleNamespace(
            lm_head=SimpleNamespace(
                weight=torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [0.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
                bias=torch.zeros((3,), dtype=torch.float32),
            )
        )
        sampler = _FakeSampler()
        runner = SimpleNamespace(model=model, sampler=sampler)
        sampler_patch.bind_runner_sampler(runtime=runtime, runner=runner)
        sampling_metadata = SimpleNamespace(
            all_random=False,
            all_greedy=False,
            temperature=torch.tensor([1e-6, 1e-6], dtype=torch.float32),
            logitsprocs=SimpleNamespace(argmax_invariant=[]),
            top_k=torch.tensor([1, 1], dtype=torch.int64),
            top_p=None,
            generators={},
        )
        logits = torch.tensor(
            [
                [0.9, 0.8, 0.1],
                [0.1, 0.2, 0.7],
            ],
            dtype=torch.float32,
        )

        sampled = sampler_patch.wrapped_sampler_sample(
            sampler=sampler,
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        engine.predict_hidden_for_sampling.assert_called_once()
        args = engine.predict_hidden_for_sampling.call_args.args
        self.assertTrue(torch.equal(args[0], torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32)))
        self.assertEqual(tuple(args[1]), (7, 8))
        self.assertTrue(torch.equal(sampled, torch.tensor([0, 2], dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
