#!/usr/bin/env python3
"""Unit tests for tLLM's V1 InputBatch min-p metadata patch."""

from __future__ import annotations

import unittest

import torch
from vllm import SamplingParams

from tllm.runtime.vllm_patch.port_runtime_hooks import install_input_batch_min_p_patch


class VllmInputBatchMinPPatchUnitTest(unittest.TestCase):
    def test_patch_exposes_min_p_on_sampling_metadata(self) -> None:
        install_input_batch_min_p_patch()
        from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

        batch = InputBatch(
            max_num_reqs=2,
            max_model_len=16,
            max_num_batched_tokens=16,
            device=torch.device("cpu"),
            pin_memory=False,
            vocab_size=10,
            block_sizes=[16],
        )
        request = CachedRequestState(
            req_id="req0",
            prompt_token_ids=[1, 2],
            mm_kwargs=[],
            mm_positions=[],
            sampling_params=SamplingParams(
                temperature=0.8,
                top_p=1.0,
                top_k=-1,
                min_p=0.2,
                max_tokens=1,
            ),
            pooling_params=None,
            generator=None,
            block_ids=([],),
            num_computed_tokens=2,
            output_token_ids=[],
        )

        batch.add_request(request)
        metadata = batch._make_sampling_metadata()

        self.assertIsInstance(getattr(metadata, "min_p", None), torch.Tensor)
        self.assertAlmostEqual(float(metadata.min_p[0]), 0.2, places=5)


if __name__ == "__main__":
    unittest.main()
