#!/usr/bin/env python3
"""Unit tests for min-p candidate filtering helpers."""

from __future__ import annotations

import unittest
from unittest import mock

import pytest
import torch

from tllm.runtime.sampler_bridge.minp_kernel import (
    MinPCandidateKernelRequest,
    TritonMinPCandidateKernel,
    TorchMinPCandidateKernel,
    select_minp_candidate_kernel,
)
from tllm.runtime.sampler_bridge.min_p import apply_min_p, min_p_keep_mask
from tllm.runtime.sampler_bridge.types import CandidateModifierState


class SamplerBridgeMinPUnitTest(unittest.TestCase):
    def test_apply_min_p_keeps_tokens_above_scaled_max_probability(self) -> None:
        logits = torch.log(torch.tensor([[0.60, 0.30, 0.06, 0.04]], dtype=torch.float32))
        min_p = torch.tensor([0.2], dtype=torch.float32)

        filtered = apply_min_p(logits.clone(), min_p)

        self.assertTrue(torch.isfinite(filtered[0, 0]))
        self.assertTrue(torch.isfinite(filtered[0, 1]))
        self.assertFalse(torch.isfinite(filtered[0, 2]))
        self.assertFalse(torch.isfinite(filtered[0, 3]))

    def test_apply_min_p_keeps_at_least_one_token(self) -> None:
        logits = torch.tensor([[0.0, -100.0, -101.0]], dtype=torch.float32)
        min_p = torch.tensor([0.9], dtype=torch.float32)

        filtered = apply_min_p(logits.clone(), min_p)

        self.assertTrue(torch.isfinite(filtered[0, 0]))
        self.assertFalse(torch.isfinite(filtered[0, 1]))
        self.assertFalse(torch.isfinite(filtered[0, 2]))

    def test_apply_min_p_uses_logit_threshold_without_softmax(self) -> None:
        logits = torch.log(torch.tensor([[0.60, 0.30, 0.06, 0.04]], dtype=torch.float32))
        min_p = torch.tensor([0.2], dtype=torch.float32)

        with mock.patch.object(
            torch.Tensor,
            "softmax",
            side_effect=AssertionError("min-p should use the equivalent logit-space threshold"),
        ):
            filtered = apply_min_p(logits.clone(), min_p)

        self.assertTrue(torch.isfinite(filtered[0, 0]))
        self.assertTrue(torch.isfinite(filtered[0, 1]))
        self.assertFalse(torch.isfinite(filtered[0, 2]))
        self.assertFalse(torch.isfinite(filtered[0, 3]))

    def test_min_p_keep_mask_matches_apply_min_p_finite_positions(self) -> None:
        logits = torch.tensor([[2.0, 1.8, -5.0, -6.0], [0.1, 3.0, 2.9, -7.0]], dtype=torch.float32)
        min_p = torch.tensor([0.05, 0.05], dtype=torch.float32)

        mask = min_p_keep_mask(logits, min_p)
        filtered = apply_min_p(logits.clone(), min_p)

        self.assertTrue(torch.equal(mask, torch.isfinite(filtered)))

    def test_torch_minp_kernel_keep_mask_matches_reference(self) -> None:
        logits = torch.tensor([[2.0, 1.8, -5.0, -6.0], [0.1, 3.0, 2.9, -7.0]], dtype=torch.float32)
        min_p = torch.tensor([0.05, 0.05], dtype=torch.float32)

        kernel = TorchMinPCandidateKernel()

        self.assertTrue(torch.equal(kernel.keep_mask(logits, min_p), min_p_keep_mask(logits, min_p)))

    def test_torch_minp_kernel_greedy_samples_modified_candidates(self) -> None:
        logits = torch.tensor([[2.0, 1.9, -5.0]], dtype=torch.float32)
        state = CandidateModifierState(
            beta=1.0,
            backend="post_filter_exact",
            affected_row_ids=torch.tensor([0], dtype=torch.long),
            pred_hidden=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            lm_head_weight=torch.tensor(
                [
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [-100.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            lm_head_bias=None,
        )
        request = MinPCandidateKernelRequest(
            logits=logits,
            min_p=torch.tensor([0.5], dtype=torch.float32),
            state=state,
            greedy=True,
        )

        result = TorchMinPCandidateKernel().sample(request)

        self.assertTrue(torch.equal(result.sampled_token_ids, torch.tensor([0], dtype=torch.long)))
        assert result.debug_stats is not None
        self.assertEqual(result.debug_stats["kernel"], "torch")
        self.assertEqual(result.debug_stats["candidate_count"], 2)

    def test_torch_minp_kernel_random_sample_stays_inside_candidate_set(self) -> None:
        torch.manual_seed(1234)
        logits = torch.tensor([[2.0, 1.8, -5.0], [0.1, 3.0, 2.9]], dtype=torch.float32)
        state = CandidateModifierState(
            beta=0.0,
            backend="post_filter_exact",
            affected_row_ids=torch.tensor([0, 1], dtype=torch.long),
            pred_hidden=torch.zeros((2, 2), dtype=torch.float32),
            lm_head_weight=torch.zeros((3, 2), dtype=torch.float32),
            lm_head_bias=None,
        )
        request = MinPCandidateKernelRequest(
            logits=logits,
            min_p=torch.tensor([0.05, 0.05], dtype=torch.float32),
            state=state,
            greedy=False,
        )

        result = TorchMinPCandidateKernel().sample(request)
        mask = min_p_keep_mask(logits, request.min_p)

        self.assertEqual(tuple(result.sampled_token_ids.shape), (2,))
        for row_i, token_i in enumerate(result.sampled_token_ids.tolist()):
            self.assertTrue(bool(mask[row_i, token_i]))

    def test_minp_kernel_dispatcher_defaults_to_torch_reference(self) -> None:
        kernel = select_minp_candidate_kernel(prefer_triton=True, logits_device=torch.device("cpu"))

        self.assertIsInstance(kernel, TorchMinPCandidateKernel)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton min-p kernel tests")
    def test_triton_minp_keep_mask_matches_torch_reference_on_cuda(self) -> None:
        logits = torch.tensor(
            [[2.0, 1.8, -5.0, -6.0], [0.1, 3.0, 2.9, -7.0]],
            dtype=torch.float32,
            device="cuda",
        )
        min_p = torch.tensor([0.05, 0.05], dtype=torch.float32, device="cuda")

        kernel = TritonMinPCandidateKernel()
        mask = kernel.keep_mask(logits, min_p)

        self.assertTrue(torch.equal(mask.cpu(), min_p_keep_mask(logits, min_p).cpu()))
        self.assertTrue(bool(mask[0, 0]))
        self.assertTrue(bool(mask[1, 1]))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton min-p kernel tests")
    def test_triton_minp_kernel_random_sample_stays_inside_candidate_set_on_cuda(self) -> None:
        logits = torch.tensor(
            [[2.0, 1.8, -5.0], [0.1, 3.0, 2.9]],
            dtype=torch.float32,
            device="cuda",
        )
        state = CandidateModifierState(
            beta=0.0,
            backend="post_filter_exact",
            affected_row_ids=torch.tensor([0, 1], dtype=torch.long, device="cuda"),
            pred_hidden=torch.zeros((2, 2), dtype=torch.float32, device="cuda"),
            lm_head_weight=torch.zeros((3, 2), dtype=torch.float32, device="cuda"),
            lm_head_bias=None,
        )
        request = MinPCandidateKernelRequest(
            logits=logits,
            min_p=torch.tensor([0.05, 0.05], dtype=torch.float32, device="cuda"),
            state=state,
            greedy=False,
        )

        result = TritonMinPCandidateKernel().sample(request)
        mask = min_p_keep_mask(logits, request.min_p)

        self.assertEqual(tuple(result.sampled_token_ids.shape), (2,))
        assert result.debug_stats is not None
        self.assertIn(result.debug_stats["kernel"], {"triton_minp_keep_mask", "torch"})
        for row_i, token_i in enumerate(result.sampled_token_ids.cpu().tolist()):
            self.assertTrue(bool(mask[row_i, token_i]))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton min-p kernel tests")
    def test_minp_kernel_dispatcher_prefers_triton_on_cuda(self) -> None:
        kernel = select_minp_candidate_kernel(prefer_triton=True, logits_device=torch.device("cuda"))

        self.assertIsInstance(kernel, TritonMinPCandidateKernel)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton min-p kernel tests")
    def test_triton_minp_keep_mask_supports_qwen_vocab_shape_on_cuda(self) -> None:
        vocab = 151936
        logits = torch.full((1, vocab), -10.0, dtype=torch.float32, device="cuda")
        logits[0, 17] = 3.0
        logits[0, 23] = 2.95
        min_p = torch.tensor([0.05], dtype=torch.float32, device="cuda")

        kernel = select_minp_candidate_kernel(
            prefer_triton=True,
            logits_device=logits.device,
            logits=logits,
        )
        mask = kernel.keep_mask(logits, min_p)

        self.assertIsInstance(kernel, TritonMinPCandidateKernel)
        self.assertTrue(bool(mask[0, 17]))
        self.assertTrue(bool(mask[0, 23]))
        self.assertFalse(bool(mask[0, 12345]))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton min-p kernel tests")
    def test_minp_kernel_dispatcher_falls_back_before_runtime_exception_for_large_vocab(self) -> None:
        logits = torch.empty((1, 300000), dtype=torch.float32, device="cuda")

        kernel = select_minp_candidate_kernel(
            prefer_triton=True,
            logits_device=logits.device,
            logits=logits,
        )

        self.assertIsInstance(kernel, TorchMinPCandidateKernel)


if __name__ == "__main__":
    unittest.main()
