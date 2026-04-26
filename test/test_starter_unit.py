#!/usr/bin/env python3
"""Unit tests for the top-level ESamp starter example."""

from __future__ import annotations

import importlib
import sys
import unittest
from unittest import mock


class StarterUnitTest(unittest.TestCase):
    def _load_starter(self):
        sys.modules.pop("starter", None)
        return importlib.import_module("starter")

    def test_defaults_target_configured_model_and_sixteen_parallel_answers(self) -> None:
        starter = self._load_starter()

        args = starter._parse_args([])

        self.assertEqual(args.model_name, starter.DEFAULT_MODEL_NAME)
        self.assertEqual(args.num_answers, 16)
        self.assertEqual(args.seed_mode, "shared")
        self.assertTrue(args.enable_distiller_intervention)
        self.assertEqual(args.distiller_sampler_backend, "post_filter_exact")

    def test_parallel_requests_default_to_shared_seed_for_flashinfer_sampler(self) -> None:
        starter = self._load_starter()
        args = starter._parse_args(["--prompt", "Explain tLLM.", "--num-answers", "16", "--seed", "123"])

        prompts, params, prompt_indices, sample_indices = starter._build_parallel_requests(args)

        self.assertEqual(prompts, ["Explain tLLM."] * 16)
        self.assertEqual(prompt_indices, [0] * 16)
        self.assertEqual(sample_indices, list(range(16)))
        self.assertEqual(len(params), 16)
        self.assertTrue(all(getattr(p, "n") == 1 for p in params))
        self.assertEqual([getattr(p, "seed") for p in params[:3]], [None, None, None])
        self.assertEqual(starter._llm_seed(args), 123)

    def test_parallel_requests_can_use_per_request_seed_mode(self) -> None:
        starter = self._load_starter()
        args = starter._parse_args(
            ["--prompt", "Explain tLLM.", "--num-answers", "16", "--seed", "123", "--seed-mode", "per-request"]
        )

        _prompts, params, _prompt_indices, _sample_indices = starter._build_parallel_requests(args)

        self.assertEqual([getattr(p, "seed") for p in params[:3]], [123, 124, 125])
        self.assertIsNone(starter._llm_seed(args))

    def test_starter_uses_runtime_make_llm_so_vllm_hooks_are_installed(self) -> None:
        starter = self._load_starter()
        tllm = importlib.import_module("tllm")
        tools = importlib.import_module("tllm.util.tools")

        self.assertIs(starter.make_llm, tllm.make_llm)
        self.assertIsNot(starter.make_llm, tools.make_plain_llm)
        self.assertFalse(hasattr(tools, "make_llm"))

    def test_configure_starter_runtime_uses_esamp_model_bank_and_distiller(self) -> None:
        starter = self._load_starter()
        args = starter._parse_args(["--num-answers", "16", "--distiller-beta", "0.25"])

        with mock.patch.object(starter.esamp_support, "configure_esamp_runtime") as configure:
            starter._configure_esamp(args)

        kwargs = configure.call_args.kwargs
        self.assertTrue(kwargs["enable_esamp_training"])
        self.assertTrue(kwargs["per_request_model_bank"])
        self.assertEqual(kwargs["model_bank_slots"], 16)
        self.assertTrue(kwargs["enable_distiller_intervention"])
        self.assertEqual(kwargs["distiller_beta"], 0.25)

    def test_starter_summary_includes_sampler_guidance_counters(self) -> None:
        starter = self._load_starter()
        stats = mock.Mock(loss_avg=1.25, loss_count=8)
        timing = mock.Mock(
            port_publish_hit_count=7,
            candidate_sample_count=6,
            candidate_token_count=123,
            candidate_max_count=19,
        )

        summary = starter._format_esamp_summary(
            stats=stats,
            timing=timing,
            answers=16,
            distiller_enabled=True,
            distiller_beta=0.9,
        )

        self.assertIn("distiller_candidate_samples=6", summary)
        self.assertIn("distiller_candidate_tokens=123", summary)
        self.assertIn("distiller_port_hits=7", summary)


if __name__ == "__main__":
    unittest.main()
