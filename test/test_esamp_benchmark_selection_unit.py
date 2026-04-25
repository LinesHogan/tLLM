#!/usr/bin/env python3
"""Unit tests for ESamp benchmark implementation selection helpers."""

from __future__ import annotations

import json
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from tllm.workflows.benchmarks import side_train_benchmark as bench


class ESampBenchmarkSelectionUnitTest(unittest.TestCase):
    def test_cli_no_longer_exposes_consumer_implementation_selector(self) -> None:
        with mock.patch.object(sys, "argv", ["side_train_benchmark"]):
            args = bench._parse_args()
        self.assertFalse(hasattr(args, "consumer_implementation"))
        self.assertFalse(hasattr(args, "compare_consumer_implementations"))

    def test_compute_esamp_ratio_uses_out_tok_per_s(self) -> None:
        legacy = {"out_tok_per_s": 100.0}
        base_consumer = {"out_tok_per_s": 97.0}
        self.assertAlmostEqual(bench._compute_esamp_ratio(legacy, base_consumer), 0.97)

    def test_esamp_ratio_handles_zero_legacy(self) -> None:
        self.assertEqual(bench._compute_esamp_ratio({"out_tok_per_s": 0.0}, {"out_tok_per_s": 1.0}), 0.0)

    def test_build_comparison_summary_applies_95_percent_gate(self) -> None:
        summary = bench._build_comparison_summary(
            legacy_summary={"with_train": {"out_tok_per_s": 100.0}},
            base_consumer_summary={"with_train": {"out_tok_per_s": 95.0}},
        )
        self.assertEqual(summary["min_ratio"], bench.ESAMP_MIN_OUT_TOK_RATIO)
        self.assertAlmostEqual(summary["ratio"], 0.95)
        self.assertTrue(summary["passed"])

    def test_compare_mode_runs_each_implementation_in_subprocess(self) -> None:
        args = SimpleNamespace(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            prompt=["hello"],
            prompt_file="",
            dtype="bfloat16",
            gpu_memory_utilization=0.4,
            max_model_len=128,
            enforce_eager=True,
            graph_scratch_rows=0,
            tap_layer_path=[],
            source_layer_path="model.model.layers[0].input_layernorm",
            target_layer_path="model.model.layers[-1].input_layernorm",
            side_hidden_dim=64,
            side_lr=1e-3,
            consumer_implementation="legacy",
            compare_consumer_implementations=True,
            benchmark_batch_size=1,
            benchmark_max_new_tokens=4,
            benchmark_warmup_rounds=0,
            benchmark_rounds=1,
            benchmark_bidirectional=False,
            benchmark_ignore_eos=True,
            benchmark_disable_prefix_caching=True,
            benchmark_case_cooldown_s=0.5,
            benchmark_log_memory=False,
            emit_json_summary=False,
        )
        payloads = [
            {
                "implementation": "legacy",
                "summary": {
                    "tap_only": {"out_tok_per_s": 100.0},
                    "with_train": {"out_tok_per_s": 95.0},
                },
            },
            {
                "implementation": "base_consumer",
                "summary": {
                    "tap_only": {"out_tok_per_s": 100.0},
                    "with_train": {"out_tok_per_s": 96.0},
                },
            },
        ]
        completed = [
            mock.Mock(returncode=0, stdout=f"logs\n{bench.JSON_SUMMARY_PREFIX}{json.dumps(payload)}\n", stderr="")
            for payload in payloads
        ]

        with mock.patch.object(bench.subprocess, "run", side_effect=completed) as run_mock:
            results = bench._run_compare_consumer_implementations(args)

        self.assertEqual(run_mock.call_count, 2)
        first_cmd = run_mock.call_args_list[0].args[0]
        second_cmd = run_mock.call_args_list[1].args[0]
        self.assertIn("--emit-json-summary", first_cmd)
        self.assertIn("--emit-json-summary", second_cmd)
        self.assertEqual(first_cmd[first_cmd.index("--consumer-implementation") + 1], "legacy")
        self.assertEqual(second_cmd[second_cmd.index("--consumer-implementation") + 1], "base_consumer")
        self.assertAlmostEqual(results["ratio"], 96.0 / 95.0)
        self.assertEqual(results["min_ratio"], bench.ESAMP_MIN_OUT_TOK_RATIO)
        self.assertTrue(results["passed"])

    def test_extract_json_summary_requires_marker(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON summary marker"):
            bench._extract_json_summary("no summary here")


if __name__ == "__main__":
    unittest.main()
