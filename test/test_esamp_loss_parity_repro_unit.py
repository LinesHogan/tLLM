#!/usr/bin/env python3
"""Unit tests for the ESamp loss parity repro entrypoint."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest
from unittest import mock


class ESampLossParityReproUnitTest(unittest.TestCase):
    def _drop_modules(self, *prefixes: str) -> None:
        for name in list(sys.modules):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                sys.modules.pop(name, None)

    def test_loss_parity_repro_emits_machine_readable_summary(self) -> None:
        self._drop_modules("tllm.workflows.repro.repro_esamp_loss_parity")
        repro = importlib.import_module("tllm.workflows.repro.repro_esamp_loss_parity")

        summary = repro._build_result_summary(
            payload={
                "cases": {
                    "single_on": {"loss_avg": 1.0, "loss_count": 10},
                    "per_request_on": {"loss_avg": 1.1, "loss_count": 20},
                    "model_bank_on": {"loss_avg": 1.2, "loss_count": 30},
                }
            },
            guardrail_doc="doc/guides/validation.md",
        )

        self.assertIn("parity_guardrail_doc", summary)
        self.assertIn("parity_guardrail_doc_available", summary)
        self.assertIn("training_active_passed", summary)

    def test_automated_tests_lists_esamp_loss_parity_scenario(self) -> None:
        automated_tests = importlib.import_module("tllm.verification.automated_tests")
        with mock.patch.object(sys, "argv", ["automated_tests"]):
            args = automated_tests._parse_args()
        scenarios = automated_tests._build_scenarios(args)
        ids = {scenario.scenario_id for scenario in scenarios}
        self.assertIn("esamp_loss_parity_qwen2p5_0p5b", ids)

    def test_loss_parity_repro_no_longer_depends_on_esamp_runtime_adapter(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        text = (repo_root / "tllm" / "workflows" / "repro" / "repro_esamp_loss_parity.py").read_text(encoding="utf-8")
        self.assertNotIn("tllm.consumers.esamp.runtime_adapter", text)
        self.assertIn("residual_runtime", text)


if __name__ == "__main__":
    unittest.main()
