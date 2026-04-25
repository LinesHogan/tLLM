#!/usr/bin/env python3
"""Unit tests for lightweight runtime import boundaries."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]


class RuntimeImportBoundariesUnitTest(unittest.TestCase):
    def test_port_runtime_hooks_no_longer_imports_legacy_esamp_modules(self) -> None:
        text = (REPO_ROOT / "tllm/runtime/vllm_patch/port_runtime_hooks.py").read_text(encoding="utf-8")
        self.assertNotIn("legacy_esamp_bridge", text)
        self.assertNotIn("legacy_esamp_setup", text)

    def test_residual_capture_hooks_no_longer_imports_legacy_esamp_bridge(self) -> None:
        text = (REPO_ROOT / "tllm/runtime/ports/residual_capture_hooks.py").read_text(encoding="utf-8")
        self.assertNotIn("legacy_esamp_bridge", text)

    def _drop_modules(self, *prefixes: str) -> None:
        for name in list(sys.modules):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                sys.modules.pop(name, None)

    def test_importing_port_contracts_does_not_import_vllm(self) -> None:
        self._drop_modules("tllm.ports", "vllm")
        importlib.import_module("tllm.ports.base")
        self.assertNotIn("vllm", sys.modules)

    def test_runtime_package_no_longer_exposes_esamp_runtime_adapter_module(self) -> None:
        self._drop_modules("tllm.consumers.esamp.runtime_adapter")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("tllm.consumers.esamp.runtime_adapter")

    def test_importing_residual_runtime_does_not_eager_import_tool_helpers(self) -> None:
        code = (
            "import importlib, json, sys; "
            "importlib.import_module('tllm.runtime.residual_runtime'); "
            "print(json.dumps({'tools': 'tllm.util.tools' in sys.modules, 'vllm_worker': 'vllm.v1.worker.gpu_model_runner' in sys.modules}))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(completed.stdout.strip().splitlines()[-1], '{"tools": false, "vllm_worker": false}')

    def test_runtime_package_no_longer_exposes_esamp_runtime_module(self) -> None:
        self._drop_modules("tllm.runtime.esamp_runtime")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("tllm.runtime.esamp_runtime")

    def test_importing_port_runtime_hooks_uses_neutral_runtime_module(self) -> None:
        self._drop_modules("tllm.runtime.vllm_patch.port_runtime_hooks", "vllm")
        mod = importlib.import_module("tllm.runtime.vllm_patch.port_runtime_hooks")
        self.assertEqual(mod.__name__, "tllm.runtime.vllm_patch.port_runtime_hooks")
        self.assertNotIn("vllm", sys.modules)

    def test_importing_verification_harness_uses_verification_namespace(self) -> None:
        self._drop_modules("tllm.verification.automated_tests")
        mod = importlib.import_module("tllm.verification.automated_tests")
        self.assertEqual(mod.__name__, "tllm.verification.automated_tests")

    def test_workflows_package_no_longer_exposes_automation_test_harness(self) -> None:
        self._drop_modules("tllm.workflows.automation")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("tllm.workflows.automation.automated_tests")

    def test_workflows_package_no_longer_exposes_dummy_capture_benchmarks(self) -> None:
        for name in (
            "tllm.workflows.benchmarks.benchmark",
            "tllm.workflows.benchmarks.throughput_triplet",
            "tllm.consumers.dummy_tap_mlp",
        ):
            self._drop_modules(name)
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module(name)

    def test_verification_harness_no_longer_lists_triplet_throughput_scenarios(self) -> None:
        mod = importlib.import_module("tllm.verification.automated_tests")
        with mock.patch.object(sys, "argv", ["automated_tests"]):
            args = mod._parse_args()
        scenarios = mod._build_scenarios(args)
        ids = {s.scenario_id for s in scenarios}
        self.assertNotIn("throughput_triplet_qwen2p5_0p5b", ids)
        self.assertNotIn("throughput_triplet_qwen3_4b", ids)


if __name__ == "__main__":
    unittest.main()
