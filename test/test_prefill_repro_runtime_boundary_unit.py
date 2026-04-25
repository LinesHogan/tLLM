#!/usr/bin/env python3
"""Unit tests for removing prefill repro dependence on the old capture runtime."""

from __future__ import annotations

from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class PrefillReproRuntimeBoundaryUnitTest(unittest.TestCase):
    def test_prefill_repro_no_longer_imports_old_capture_runtime(self) -> None:
        text = (REPO_ROOT / "tllm" / "workflows" / "repro" / "repro_prefill_sampling_mse.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("from tllm.runtime import core", text)
        self.assertNotIn("import tllm.runtime.core", text)

    def test_prefill_repro_no_longer_calls_old_capture_runtime_entrypoints(self) -> None:
        text = (REPO_ROOT / "tllm" / "workflows" / "repro" / "repro_prefill_sampling_mse.py").read_text(
            encoding="utf-8"
        )
        for symbol in (
            "run_capture(",
            "reset_capture_runtime_state(",
            "attach_capture_consumer(",
        ):
            self.assertNotIn(symbol, text, msg=symbol)


if __name__ == "__main__":
    unittest.main()
