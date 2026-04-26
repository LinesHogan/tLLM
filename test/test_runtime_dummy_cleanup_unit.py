#!/usr/bin/env python3
"""Unit tests for cleanup of the dummy capture runtime surface."""

from __future__ import annotations

import importlib
from pathlib import Path
import unittest

from tllm.common import state as common_state

REPO_ROOT = Path(__file__).resolve().parents[1]


class RuntimeDummyCleanupUnitTest(unittest.TestCase):
    def test_runtime_state_uses_clear_dummy_runtime_names(self) -> None:
        state = common_state.RuntimeState()
        self.assertFalse(hasattr(state, "side_consumer"))
        self.assertTrue(hasattr(state, "decode_hidden_rows"))
        self.assertTrue(hasattr(state, "prefill_hidden_rows"))
        self.assertTrue(hasattr(state, "tap_consumer"))

    def test_common_state_no_longer_exposes_find_first_layer_helper(self) -> None:
        self.assertFalse(hasattr(common_state, "find_first_layer"))

    def test_old_capture_runtime_and_dummy_benchmarks_are_removed(self) -> None:
        for name in (
            "tllm.runtime.core",
            "tllm.workflows.benchmarks.benchmark",
            "tllm.workflows.benchmarks.throughput_triplet",
            "tllm.consumers.dummy_tap_mlp",
        ):
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module(name)

    def test_dummy_tap_consumer_module_is_removed(self) -> None:
        self.assertFalse((REPO_ROOT / "tllm" / "consumers" / "dummy_tap_mlp.py").exists())
        self.assertFalse((REPO_ROOT / "tllm" / "consumers" / "side_mlp.py").exists())


if __name__ == "__main__":
    unittest.main()
