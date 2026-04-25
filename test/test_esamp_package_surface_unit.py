#!/usr/bin/env python3
"""Unit tests for the target ESamp package surface."""

from __future__ import annotations

from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class ESampPackageSurfaceUnitTest(unittest.TestCase):
    def test_esamp_package_owns_only_consumer_side_files(self) -> None:
        root = REPO_ROOT / "tllm" / "consumers" / "esamp"

        self.assertTrue((root / "consumer.py").is_file())
        self.assertTrue((root / "engine.py").is_file())
        self.assertFalse((root / "runtime_adapter.py").exists())
        self.assertFalse((root / "runtime_support.py").exists())
        self.assertFalse((root / "workflow_support.py").exists())
        self.assertFalse((root / "engine").exists())

    def test_runtime_package_exposes_generic_residual_runtime_host(self) -> None:
        self.assertTrue((REPO_ROOT / "tllm" / "runtime" / "residual_runtime.py").is_file())


if __name__ == "__main__":
    unittest.main()
