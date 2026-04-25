#!/usr/bin/env python3
"""Unit tests for runtime common hook helpers."""

from __future__ import annotations

from enum import Enum
import unittest
from types import SimpleNamespace

from tllm.runtime.vllm_patch import common_hooks


class _GraphMode(Enum):
    NONE = 0
    PIECEWISE = 1


class RuntimeCommonHooksUnitTest(unittest.TestCase):
    def test_runner_uses_cudagraph_accepts_enum_mode_values(self) -> None:
        runner = SimpleNamespace(
            use_cuda_graph=False,
            compilation_config=SimpleNamespace(level=0, cudagraph_mode=_GraphMode.PIECEWISE),
        )

        self.assertTrue(common_hooks._runner_uses_cudagraph(runner))

    def test_runner_uses_cudagraph_handles_none_mode(self) -> None:
        runner = SimpleNamespace(
            use_cuda_graph=False,
            compilation_config=SimpleNamespace(level=0, cudagraph_mode=None),
        )

        self.assertFalse(common_hooks._runner_uses_cudagraph(runner))


if __name__ == "__main__":
    unittest.main()
