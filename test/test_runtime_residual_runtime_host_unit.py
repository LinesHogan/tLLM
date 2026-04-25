#!/usr/bin/env python3
"""Unit tests for the generic residual runtime host."""

from __future__ import annotations

import importlib
import unittest


class RuntimeResidualRuntimeHostUnitTest(unittest.TestCase):
    def test_residual_runtime_host_exposes_runtime_state_and_patch_entrypoints(self) -> None:
        host = importlib.import_module("tllm.runtime.residual_runtime")

        self.assertTrue(hasattr(host, "RUNTIME"))
        self.assertTrue(callable(host.configure_runtime))
        self.assertTrue(callable(host.make_llm))
        self.assertTrue(callable(host.register_dispatch_consumer))
        self.assertTrue(callable(host.replace_dispatch_consumers))
        self.assertTrue(callable(host.clear_dispatch_consumers))

    def test_public_tllm_make_llm_is_runtime_hook_installing_entrypoint(self) -> None:
        tllm = importlib.import_module("tllm")
        host = importlib.import_module("tllm.runtime.residual_runtime")

        self.assertIs(tllm.make_llm, host.make_llm)

    def test_util_tools_exposes_only_explicit_plain_llm_constructor(self) -> None:
        tools = importlib.import_module("tllm.util.tools")

        self.assertTrue(callable(tools.make_plain_llm))
        self.assertFalse(hasattr(tools, "make_llm"))


if __name__ == "__main__":
    unittest.main()
