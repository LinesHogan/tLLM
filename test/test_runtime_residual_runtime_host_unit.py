#!/usr/bin/env python3
"""Unit tests for the generic residual runtime host."""

from __future__ import annotations

import importlib
import sys
import unittest


class RuntimeResidualRuntimeHostUnitTest(unittest.TestCase):
    def _drop_modules(self, *prefixes: str) -> None:
        for name in list(sys.modules):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                sys.modules.pop(name, None)

    def test_residual_runtime_host_exposes_runtime_state_and_patch_entrypoints(self) -> None:
        self._drop_modules("tllm.runtime.residual_runtime")
        host = importlib.import_module("tllm.runtime.residual_runtime")

        self.assertTrue(hasattr(host, "RUNTIME"))
        self.assertTrue(callable(host.configure_runtime))
        self.assertTrue(callable(host.make_llm))
        self.assertTrue(callable(host.register_dispatch_consumer))
        self.assertTrue(callable(host.replace_dispatch_consumers))
        self.assertTrue(callable(host.clear_dispatch_consumers))


if __name__ == "__main__":
    unittest.main()
