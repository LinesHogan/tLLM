#!/usr/bin/env python3
"""Unit tests for the logits port."""

from __future__ import annotations

import unittest

from tllm.ports.logits import Logits, LogitsLocator


class LogitsPortUnitTest(unittest.TestCase):
    def test_logits_port_is_read_only_for_now(self) -> None:
        self.assertTrue(Logits.READABLE)
        self.assertFalse(Logits.WRITABLE)
        self.assertEqual(Logits.KIND.value, "logits")
        self.assertTrue(Logits.BACKING_VLLM_STRUCT)

    def test_logits_read_builds_typed_locator(self) -> None:
        read_spec = Logits.read(step_scope="current")

        self.assertIsInstance(read_spec.locator, LogitsLocator)
        self.assertEqual(read_spec.locator.step_scope, "current")


if __name__ == "__main__":
    unittest.main()
