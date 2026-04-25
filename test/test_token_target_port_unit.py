#!/usr/bin/env python3
"""Unit tests for the token target port."""

from __future__ import annotations

import unittest

from tllm.ports.token_target import TokenTarget, TokenTargetLocator


class TokenTargetPortUnitTest(unittest.TestCase):
    def test_token_target_is_read_only(self) -> None:
        self.assertTrue(TokenTarget.READABLE)
        self.assertFalse(TokenTarget.WRITABLE)
        self.assertEqual(TokenTarget.KIND.value, "token_target")
        self.assertTrue(TokenTarget.BACKING_VLLM_STRUCT)

    def test_token_target_read_builds_locator(self) -> None:
        read_spec = TokenTarget.read(step_scope="current")

        self.assertIsInstance(read_spec.locator, TokenTargetLocator)
        self.assertEqual(read_spec.locator.step_scope, "current")


if __name__ == "__main__":
    unittest.main()
