#!/usr/bin/env python3
"""Unit tests for the request metadata port."""

from __future__ import annotations

import unittest

from tllm.ports.request_meta import RequestMeta, RequestMetaLocator


class RequestMetaPortUnitTest(unittest.TestCase):
    def test_request_meta_is_read_only(self) -> None:
        self.assertTrue(RequestMeta.READABLE)
        self.assertFalse(RequestMeta.WRITABLE)
        self.assertEqual(RequestMeta.KIND.value, "request_meta")
        self.assertTrue(RequestMeta.BACKING_VLLM_STRUCT)

    def test_request_meta_read_uses_empty_locator(self) -> None:
        read_spec = RequestMeta.read()

        self.assertIsInstance(read_spec.locator, RequestMetaLocator)
        self.assertEqual(read_spec.kind, RequestMeta.KIND)


if __name__ == "__main__":
    unittest.main()
