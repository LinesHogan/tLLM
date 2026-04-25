#!/usr/bin/env python3
"""Unit tests for the KV cache port."""

from __future__ import annotations

import unittest

from tllm.ports.kv_cache import KVCache, KVLocator


class KVCachePortUnitTest(unittest.TestCase):
    def test_kv_cache_port_is_read_write(self) -> None:
        self.assertTrue(KVCache.READABLE)
        self.assertTrue(KVCache.WRITABLE)
        self.assertEqual(KVCache.KIND.value, "kv_cache")
        self.assertTrue(KVCache.BACKING_VLLM_STRUCT)

    def test_kv_cache_locator_tracks_layer_phase_and_step_scope(self) -> None:
        locator = KVLocator(layer=20, phase="decode", step_scope="next")

        self.assertEqual(locator.layer, 20)
        self.assertEqual(locator.phase, "decode")
        self.assertEqual(locator.step_scope, "next")

    def test_kv_cache_builders_attach_typed_locator(self) -> None:
        read_spec = KVCache.read(layer=3, phase="decode", step_scope="current")
        write_spec = KVCache.write(layer=3, phase="decode", step_scope="next")

        self.assertIsInstance(read_spec.locator, KVLocator)
        self.assertIsInstance(write_spec.locator, KVLocator)


if __name__ == "__main__":
    unittest.main()
