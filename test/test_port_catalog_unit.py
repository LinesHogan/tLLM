#!/usr/bin/env python3
"""Unit tests for the bounded public port catalog."""

from __future__ import annotations

import unittest

from tllm.ports.base import PortKind
from tllm.ports.catalog import PUBLIC_PORT_KINDS


class PortCatalogUnitTest(unittest.TestCase):
    def test_port_kind_is_closed_and_matches_public_catalog(self) -> None:
        expected = [
            "residual_stream",
            "kv_cache",
            "logits",
            "sampler",
            "token_target",
            "request_meta",
            "cpu_export",
        ]

        self.assertEqual([kind.value for kind in PortKind], expected)
        self.assertEqual([kind.value for kind in PUBLIC_PORT_KINDS], expected)
        self.assertEqual(set(PUBLIC_PORT_KINDS), set(PortKind))


if __name__ == "__main__":
    unittest.main()
