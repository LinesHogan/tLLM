#!/usr/bin/env python3
"""Unit tests for the residual stream port."""

from __future__ import annotations

import unittest

from tllm.ports.base import PortKind
from tllm.ports.residual_stream import ResidualLocator, ResidualStream


class ResidualStreamPortUnitTest(unittest.TestCase):
    def test_residual_stream_exposes_approved_site_vocabulary(self) -> None:
        self.assertEqual(
            ResidualStream.SUPPORTED_SITES,
            (
                "block_input",
                "attn_input",
                "attn_output",
                "mlp_input",
                "block_output",
            ),
        )
        self.assertEqual(ResidualStream.KIND, PortKind.RESIDUAL_STREAM)
        self.assertTrue(ResidualStream.BACKING_VLLM_STRUCT)

    def test_residual_locator_validates_layer_site_and_phase(self) -> None:
        locator = ResidualLocator(layer=-1, site="block_output", phase="decode")
        self.assertEqual(locator.layer, -1)
        self.assertEqual(locator.site, "block_output")
        self.assertEqual(locator.phase, "decode")

        with self.assertRaises(ValueError):
            ResidualLocator(layer=0, site="not_a_site", phase="decode")
        with self.assertRaises(ValueError):
            ResidualLocator(layer=0, site="block_output", phase="invalid")

    def test_residual_stream_builders_attach_typed_locator(self) -> None:
        read_spec = ResidualStream.read(layer=0, site="block_output", phase="decode", role="source")
        write_spec = ResidualStream.write(layer=1, site="attn_output", phase="decode")

        self.assertIsInstance(read_spec.locator, ResidualLocator)
        self.assertEqual(read_spec.role, "source")
        self.assertIsInstance(write_spec.locator, ResidualLocator)
        self.assertEqual(write_spec.kind, PortKind.RESIDUAL_STREAM)


if __name__ == "__main__":
    unittest.main()
