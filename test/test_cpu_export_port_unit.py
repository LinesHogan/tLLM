#!/usr/bin/env python3
"""Unit tests for the CPU export port."""

from __future__ import annotations

import unittest

from tllm.ports.cpu_export import CpuExport, CpuExportLocator


class CpuExportPortUnitTest(unittest.TestCase):
    def test_cpu_export_is_write_only(self) -> None:
        self.assertFalse(CpuExport.READABLE)
        self.assertTrue(CpuExport.WRITABLE)
        self.assertEqual(CpuExport.KIND.value, "cpu_export")
        self.assertTrue(CpuExport.BACKING_VLLM_STRUCT)

    def test_cpu_export_locator_carries_channel_format_and_optional_schema(self) -> None:
        locator = CpuExportLocator(channel="hidden_db", format="row_batch", schema="v1")
        self.assertEqual(locator.channel, "hidden_db")
        self.assertEqual(locator.format, "row_batch")
        self.assertEqual(locator.schema, "v1")

    def test_cpu_export_write_builds_typed_port_write(self) -> None:
        write_spec = CpuExport.write(channel="sae_activation", format="tensor_dict", schema=None)
        self.assertIsInstance(write_spec.locator, CpuExportLocator)
        self.assertEqual(write_spec.kind, CpuExport.KIND)


if __name__ == "__main__":
    unittest.main()
