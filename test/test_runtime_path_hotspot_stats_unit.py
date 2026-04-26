#!/usr/bin/env python3
"""Unit tests for lightweight runtime path-hotspot accounting."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from tllm.runtime import residual_runtime


class RuntimePathHotspotStatsUnitTest(unittest.TestCase):
    def tearDown(self) -> None:
        residual_runtime.RUNTIME.path_hotspot_enabled = False
        residual_runtime.RUNTIME.path_hotspot_cpu_ms = {}
        residual_runtime.RUNTIME.path_hotspot_counts = {}

    def test_path_hotspot_stats_are_disabled_by_default(self) -> None:
        residual_runtime.RUNTIME.path_hotspot_enabled = False

        residual_runtime.record_path_hotspot_cpu("execute_model_tail", 1.25)
        stats = residual_runtime.read_and_reset_path_hotspot_stats(sync=False)

        self.assertEqual(stats.cpu_ms_total, {})
        self.assertEqual(stats.counts, {})

    def test_path_hotspot_stats_accumulate_and_reset_when_enabled(self) -> None:
        residual_runtime.RUNTIME.path_hotspot_enabled = True

        residual_runtime.record_path_hotspot_cpu("execute_model_tail", 1.25)
        residual_runtime.record_path_hotspot_cpu("execute_model_tail", 0.75)
        residual_runtime.record_path_hotspot_cpu("prepare_decode_localization", 0.5)

        stats = residual_runtime.read_and_reset_path_hotspot_stats(sync=False)

        self.assertEqual(stats.cpu_ms_total["execute_model_tail"], 2.0)
        self.assertEqual(stats.cpu_ms_total["prepare_decode_localization"], 0.5)
        self.assertEqual(stats.counts["execute_model_tail"], 2)
        self.assertEqual(stats.counts["prepare_decode_localization"], 1)
        self.assertEqual(residual_runtime.RUNTIME.path_hotspot_cpu_ms, {})
        self.assertEqual(residual_runtime.RUNTIME.path_hotspot_counts, {})

    def test_configure_runtime_reads_path_hotspot_env_flag(self) -> None:
        with mock.patch.dict(os.environ, {"TLLM_TRACE_PATH_HOTSPOTS": "1"}):
            residual_runtime.configure_runtime(
                graph_scratch_rows=4,
                tap_layer_paths=[],
                source_layer_path="source",
                target_layer_path="target",
                enable_esamp_training=False,
                distiller_hidden_dim=2,
                distiller_lr=1e-3,
            )

        self.assertTrue(residual_runtime.RUNTIME.path_hotspot_enabled)


if __name__ == "__main__":
    unittest.main()
