#!/usr/bin/env python3
"""Unit tests for residual tap buffer setup helpers."""

from __future__ import annotations

import unittest

import torch

from tllm.runtime.ports import residual_capture_buffers


class RuntimeResidualTapBuffersUnitTest(unittest.TestCase):
    def test_initialize_runtime_tap_buffers_registers_buffers_and_maps(self) -> None:
        runtime = type("Runtime", (), {})()
        runtime.tap_layers = {}
        runtime.tap_scratch = {}
        runtime.tap_decode_hidden = {}

        layer = torch.nn.Linear(4, 4)

        residual_capture_buffers.initialize_runtime_tap_buffers(
            runtime=runtime,
            resolved_layers={"layers.0": layer},
            device=torch.device("cpu"),
            rows=3,
            hidden_size=4,
            hidden_dtype=torch.float32,
        )

        self.assertIs(runtime.tap_layers["layers.0"], layer)
        self.assertEqual(tuple(runtime.tap_scratch["layers.0"].shape), (3, 4))
        self.assertEqual(tuple(runtime.tap_decode_hidden["layers.0"].shape), (3, 4))
        self.assertEqual(runtime.tap_scratch["layers.0"].dtype, torch.float32)
        self.assertEqual(runtime.tap_decode_hidden["layers.0"].dtype, torch.float32)
        self.assertIn("tllm_residual_capture_scratch_0", layer._buffers)
        self.assertIn("tllm_residual_capture_rows_0", layer._buffers)
        self.assertNotIn("esamp_scratch_0", layer._buffers)
        self.assertNotIn("esamp_decode_0", layer._buffers)


if __name__ == "__main__":
    unittest.main()
