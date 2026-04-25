#!/usr/bin/env python3
"""Unit tests for residual capture forward-hook installation."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.runtime.ports import residual_capture_hooks


class RuntimeResidualCaptureHooksUnitTest(unittest.TestCase):
    def test_install_layer_forward_taps_captures_decode_rows_and_calls_runtime_helpers(self) -> None:
        class _Layer(torch.nn.Module):
            def forward(self, x):
                return x + 1

        layer = _Layer()
        runtime = type("Runtime", (), {})()
        runtime.decode_row_idx = torch.tensor([2, 0], dtype=torch.long)
        runtime.decode_valid_mask = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        runtime.tap_decode_hidden = {"layers.0": torch.empty((2, 4), dtype=torch.float32)}
        runtime.launch_consumer_from_hooks = True
        runtime.dispatch_plan = type(
            "Plan",
            (),
            {"has_active_targets": staticmethod(lambda: True)},
        )()
        runtime.consumer = None
        runtime.source_resolved_path = "layers.0"
        runtime.target_resolved_path = "layers.1"
        runtime.distiller_port_capture_step_id = -1
        core = type("Core", (), {"RUNTIME": runtime})()
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        with mock.patch.object(
            residual_capture_hooks._sampler_patch,
            "maybe_capture_source_precompute",
        ) as p_precompute, mock.patch.object(
            residual_capture_hooks._hidden_bridge, "dispatch_layer_lifecycle_events"
        ) as p_events:
            residual_capture_hooks.install_layer_forward_taps(
                core=core,
                runner=runner,
                resolved_layers={"layers.0": layer},
            )
            x = torch.tensor(
                [
                    [10.0, 11.0, 12.0, 13.0],
                    [20.0, 21.0, 22.0, 23.0],
                    [30.0, 31.0, 32.0, 33.0],
                ],
                dtype=torch.float32,
            )
            out = layer(x)

        self.assertTrue(torch.equal(out, x + 1))
        self.assertTrue(
            torch.equal(
                runtime.tap_decode_hidden["layers.0"],
                torch.tensor(
                    [
                        [31.0, 32.0, 33.0, 34.0],
                        [11.0, 12.0, 13.0, 14.0],
                    ],
                    dtype=torch.float32,
                ),
            )
        )
        p_precompute.assert_called_once_with(
            runtime=runtime,
            runner=runner,
            layer_path="layers.0",
        )
        self.assertEqual(runtime.distiller_port_capture_step_id, -1)
        p_events.assert_called_once()

    def test_install_layer_forward_taps_noops_when_runtime_is_inactive(self) -> None:
        class _Layer(torch.nn.Module):
            def forward(self, x):
                return x + 1

        layer = _Layer()
        runtime = type("Runtime", (), {})()
        runtime.decode_row_idx = torch.tensor([1, 0], dtype=torch.long)
        runtime.decode_valid_mask = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        runtime.tap_decode_hidden = {"layers.0": torch.full((2, 4), -1.0, dtype=torch.float32)}
        runtime.launch_consumer_from_hooks = True
        runtime.dispatch_plan = None
        runtime.consumer = None
        runtime.source_resolved_path = "layers.0"
        runtime.target_resolved_path = "layers.1"
        runtime.distiller_port_capture_step_id = -1
        core = type("Core", (), {"RUNTIME": runtime})()
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        with mock.patch.object(residual_capture_hooks._hidden_bridge, "dispatch_layer_lifecycle_events") as p_events:
            residual_capture_hooks.install_layer_forward_taps(
                core=core,
                runner=runner,
                resolved_layers={"layers.0": layer},
            )
            x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            out = layer(x)

        self.assertTrue(torch.equal(out, x + 1))
        self.assertTrue(torch.equal(runtime.tap_decode_hidden["layers.0"], torch.full((2, 4), -1.0, dtype=torch.float32)))
        p_events.assert_not_called()


if __name__ == "__main__":
    unittest.main()
