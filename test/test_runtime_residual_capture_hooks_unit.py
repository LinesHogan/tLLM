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
        runtime.decode_count = 2
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
        p_events.assert_called_once()

    def test_install_layer_forward_taps_noops_when_runtime_is_inactive(self) -> None:
        class _Layer(torch.nn.Module):
            def forward(self, x):
                return x + 1

        layer = _Layer()
        runtime = type("Runtime", (), {})()
        runtime.decode_row_idx = torch.tensor([1, 0], dtype=torch.long)
        runtime.decode_valid_mask = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        runtime.decode_count = 2
        runtime.tap_decode_hidden = {"layers.0": torch.full((2, 4), -1.0, dtype=torch.float32)}
        runtime.launch_consumer_from_hooks = True
        runtime.dispatch_plan = None
        runtime.consumer = None
        runtime.source_resolved_path = "layers.0"
        runtime.target_resolved_path = "layers.1"
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

    def test_install_layer_forward_taps_does_not_mask_inactive_tail(self) -> None:
        class _Layer(torch.nn.Module):
            def forward(self, x):
                return x + 1

        layer = _Layer()
        runtime = type("Runtime", (), {})()
        runtime.decode_row_idx = torch.tensor([2, 0, 1], dtype=torch.long)
        runtime.decode_valid_mask = torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float32)
        runtime.decode_count = 2
        runtime.tap_decode_hidden = {"layers.0": torch.empty((3, 2), dtype=torch.float32)}
        runtime.launch_consumer_from_hooks = True
        runtime.dispatch_plan = type(
            "Plan",
            (),
            {"has_active_targets": staticmethod(lambda: True)},
        )()
        runtime.consumer = None
        runtime.source_resolved_path = "layers.0"
        runtime.target_resolved_path = "layers.1"
        core = type("Core", (), {"RUNTIME": runtime})()
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        with mock.patch.object(residual_capture_hooks._hidden_bridge, "dispatch_layer_lifecycle_events"):
            residual_capture_hooks.install_layer_forward_taps(
                core=core,
                runner=runner,
                resolved_layers={"layers.0": layer},
            )
            x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
            layer(x)

        self.assertTrue(torch.equal(runtime.tap_decode_hidden["layers.0"][2], torch.tensor([4.0, 5.0])))

    def test_install_layer_forward_taps_populates_compact_lane_without_mutating_full_lane(self) -> None:
        class _Layer(torch.nn.Module):
            def forward(self, x):
                return x + 1

        layer = _Layer()
        runtime = type("Runtime", (), {})()
        runtime.decode_row_idx = torch.tensor([2, 0, 1], dtype=torch.long)
        runtime.decode_valid_mask = torch.ones((3, 1), dtype=torch.float32)
        runtime.decode_compact_row_idx = torch.tensor([2, 1], dtype=torch.long)
        runtime.decode_compact_count = 2
        runtime.tap_decode_hidden = {"layers.0": torch.empty((3, 2), dtype=torch.float32)}
        runtime.tap_decode_hidden_compact = {"layers.0": torch.empty((2, 2), dtype=torch.float32)}
        runtime.launch_consumer_from_hooks = True
        runtime.dispatch_plan = type(
            "Plan",
            (),
            {"has_active_targets": staticmethod(lambda: True)},
        )()
        runtime.consumer = None
        runtime.source_resolved_path = "layers.0"
        runtime.target_resolved_path = "layers.1"
        core = type("Core", (), {"RUNTIME": runtime})()
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        with mock.patch.object(residual_capture_hooks._hidden_bridge, "dispatch_layer_lifecycle_events"):
            residual_capture_hooks.install_layer_forward_taps(
                core=core,
                runner=runner,
                resolved_layers={"layers.0": layer},
            )
            x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
            layer(x)

        self.assertTrue(
            torch.equal(
                runtime.tap_decode_hidden["layers.0"],
                torch.tensor([[6.0, 7.0], [2.0, 3.0], [4.0, 5.0]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                runtime.tap_decode_hidden_compact["layers.0"],
                torch.tensor([[6.0, 7.0], [4.0, 5.0]], dtype=torch.float32),
            )
        )

    def test_install_layer_forward_taps_can_skip_full_lane_for_compact_only_runtime(self) -> None:
        class _Layer(torch.nn.Module):
            def forward(self, x):
                return x + 1

        layer = _Layer()
        runtime = type("Runtime", (), {})()
        runtime.decode_row_idx = torch.tensor([2, 0, 1], dtype=torch.long)
        runtime.decode_valid_mask = torch.ones((3, 1), dtype=torch.float32)
        runtime.decode_compact_row_idx = torch.tensor([2, 1], dtype=torch.long)
        runtime.decode_compact_count = 2
        runtime.capture_full_residual_rows = False
        runtime.tap_decode_hidden = {"layers.0": torch.full((3, 2), -1.0, dtype=torch.float32)}
        runtime.tap_decode_hidden_compact = {"layers.0": torch.empty((2, 2), dtype=torch.float32)}
        runtime.launch_consumer_from_hooks = True
        runtime.dispatch_plan = type(
            "Plan",
            (),
            {"has_active_targets": staticmethod(lambda: True)},
        )()
        runtime.consumer = None
        runtime.source_resolved_path = "layers.0"
        runtime.target_resolved_path = "layers.1"
        core = type("Core", (), {"RUNTIME": runtime})()
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        with mock.patch.object(residual_capture_hooks._hidden_bridge, "dispatch_layer_lifecycle_events"):
            residual_capture_hooks.install_layer_forward_taps(
                core=core,
                runner=runner,
                resolved_layers={"layers.0": layer},
            )
            x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
            layer(x)

        self.assertTrue(torch.equal(runtime.tap_decode_hidden["layers.0"], torch.full((3, 2), -1.0, dtype=torch.float32)))
        self.assertTrue(
            torch.equal(
                runtime.tap_decode_hidden_compact["layers.0"],
                torch.tensor([[6.0, 7.0], [4.0, 5.0]], dtype=torch.float32),
            )
        )

    def test_install_layer_forward_taps_only_updates_active_compact_rows(self) -> None:
        class _Layer(torch.nn.Module):
            def forward(self, x):
                return x + 1

        layer = _Layer()
        runtime = type("Runtime", (), {})()
        runtime.decode_row_idx = torch.tensor([0], dtype=torch.long)
        runtime.decode_valid_mask = torch.ones((1, 1), dtype=torch.float32)
        runtime.decode_compact_row_idx = torch.tensor([0, 99, 99], dtype=torch.long)
        runtime.decode_compact_count = 1
        runtime.capture_full_residual_rows = False
        runtime.tap_decode_hidden = {"layers.0": torch.full((3, 2), -1.0, dtype=torch.float32)}
        runtime.tap_decode_hidden_compact = {
            "layers.0": torch.full((3, 2), -7.0, dtype=torch.float32)
        }
        runtime.launch_consumer_from_hooks = True
        runtime.dispatch_plan = type(
            "Plan",
            (),
            {"has_active_targets": staticmethod(lambda: True)},
        )()
        runtime.consumer = None
        runtime.source_resolved_path = "layers.0"
        runtime.target_resolved_path = "layers.1"
        core = type("Core", (), {"RUNTIME": runtime})()
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        with mock.patch.object(residual_capture_hooks._hidden_bridge, "dispatch_layer_lifecycle_events"):
            residual_capture_hooks.install_layer_forward_taps(
                core=core,
                runner=runner,
                resolved_layers={"layers.0": layer},
            )
            x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
            layer(x)

        self.assertTrue(
            torch.equal(
                runtime.tap_decode_hidden_compact["layers.0"],
                torch.tensor([[2.0, 3.0], [-7.0, -7.0], [-7.0, -7.0]], dtype=torch.float32),
            )
        )


if __name__ == "__main__":
    unittest.main()
