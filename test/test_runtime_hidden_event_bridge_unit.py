#!/usr/bin/env python3
"""Unit tests for the hidden event bridge extracted from runtime hooks."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.contracts.subscription import ConsumerSubscription
from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime import hidden_event_bridge


class RuntimeHiddenEventBridgeUnitTest(unittest.TestCase):
    def _core(self):
        runtime = type("Runtime", (), {})()
        runtime.tap_decode_hidden = {
            "layers.0": torch.tensor([[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]], dtype=torch.float32),
            "layers.1": torch.tensor([[5.0, 6.0], [7.0, 8.0], [88.0, 88.0]], dtype=torch.float32),
        }
        runtime.decode_row_idx = torch.tensor([0, 1, 2], dtype=torch.long)
        runtime.decode_valid_mask = torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float32)
        runtime.decode_count = 2
        runtime.decode_prompt_idxs = [10, 11]
        runtime.decode_sample_idxs = [0, 1]
        runtime.decode_prompt_idx_tensor = None
        runtime.decode_sample_idx_tensor = None
        runtime.decode_request_ids = ["reqA", "reqB"]
        runtime.residual_bindings = {
            "layers.0": ResidualPathBinding(
                locator=ResidualLocator(layer=0, site="block_output", phase="decode"),
                resolved_path="layers.0",
                include_request_meta=True,
            ),
            "layers.1": ResidualPathBinding(
                locator=ResidualLocator(layer=-1, site="block_output", phase="decode"),
                resolved_path="layers.1",
                include_request_meta=False,
            ),
        }
        runtime.source_resolved_path = "layers.0"
        runtime.target_resolved_path = "layers.1"
        runtime.event_step_id = 9
        return type("Core", (), {"RUNTIME": runtime})()

    class _LayerEventConsumer:
        consumer_id = "layer-event"

        def __init__(self) -> None:
            self.seen_events: list[str] = []

        def subscriptions(self):
            return [
                ConsumerSubscription(
                    consumer_id=self.consumer_id,
                    event_name="layer.post",
                    phase_filter="decode",
                    layer_filter="layers.0",
                    capture_policy="never",
                    dispatch_mode="inline",
                )
            ]

        def flows(self):
            return ()

        def consume(self, batch, ctx) -> None:
            _ = (batch, ctx)

        def on_tick(self, event_name, ctx) -> None:
            _ = ctx
            self.seen_events.append(str(event_name))

    def test_build_runtime_hidden_batch_uses_active_decode_rows_and_metadata(self) -> None:
        core = self._core()

        batch = hidden_event_bridge.build_runtime_hidden_batch(core=core, layer_path="layers.0")

        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertEqual(batch.step_id, 9)
        self.assertEqual(batch.phase, "decode")
        self.assertEqual(batch.layer_path, "layers.0")
        self.assertTrue(torch.equal(batch.rows_hidden, torch.tensor([[1.0, 2.0], [3.0, 4.0]])))
        self.assertEqual(batch.metadata["prompt_idxs"], [10, 11])
        self.assertEqual(batch.metadata["sample_idxs"], [0, 1])

    def test_build_runtime_hidden_batch_reuses_decode_metadata_tensors(self) -> None:
        core = self._core()
        core.RUNTIME.decode_prompt_idx_tensor = torch.tensor([10, 11], dtype=torch.long)
        core.RUNTIME.decode_sample_idx_tensor = torch.tensor([0, 1], dtype=torch.long)

        batch = hidden_event_bridge.build_runtime_hidden_batch(core=core, layer_path="layers.0")

        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertEqual(batch.prompt_idx.data_ptr(), core.RUNTIME.decode_prompt_idx_tensor.data_ptr())
        self.assertEqual(batch.sample_idx.data_ptr(), core.RUNTIME.decode_sample_idx_tensor.data_ptr())

    def test_build_runtime_hidden_batch_rejects_inconsistent_decode_metadata(self) -> None:
        core = self._core()
        core.RUNTIME.decode_request_ids = ["reqA"]

        with self.assertRaisesRegex(RuntimeError, "decode runtime metadata is inconsistent"):
            hidden_event_bridge.build_runtime_hidden_batch(core=core, layer_path="layers.0")

    def test_dispatch_deferred_layer_batches_dispatches_each_unique_layer_once(self) -> None:
        core = self._core()
        core.RUNTIME.target_resolved_path = "layers.0"
        core.RUNTIME.residual_bindings["layers.0"] = ResidualPathBinding(
            locator=ResidualLocator(layer=-1, site="block_output", phase="decode"),
            resolved_path="layers.0",
            include_request_meta=False,
        )
        core.RUNTIME.residual_bindings.pop("layers.1")
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        with mock.patch.object(hidden_event_bridge._common_hooks, "dispatch_runtime_event", side_effect=[1]) as p_dispatch:
            dispatched = hidden_event_bridge.dispatch_deferred_layer_batches(core=core, runner=runner)

        self.assertEqual(dispatched, 1)
        p_dispatch.assert_called_once()
        kwargs = p_dispatch.call_args.kwargs
        self.assertEqual(kwargs["event_name"], "layer.post")
        self.assertEqual(kwargs["phase"], "decode")
        self.assertEqual(kwargs["layer_path"], "layers.0")
        self.assertNotIn("batch", kwargs)
        self.assertIn("batch_factory", kwargs)
        batch = kwargs["batch_factory"]()
        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertEqual(batch.layer_path, "layers.0")

    def test_dispatch_deferred_layer_batches_defers_batch_materialization_to_subscriber_path(self) -> None:
        core = self._core()
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        with mock.patch.object(hidden_event_bridge, "build_runtime_hidden_batch") as p_build, mock.patch.object(
            hidden_event_bridge._common_hooks, "dispatch_runtime_event", return_value=0
        ):
            dispatched = hidden_event_bridge.dispatch_deferred_layer_batches(core=core, runner=runner)

        self.assertEqual(dispatched, 0)
        p_build.assert_not_called()

    def test_dispatch_deferred_layer_batches_skips_subscriber_when_hidden_batch_is_missing(self) -> None:
        core = self._core()
        core.RUNTIME.decode_count = 0
        consumer = self._LayerEventConsumer()
        core.RUNTIME.dispatch_plan = DispatchPlan.build([consumer])
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        dispatched = hidden_event_bridge.dispatch_deferred_layer_batches(core=core, runner=runner)

        self.assertEqual(dispatched, 0)
        self.assertEqual(consumer.seen_events, [])

    def test_dispatch_layer_lifecycle_events_emits_stack_end_only_for_target_layer(self) -> None:
        core = self._core()
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        with mock.patch.object(hidden_event_bridge._common_hooks, "dispatch_runtime_event") as p_dispatch:
            hidden_event_bridge.dispatch_layer_lifecycle_events(
                core=core,
                runner=runner,
                layer_path="layers.1",
                capture_enabled=True,
            )

        event_names = [call.kwargs["event_name"] for call in p_dispatch.call_args_list]
        self.assertEqual(event_names, ["layer.pre", "layer.post", "block.end", "stack.end"])

    def test_hidden_event_bridge_can_resolve_default_paths_from_residual_bindings(self) -> None:
        core = self._core()
        core.RUNTIME.source_resolved_path = ""
        core.RUNTIME.target_resolved_path = ""
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        with mock.patch.object(hidden_event_bridge._common_hooks, "dispatch_runtime_event", side_effect=[1, 1]) as p_dispatch:
            dispatched = hidden_event_bridge.dispatch_deferred_layer_batches(core=core, runner=runner)

        self.assertEqual(dispatched, 2)
        layer_paths = [call.kwargs["layer_path"] for call in p_dispatch.call_args_list]
        self.assertEqual(layer_paths, ["layers.0", "layers.1"])


if __name__ == "__main__":
    unittest.main()
