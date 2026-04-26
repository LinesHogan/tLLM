#!/usr/bin/env python3
"""Unit tests for ESamp flow-based public contracts."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig
from tllm.contracts.gpu_stage import DeviceTensorLease
from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.request_meta_view import RowBatchMeta
from tllm.contracts.runtime_context import RuntimeContext
from tllm.ports.base import PortKind
from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.vllm_patch import port_runtime_hooks


class ESampFlowContractsUnitTest(unittest.TestCase):
    def test_flows_cover_source_target_and_request_metadata(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(), engine=mock.Mock())
        flows = consumer.flows()

        self.assertEqual(len(flows), 1)
        self.assertFalse(hasattr(consumer, "consume"))
        self.assertFalse(hasattr(consumer, "on_tick"))
        flow = flows[0]
        self.assertEqual(flow.window, "out_of_band")
        self.assertEqual(
            [read.kind for read in flow.reads],
            [PortKind.RESIDUAL_STREAM, PortKind.RESIDUAL_STREAM, PortKind.REQUEST_META],
        )
        self.assertEqual([read.role for read in flow.reads[:2]], ["source", "target"])
        self.assertEqual(flow.bundle_key, ("engine_step_id", "phase"))
        self.assertEqual(flow.delivery, "device_lease")
        self.assertEqual(flow.ownership, "runtime_lease")
        self.assertEqual(flow.row_compaction, "none")

    def test_model_bank_flow_requests_first_row_per_prompt_delivery(self) -> None:
        consumer = ESampConsumer(
            ESampConsumerConfig(per_request_models=True, per_request_model_bank=True),
            engine=mock.Mock(),
        )

        flow = consumer.flows()[0]

        self.assertEqual(flow.delivery, "device_lease")
        self.assertEqual(flow.row_compaction, "first_per_prompt")

    def test_consume_bundle_routes_aggregated_source_target_rows_into_engine(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=4), engine=engine)
        ctx = RuntimeContext(
            runner=SimpleNamespace(max_num_reqs=8),
            model=None,
            device=torch.device("cpu"),
            main_stream=None,
            is_compiling=False,
            uses_cudagraph=False,
            event_name="flow:decode",
        )
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="decode", request_id="reqA", sample_idx=0),
            entries={
                "source": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
                "target": torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
                "request_meta": RowBatchMeta(
                    request_ids=("reqA", "reqB"),
                    prompt_idxs=(7, 9),
                    sample_idxs=(0, 1),
                    phase="decode",
                    engine_step_id=1,
                ),
            },
        )

        with mock.patch.object(consumer, "_ensure_runtime_resources", return_value=None):
            consumer.consume_bundle(bundle, ctx)

        engine.launch_step.assert_called_once()
        self.assertTrue(torch.equal(engine.launch_step.call_args.args[0], bundle.entries["source"]))
        self.assertTrue(torch.equal(engine.launch_step.call_args.args[1], bundle.entries["target"]))
        engine.launch_forward.assert_not_called()
        engine.launch_target.assert_not_called()
        engine.launch_delayed_backward.assert_not_called()
        consumer.on_step_end(ctx)
        engine.launch_delayed_backward.assert_called_once_with(2, prompt_idxs=(7, 9))

    def test_consume_bundle_passes_device_lease_to_engine_owned_step_stage(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=4), engine=engine)
        ctx = RuntimeContext(
            runner=SimpleNamespace(max_num_reqs=8),
            model=None,
            device=torch.device("cpu"),
            main_stream=None,
            is_compiling=False,
            uses_cudagraph=False,
            event_name="flow:decode",
        )
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        target = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="decode", request_id="reqA", sample_idx=0),
            entries={
                "device_lease": DeviceTensorLease(entries={"source": source, "target": target}, active_rows=2),
                "request_meta": RowBatchMeta(
                    request_ids=("reqA", "reqB"),
                    prompt_idxs=(7, 9),
                    sample_idxs=(0, 1),
                    phase="decode",
                    engine_step_id=1,
                ),
            },
        )

        with mock.patch.object(consumer, "_ensure_runtime_resources", return_value=None):
            consumer.consume_bundle(bundle, ctx)

        engine.launch_step.assert_called_once_with(source, target)
        engine.launch_forward.assert_not_called()
        engine.launch_target.assert_not_called()

    def test_runtime_bridge_can_dispatch_step_scope_bundle_into_esamp(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=4), engine=engine)
        core = type("Core", (), {})()
        runtime = type("Runtime", (), {})()
        runtime.tap_decode_hidden = {
            "layers.0": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            "layers.1": torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        }
        runtime.decode_count = 2
        runtime.decode_prompt_idxs = [7, 9]
        runtime.decode_sample_idxs = [0, 1]
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
        runtime.event_step_id = 5
        runtime.dispatch_plan = DispatchPlan.build([consumer])
        core.RUNTIME = runtime
        runner = SimpleNamespace(device=torch.device("cpu"), model=None, max_num_reqs=8)
        ctx = RuntimeContext(
            runner=runner,
            model=None,
            device=torch.device("cpu"),
            main_stream=None,
            is_compiling=False,
            uses_cudagraph=False,
            event_name="flow:decode",
        )

        with mock.patch.object(consumer, "_ensure_runtime_resources", return_value=None):
            dispatched = port_runtime_hooks.dispatch_decode_port_bundles(core=core, runner=runner)

        self.assertEqual(dispatched, 1)
        engine.launch_step.assert_called_once()
        self.assertTrue(torch.equal(engine.launch_step.call_args.args[0], runtime.tap_decode_hidden["layers.0"]))
        self.assertTrue(torch.equal(engine.launch_step.call_args.args[1], runtime.tap_decode_hidden["layers.1"]))
        engine.launch_forward.assert_not_called()
        engine.launch_target.assert_not_called()
        engine.launch_delayed_backward.assert_called_once_with(2, prompt_idxs=(7, 9))


if __name__ == "__main__":
    unittest.main()
