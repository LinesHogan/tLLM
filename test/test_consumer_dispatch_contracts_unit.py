#!/usr/bin/env python3
"""Unit tests for producer-consumer contracts and runtime dispatch plan."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import List

import torch

from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.hidden_batch import HiddenBatch
from tllm.contracts.runtime_context import RuntimeContext
from tllm.contracts.subscription import ConsumerSubscription
from tllm.consumers.base import BaseConsumer
from tllm.ports.base import ConsumerFlow, Locator, PortKind, PortRead
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.vllm_patch import common_hooks


@dataclass
class _DummyConsumer(BaseConsumer):
    cid: str
    subs: List[ConsumerSubscription]

    @property
    def consumer_id(self) -> str:
        return self.cid

    def subscriptions(self) -> List[ConsumerSubscription]:
        return list(self.subs)

    def consume(self, batch: HiddenBatch, ctx: RuntimeContext) -> None:
        _ = (batch, ctx)

    def on_tick(self, event_name: str, ctx: RuntimeContext) -> None:
        _ = (event_name, ctx)

    def on_step_end(self, ctx: RuntimeContext) -> None:
        _ = ctx


@dataclass
class _FlowConsumer(BaseConsumer):
    cid: str
    seen_bundles: List[PortBundle]

    @property
    def consumer_id(self) -> str:
        return self.cid

    def flows(self):
        return [
            ConsumerFlow(
                reads=(PortRead(kind=PortKind.REQUEST_META, locator=Locator()),),
                writes=(),
                window="background",
            )
        ]

    def consume(self, batch: HiddenBatch, ctx: RuntimeContext) -> None:
        _ = (batch, ctx)

    def consume_bundle(self, bundle: PortBundle, ctx: RuntimeContext) -> None:
        self.seen_bundles.append(bundle)

    def on_tick(self, event_name: str, ctx: RuntimeContext) -> None:
        _ = (event_name, ctx)

    def on_step_end(self, ctx: RuntimeContext) -> None:
        _ = ctx


class ConsumerDispatchContractsUnitTest(unittest.TestCase):
    def test_hidden_batch_holds_minimal_payload_fields(self) -> None:
        x = torch.zeros((2, 4), dtype=torch.float32)
        batch = HiddenBatch(
            step_id=3,
            phase="decode",
            layer_path="model.model.layers[0]",
            rows_hidden=x,
            row_idx=torch.tensor([0, 4], dtype=torch.long),
            valid_mask=torch.tensor([1.0, 0.0], dtype=torch.float32),
            prompt_idx=torch.tensor([7, -1], dtype=torch.long),
            sample_idx=torch.tensor([0, -1], dtype=torch.long),
            metadata={"source": "unit"},
        )
        self.assertEqual(batch.step_id, 3)
        self.assertEqual(batch.phase, "decode")
        self.assertEqual(tuple(batch.rows_hidden.shape), (2, 4))
        self.assertEqual(batch.metadata.get("source"), "unit")

    def test_dispatch_plan_filters_event_phase_layer_and_capture_policy(self) -> None:
        c1 = _DummyConsumer(
            cid="c1",
            subs=[
                ConsumerSubscription(
                    consumer_id="c1",
                    event_name="layer.post",
                    phase_filter="decode",
                    layer_filter="model.model.layers[0]",
                    capture_policy="prefer",
                    dispatch_mode="inline",
                )
            ],
        )
        c2 = _DummyConsumer(
            cid="c2",
            subs=[
                ConsumerSubscription(
                    consumer_id="c2",
                    event_name="layer.post",
                    phase_filter="prefill",
                    layer_filter=None,
                    capture_policy="never",
                    dispatch_mode="consumer_async",
                )
            ],
        )

        plan = DispatchPlan.build([c1, c2])

        selected = plan.select(
            event_name="layer.post",
            phase="decode",
            layer_path="model.model.layers[0]",
            capture_enabled=True,
        )
        self.assertEqual([x.consumer.consumer_id for x in selected], ["c1"])

        selected_no_capture = plan.select(
            event_name="layer.post",
            phase="prefill",
            layer_path="model.model.layers[3]",
            capture_enabled=False,
        )
        self.assertEqual([x.consumer.consumer_id for x in selected_no_capture], ["c2"])

    def test_dispatch_plan_keeps_flow_targets_separate_from_event_targets(self) -> None:
        flow_consumer = _FlowConsumer(cid="flow", seen_bundles=[])
        plan = DispatchPlan.build([flow_consumer])

        self.assertEqual(plan.select(event_name="layer.post", phase="decode", layer_path="x", capture_enabled=True), [])
        flow_targets = plan.flow_targets()
        self.assertEqual(len(flow_targets), 1)
        self.assertIs(flow_targets[0].consumer, flow_consumer)
        self.assertEqual(flow_targets[0].flow.window, "background")

    def test_runtime_can_dispatch_port_bundle_to_flow_consumer(self) -> None:
        flow_consumer = _FlowConsumer(cid="flow", seen_bundles=[])
        runtime = type("Runtime", (), {})()
        runtime.dispatch_plan = DispatchPlan.build([flow_consumer])

        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="decode", request_id="req", sample_idx=0),
            entries={"request_meta": {"request_id": "req"}},
        )

        dispatched = common_hooks.dispatch_port_bundle(
            runtime=runtime,
            runner=runner,
            bundle=bundle,
            window="background",
        )

        self.assertEqual(dispatched, 1)
        self.assertEqual(len(flow_consumer.seen_bundles), 1)
        self.assertEqual(flow_consumer.seen_bundles[0].key.request_id, "req")

    def test_dispatch_plan_reports_required_residual_layers_from_flow_consumers(self) -> None:
        @dataclass
        class _ResidualFlowConsumer(BaseConsumer):
            cid: str

            @property
            def consumer_id(self) -> str:
                return self.cid

            def flows(self):
                return [
                    ConsumerFlow(
                        reads=(PortRead(kind=PortKind.RESIDUAL_STREAM, locator=type("L", (), {"layer": 0, "site": "block_output", "phase": "decode"})()),),
                        writes=(),
                        window="background",
                    )
                ]

            def consume(self, batch: HiddenBatch, ctx: RuntimeContext) -> None:
                _ = (batch, ctx)

            def on_tick(self, event_name: str, ctx: RuntimeContext) -> None:
                _ = (event_name, ctx)

            def on_step_end(self, ctx: RuntimeContext) -> None:
                _ = ctx

        flow_consumer = _ResidualFlowConsumer(cid="flow")
        plan = DispatchPlan.build([flow_consumer])

        required = plan.required_residual_layers()

        self.assertEqual(required, {(0, "block_output", "decode")})


if __name__ == "__main__":
    unittest.main()
