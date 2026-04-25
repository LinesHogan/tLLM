#!/usr/bin/env python3
"""Unit tests for public consumer flow contracts."""

from __future__ import annotations

import unittest

from tllm.consumers.base import BaseConsumer
from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.ports.base import ConsumerFlow, Locator, PortKind, PortRead, PortWrite


class _ExampleConsumer(BaseConsumer):
    @property
    def consumer_id(self) -> str:
        return "example"

    def flows(self):
        return [
            ConsumerFlow(
                reads=(PortRead(kind=PortKind.REQUEST_META, locator=Locator()),),
                writes=(PortWrite(kind=PortKind.CPU_EXPORT, locator=Locator()),),
                window="background",
            )
        ]


class ConsumerFlowContractsUnitTest(unittest.TestCase):
    def test_consumer_flow_requires_reads_writes_and_window(self) -> None:
        flow = ConsumerFlow(
            reads=(PortRead(kind=PortKind.REQUEST_META, locator=Locator()),),
            writes=(PortWrite(kind=PortKind.CPU_EXPORT, locator=Locator()),),
            window="background",
        )

        self.assertEqual(flow.window, "background")
        self.assertEqual(flow.reads[0].kind, PortKind.REQUEST_META)
        self.assertEqual(flow.writes[0].kind, PortKind.CPU_EXPORT)
        self.assertEqual(flow.dispatch_every_n_steps, 1)
        self.assertEqual(flow.max_bundle_rows, 0)

    def test_consumer_flow_can_declare_sparse_step_dispatch(self) -> None:
        flow = ConsumerFlow(
            reads=(PortRead(kind=PortKind.REQUEST_META, locator=Locator()),),
            writes=(),
            window="background",
            bundle_key=("engine_step_id", "phase"),
            dispatch_every_n_steps=256,
        )

        self.assertEqual(flow.dispatch_every_n_steps, 256)

    def test_consumer_flow_can_declare_max_step_bundle_rows(self) -> None:
        flow = ConsumerFlow(
            reads=(PortRead(kind=PortKind.REQUEST_META, locator=Locator()),),
            writes=(),
            window="background",
            bundle_key=("engine_step_id", "phase"),
            max_bundle_rows=1,
        )

        self.assertEqual(flow.max_bundle_rows, 1)

    def test_port_bundle_carries_identity_and_entries(self) -> None:
        bundle = PortBundle(
            key=BundleKey(
                engine_step_id=3,
                phase="decode",
                request_id="req-1",
                sample_idx=2,
            ),
            entries={
                "meta": PortRead(kind=PortKind.REQUEST_META, locator=Locator()),
            },
        )

        self.assertEqual(bundle.key.engine_step_id, 3)
        self.assertEqual(bundle.key.request_id, "req-1")
        self.assertIn("meta", bundle.entries)

    def test_base_consumer_allows_flow_only_consumers(self) -> None:
        consumer = _ExampleConsumer()

        self.assertTrue(consumer.flows())
        self.assertEqual(consumer.consumer_id, "example")


if __name__ == "__main__":
    unittest.main()
