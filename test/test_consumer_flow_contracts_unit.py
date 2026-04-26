#!/usr/bin/env python3
"""Unit tests for public consumer flow contracts."""

from __future__ import annotations

import unittest

from tllm.consumers.base import BaseConsumer
from tllm import clear_consumers, contracts, register_consumer
from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.runtime import residual_runtime
from tllm.ports.base import ConsumerFlow, FlowDelivery, FlowWindow, Locator, PortKind, PortRead, PortWrite


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
    def tearDown(self) -> None:
        residual_runtime.clear_dispatch_consumers()

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
        self.assertEqual(flow.delivery, "bundle")
        self.assertEqual(flow.ownership, "borrowed")
        self.assertEqual(flow.row_compaction, "none")

    def test_public_register_consumer_wrapper_updates_runtime_registry(self) -> None:
        consumer = _ExampleConsumer()

        register_consumer(consumer)

        assert residual_runtime.RUNTIME.consumer_registry is not None
        self.assertIn(consumer, residual_runtime.RUNTIME.consumer_registry.consumers())
        clear_consumers()
        self.assertEqual(tuple(residual_runtime.RUNTIME.consumer_registry.consumers()), ())

    def test_consumer_flow_can_opt_into_device_lease_runtime_lease(self) -> None:
        flow = ConsumerFlow(
            reads=(PortRead(kind=PortKind.REQUEST_META, locator=Locator()),),
            writes=(),
            window="background",
            bundle_key=("engine_step_id", "phase"),
            delivery="device_lease",
            ownership="runtime_lease",
        )

        self.assertEqual(flow.delivery, "device_lease")
        self.assertEqual(flow.ownership, "runtime_lease")

    def test_consumer_flow_accepts_public_enum_values(self) -> None:
        flow = ConsumerFlow(
            reads=(),
            writes=(),
            window=FlowWindow.OUT_OF_BAND,
            delivery=FlowDelivery.BUNDLE,
        )

        self.assertEqual(str(flow.window), "out_of_band")
        self.assertEqual(str(flow.delivery), "bundle")

    def test_device_lease_delivery_requires_runtime_lease_ownership(self) -> None:
        with self.assertRaisesRegex(ValueError, "device_lease delivery requires ownership"):
            ConsumerFlow(
                reads=(PortRead(kind=PortKind.REQUEST_META, locator=Locator()),),
                writes=(),
                window="background",
                delivery="device_lease",
            )

    def test_device_lease_delivery_requires_step_scope_bundle_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "device_lease delivery currently requires"):
            ConsumerFlow(
                reads=(PortRead(kind=PortKind.REQUEST_META, locator=Locator()),),
                writes=(),
                window="background",
                delivery="device_lease",
                ownership="runtime_lease",
                bundle_key=("request_id",),
            )

    def test_first_per_prompt_compaction_requires_step_scope_bundle_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "first_per_prompt row compaction currently requires"):
            ConsumerFlow(
                reads=(PortRead(kind=PortKind.REQUEST_META, locator=Locator()),),
                writes=(),
                window="background",
                row_compaction="first_per_prompt",
                bundle_key=("request_id",),
            )

    def test_consumer_flow_can_declare_neutral_out_of_band_window(self) -> None:
        flow = ConsumerFlow(
            reads=(PortRead(kind=PortKind.REQUEST_META, locator=Locator()),),
            writes=(),
            window="out_of_band",
        )

        self.assertEqual(flow.window, "out_of_band")

    def test_consumer_flow_can_request_first_row_per_prompt_delivery(self) -> None:
        flow = ConsumerFlow(
            reads=(PortRead(kind=PortKind.REQUEST_META, locator=Locator()),),
            writes=(),
            window="background",
            bundle_key=("engine_step_id", "phase"),
            row_compaction="first_per_prompt",
        )

        self.assertEqual(flow.row_compaction, "first_per_prompt")

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

    def test_consumer_flow_preserves_positional_bundle_key_compatibility(self) -> None:
        flow = ConsumerFlow(
            (PortRead(kind=PortKind.REQUEST_META, locator=Locator()),),
            (),
            "background",
            ("engine_step_id", "phase"),
        )

        self.assertEqual(flow.bundle_key, ("engine_step_id", "phase"))
        self.assertEqual(flow.delivery, "bundle")

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

    def test_contract_package_exports_advanced_delivery_types(self) -> None:
        self.assertEqual(contracts.DeviceTensorLease.__name__, "DeviceTensorLease")
        self.assertEqual(contracts.RowBatchMeta.__name__, "RowBatchMeta")


if __name__ == "__main__":
    unittest.main()
