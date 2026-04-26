#!/usr/bin/env python3
"""Unit tests for runtime port frame assembly."""

from __future__ import annotations

import unittest
import torch

from tllm.contracts.port_bundle import BundleKey
from tllm.ports.base import ConsumerFlow
from tllm.ports.request_meta import RequestMeta
from tllm.ports.residual_stream import ResidualStream
from tllm.runtime.ports.assembler import BundleAssembler
from tllm.runtime.ports.frame import Ownership, PortFrame


class PortBundleAssemblerUnitTest(unittest.TestCase):
    def _flow(self) -> ConsumerFlow:
        return ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                ResidualStream.read(layer=-1, site="block_output", phase="decode", role="target"),
                RequestMeta.read(),
            ),
            writes=(),
            window="out_of_band",
        )

    def _key(self, request_id: str = "req-1", sample_idx: int = 0) -> BundleKey:
        return BundleKey(
            engine_step_id=7,
            phase="decode",
            request_id=request_id,
            sample_idx=sample_idx,
        )

    def test_assembler_waits_until_bundle_is_complete(self) -> None:
        assembler = BundleAssembler(self._flow())
        key = self._key()

        frames = [
            PortFrame(
                key=key,
                kind=ResidualStream.KIND,
                locator=ResidualStream.read(layer=0, site="block_output", phase="decode").locator,
                payload="source-hidden",
                ownership=Ownership.BORROWED,
                ready_window="same_step",
            ),
            PortFrame(
                key=key,
                kind=RequestMeta.KIND,
                locator=RequestMeta.read().locator,
                payload={"request_id": "req-1"},
                ownership=Ownership.STAGED,
                ready_window="same_step",
            ),
        ]

        self.assertEqual(assembler.push(frames[0]), [])
        self.assertEqual(assembler.push(frames[1]), [])

        bundles = assembler.push(
            PortFrame(
                key=key,
                kind=ResidualStream.KIND,
                locator=ResidualStream.read(layer=-1, site="block_output", phase="decode").locator,
                payload="target-hidden",
                ownership=Ownership.BORROWED,
                ready_window="same_step",
            )
        )

        self.assertEqual(len(bundles), 1)
        bundle = bundles[0]
        self.assertEqual(bundle.key.request_id, "req-1")
        self.assertEqual(bundle.entries["source"], "source-hidden")
        self.assertEqual(bundle.entries["target"], "target-hidden")
        self.assertEqual(bundle.entries["request_meta"], {"request_id": "req-1"})

    def test_assembler_does_not_mix_identities(self) -> None:
        assembler = BundleAssembler(self._flow())

        bundles = assembler.push(
            PortFrame(
                key=self._key("req-1"),
                kind=ResidualStream.KIND,
                locator=ResidualStream.read(layer=0, site="block_output", phase="decode").locator,
                payload="source-1",
                ownership=Ownership.BORROWED,
                ready_window="same_step",
            )
        )
        self.assertEqual(bundles, [])

        bundles = assembler.push(
            PortFrame(
                key=self._key("req-2"),
                kind=ResidualStream.KIND,
                locator=ResidualStream.read(layer=-1, site="block_output", phase="decode").locator,
                payload="target-2",
                ownership=Ownership.BORROWED,
                ready_window="same_step",
            )
        )
        self.assertEqual(bundles, [])
        self.assertEqual(assembler.pending_bundle_count(), 2)

    def test_assembler_can_aggregate_whole_step_bundle_when_flow_requests_step_scope(self) -> None:
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                ResidualStream.read(layer=-1, site="block_output", phase="decode", role="target"),
                RequestMeta.read(),
            ),
            writes=(),
            window="out_of_band",
            bundle_key=("engine_step_id", "phase"),
        )
        assembler = BundleAssembler(flow)

        key_a = self._key("reqA", 0)
        key_b = self._key("reqB", 1)
        frames = [
            PortFrame(
                key=key_a,
                kind=ResidualStream.KIND,
                locator=ResidualStream.read(layer=0, site="block_output", phase="decode").locator,
                payload=torch.tensor([1.0, 2.0]),
                ownership=Ownership.BORROWED,
                ready_window="same_step",
            ),
            PortFrame(
                key=key_b,
                kind=ResidualStream.KIND,
                locator=ResidualStream.read(layer=0, site="block_output", phase="decode").locator,
                payload=torch.tensor([3.0, 4.0]),
                ownership=Ownership.BORROWED,
                ready_window="same_step",
            ),
            PortFrame(
                key=key_a,
                kind=ResidualStream.KIND,
                locator=ResidualStream.read(layer=-1, site="block_output", phase="decode").locator,
                payload=torch.tensor([5.0, 6.0]),
                ownership=Ownership.BORROWED,
                ready_window="same_step",
            ),
            PortFrame(
                key=key_b,
                kind=ResidualStream.KIND,
                locator=ResidualStream.read(layer=-1, site="block_output", phase="decode").locator,
                payload=torch.tensor([7.0, 8.0]),
                ownership=Ownership.BORROWED,
                ready_window="same_step",
            ),
            PortFrame(
                key=key_a,
                kind=RequestMeta.KIND,
                locator=RequestMeta.read().locator,
                payload={"request_id": "reqA", "sample_idx": 0},
                ownership=Ownership.STAGED,
                ready_window="same_step",
            ),
            PortFrame(
                key=key_b,
                kind=RequestMeta.KIND,
                locator=RequestMeta.read().locator,
                payload={"request_id": "reqB", "sample_idx": 1},
                ownership=Ownership.STAGED,
                ready_window="same_step",
            ),
        ]

        for frame in frames:
            self.assertEqual(assembler.push(frame), [])

        bundles = assembler.finalize_pending()

        self.assertEqual(len(bundles), 1)
        bundle = bundles[0]
        self.assertEqual(bundle.key.engine_step_id, 7)
        self.assertTrue(torch.equal(bundle.entries["source"], torch.tensor([[1.0, 2.0], [3.0, 4.0]])))
        self.assertTrue(torch.equal(bundle.entries["target"], torch.tensor([[5.0, 6.0], [7.0, 8.0]])))
        self.assertEqual(bundle.entries["request_meta"], [{"request_id": "reqA", "sample_idx": 0}, {"request_id": "reqB", "sample_idx": 1}])


if __name__ == "__main__":
    unittest.main()
