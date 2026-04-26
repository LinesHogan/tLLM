#!/usr/bin/env python3
"""Unit tests for residual bundle dispatch helpers extracted from runtime hooks."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.contracts.gpu_stage import DeviceTensorLease
from tllm.contracts.request_meta_view import RowBatchMeta
from tllm.ports.base import ConsumerFlow
from tllm.ports.request_meta import RequestMeta
from tllm.ports.residual_stream import ResidualLocator, ResidualStream
from tllm.runtime import hidden_event_bridge
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.ports import residual_bundle_dispatch


class RuntimeResidualBundleDispatchUnitTest(unittest.TestCase):
    def _core(self):
        runtime = type("Runtime", (), {})()
        runtime.tap_decode_hidden = {
            "layers.0": torch.tensor([[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]], dtype=torch.float32),
            "layers.1": torch.tensor([[5.0, 6.0], [7.0, 8.0], [88.0, 88.0]], dtype=torch.float32),
        }
        runtime.decode_count = 2
        runtime.decode_prompt_idxs = [10, 11]
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
        runtime.event_step_id = 9
        return type("Core", (), {"RUNTIME": runtime})()

    def _core_with_duplicate_prompt_rows(self):
        core = self._core()
        core.RUNTIME.decode_count = 3
        core.RUNTIME.decode_prompt_idxs = [10, 10, 11]
        core.RUNTIME.decode_sample_idxs = [0, 1, 0]
        core.RUNTIME.decode_request_ids = ["reqA-0", "reqA-1", "reqB-0"]
        return core

    def test_build_step_scope_port_bundle_uses_residual_binding_table(self) -> None:
        core = self._core()
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

        bundle = residual_bundle_dispatch.build_step_scope_port_bundle(core=core, flow=flow)

        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertTrue(torch.equal(bundle.entries["source"], torch.tensor([[1.0, 2.0], [3.0, 4.0]])))
        self.assertTrue(torch.equal(bundle.entries["target"], torch.tensor([[5.0, 6.0], [7.0, 8.0]])))

    def test_build_device_lease_step_scope_bundle_wraps_residual_entries_in_lease(self) -> None:
        core = self._core()
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                ResidualStream.read(layer=-1, site="block_output", phase="decode", role="target"),
                RequestMeta.read(),
            ),
            writes=(),
            window="out_of_band",
            bundle_key=("engine_step_id", "phase"),
            delivery="device_lease",
            ownership="runtime_lease",
        )

        bundle = residual_bundle_dispatch.build_step_scope_port_bundle(core=core, flow=flow)

        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertIn("device_lease", bundle.entries)
        lease = bundle.entries["device_lease"]
        self.assertIsInstance(lease, DeviceTensorLease)
        self.assertTrue(torch.equal(lease.entries["source"], torch.tensor([[1.0, 2.0], [3.0, 4.0]])))
        self.assertTrue(torch.equal(lease.entries["target"], torch.tensor([[5.0, 6.0], [7.0, 8.0]])))
        self.assertEqual(lease.active_rows, 2)
        self.assertEqual(lease.ownership, "runtime_lease")
        self.assertEqual(lease.lifetime, "consume_call")
        self.assertIn("request_meta", bundle.entries)
        self.assertIsInstance(bundle.entries["request_meta"], RowBatchMeta)
        meta = bundle.entries["request_meta"]
        self.assertEqual(meta.prompt_idxs, (10, 11))
        self.assertEqual(meta.sample_idxs, (0, 1))
        self.assertEqual(meta.request_ids, ("reqA", "reqB"))
        self.assertEqual(meta.phase, "decode")
        self.assertEqual(meta.engine_step_id, 9)

    def test_first_per_prompt_compaction_shapes_device_lease_and_metadata(self) -> None:
        core = self._core_with_duplicate_prompt_rows()
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                ResidualStream.read(layer=-1, site="block_output", phase="decode", role="target"),
                RequestMeta.read(),
            ),
            writes=(),
            window="out_of_band",
            bundle_key=("engine_step_id", "phase"),
            delivery="device_lease",
            ownership="runtime_lease",
            row_compaction="first_per_prompt",
        )

        bundle = residual_bundle_dispatch.build_step_scope_port_bundle(core=core, flow=flow)

        self.assertIsNotNone(bundle)
        assert bundle is not None
        lease = bundle.entries["device_lease"]
        self.assertIsInstance(lease, DeviceTensorLease)
        self.assertEqual(lease.active_rows, 2)
        self.assertTrue(torch.equal(lease.entries["source"], torch.tensor([[1.0, 2.0], [99.0, 99.0]])))
        self.assertTrue(torch.equal(lease.entries["target"], torch.tensor([[5.0, 6.0], [88.0, 88.0]])))
        self.assertEqual(lease.entries["source"].untyped_storage().data_ptr(), core.RUNTIME.tap_decode_hidden["layers.0"].untyped_storage().data_ptr())
        self.assertEqual(lease.entries["target"].untyped_storage().data_ptr(), core.RUNTIME.tap_decode_hidden["layers.1"].untyped_storage().data_ptr())
        meta = bundle.entries["request_meta"]
        self.assertIsInstance(meta, RowBatchMeta)
        self.assertEqual(meta.request_ids, ("reqA-0", "reqB-0"))
        self.assertEqual(meta.prompt_idxs, (10, 11))
        self.assertEqual(meta.sample_idxs, (0, 0))
        self.assertEqual(meta.row_compaction, "first_per_prompt")
        self.assertEqual(meta.row_ids, (0, 2))

    def test_first_per_prompt_compaction_builds_metadata_without_full_row_materialization(self) -> None:
        core = self._core_with_duplicate_prompt_rows()
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                RequestMeta.read(),
            ),
            writes=(),
            window="out_of_band",
            bundle_key=("engine_step_id", "phase"),
            delivery="device_lease",
            ownership="runtime_lease",
            row_compaction="first_per_prompt",
        )

        with mock.patch.object(
            residual_bundle_dispatch,
            "active_request_prompt_sample_metadata",
            side_effect=AssertionError("full-row metadata should not be materialized for compact flow"),
        ):
            bundle = residual_bundle_dispatch.build_step_scope_port_bundle(core=core, flow=flow)

        self.assertIsNotNone(bundle)
        assert bundle is not None
        meta = bundle.entries["request_meta"]
        self.assertIsInstance(meta, RowBatchMeta)
        self.assertEqual(meta.request_ids, ("reqA-0", "reqB-0"))
        self.assertEqual(meta.prompt_idxs, (10, 11))
        self.assertEqual(meta.row_ids, (0, 2))

    def test_first_per_prompt_compaction_prefers_runtime_compact_capture_lane(self) -> None:
        core = self._core_with_duplicate_prompt_rows()
        core.RUNTIME.decode_compact_count = 2
        core.RUNTIME.decode_compact_row_ids = (0, 2)
        core.RUNTIME.tap_decode_hidden_compact = {
            "layers.0": torch.tensor([[101.0, 102.0], [103.0, 104.0]], dtype=torch.float32),
            "layers.1": torch.tensor([[105.0, 106.0], [107.0, 108.0]], dtype=torch.float32),
        }
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                ResidualStream.read(layer=-1, site="block_output", phase="decode", role="target"),
                RequestMeta.read(),
            ),
            writes=(),
            window="out_of_band",
            bundle_key=("engine_step_id", "phase"),
            delivery="device_lease",
            ownership="runtime_lease",
            row_compaction="first_per_prompt",
        )

        bundle = residual_bundle_dispatch.build_step_scope_port_bundle(core=core, flow=flow)

        self.assertIsNotNone(bundle)
        assert bundle is not None
        lease = bundle.entries["device_lease"]
        self.assertIsInstance(lease, DeviceTensorLease)
        self.assertEqual(lease.active_rows, 2)
        self.assertTrue(torch.equal(lease.entries["source"], torch.tensor([[101.0, 102.0], [103.0, 104.0]])))
        self.assertTrue(torch.equal(lease.entries["target"], torch.tensor([[105.0, 106.0], [107.0, 108.0]])))

    def test_first_per_prompt_compaction_does_not_change_hidden_batch_global_rows(self) -> None:
        core = self._core_with_duplicate_prompt_rows()
        core.RUNTIME.decode_row_idx = torch.tensor([4, 5, 6], dtype=torch.long)
        core.RUNTIME.decode_valid_mask = torch.ones((3, 1), dtype=torch.float32)
        core.RUNTIME.decode_prompt_idx_tensor = torch.tensor([10, 10, 11], dtype=torch.long)
        core.RUNTIME.decode_sample_idx_tensor = torch.tensor([0, 1, 0], dtype=torch.long)
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                RequestMeta.read(),
            ),
            writes=(),
            window="out_of_band",
            bundle_key=("engine_step_id", "phase"),
            delivery="device_lease",
            ownership="runtime_lease",
            row_compaction="first_per_prompt",
        )

        bundle = residual_bundle_dispatch.build_step_scope_port_bundle(core=core, flow=flow)
        batch = hidden_event_bridge.build_runtime_hidden_batch(core=core, layer_path="layers.0")

        self.assertIsNotNone(bundle)
        self.assertIsNotNone(batch)
        assert bundle is not None
        assert batch is not None
        lease = bundle.entries["device_lease"]
        self.assertIsInstance(lease, DeviceTensorLease)
        self.assertEqual(lease.active_rows, 2)
        self.assertEqual(tuple(batch.prompt_idx.tolist()), (10, 10, 11))
        self.assertEqual(tuple(batch.sample_idx.tolist()), (0, 1, 0))
        self.assertEqual(tuple(batch.row_idx.tolist()), (4, 5, 6))
        self.assertEqual(int(batch.rows_hidden.shape[0]), 3)

    def test_first_per_prompt_row_cap_overflow_is_explicit(self) -> None:
        core = self._core_with_duplicate_prompt_rows()
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                RequestMeta.read(),
            ),
            writes=(),
            window="out_of_band",
            bundle_key=("engine_step_id", "phase"),
            delivery="device_lease",
            ownership="runtime_lease",
            row_compaction="first_per_prompt",
            max_bundle_rows=1,
        )

        with self.assertRaisesRegex(RuntimeError, "flow row cap exceeded"):
            residual_bundle_dispatch.build_step_scope_port_bundle(core=core, flow=flow)

    def test_default_step_scope_bundle_keeps_list_request_meta_for_compatibility(self) -> None:
        core = self._core()
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                RequestMeta.read(),
            ),
            writes=(),
            window="background",
            bundle_key=("engine_step_id", "phase"),
        )

        bundle = residual_bundle_dispatch.build_step_scope_port_bundle(core=core, flow=flow)

        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertIsInstance(bundle.entries["request_meta"], list)
        self.assertEqual(bundle.entries["request_meta"][0]["prompt_idx"], 10)

    def test_row_batch_meta_rejects_mismatched_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            RowBatchMeta(
                request_ids=("reqA",),
                prompt_idxs=(10, 11),
                sample_idxs=(0,),
                phase="decode",
                engine_step_id=9,
            )

    def test_build_step_scope_port_bundle_rejects_inconsistent_decode_metadata(self) -> None:
        core = self._core()
        core.RUNTIME.decode_request_ids = ["reqA"]
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                RequestMeta.read(),
            ),
            writes=(),
            window="out_of_band",
            bundle_key=("engine_step_id", "phase"),
        )

        with self.assertRaisesRegex(RuntimeError, "decode runtime metadata is inconsistent"):
            residual_bundle_dispatch.build_step_scope_port_bundle(core=core, flow=flow)


if __name__ == "__main__":
    unittest.main()
