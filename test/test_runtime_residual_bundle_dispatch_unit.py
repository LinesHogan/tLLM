#!/usr/bin/env python3
"""Unit tests for residual bundle dispatch helpers extracted from runtime hooks."""

from __future__ import annotations

import unittest

import torch

from tllm.ports.base import ConsumerFlow
from tllm.ports.request_meta import RequestMeta
from tllm.ports.residual_stream import ResidualLocator, ResidualStream
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

    def test_build_step_scope_port_bundle_uses_residual_binding_table(self) -> None:
        core = self._core()
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                ResidualStream.read(layer=-1, site="block_output", phase="decode", role="target"),
                RequestMeta.read(),
            ),
            writes=(),
            window="out_of_band_train",
            bundle_key=("engine_step_id", "phase"),
        )

        bundle = residual_bundle_dispatch.build_step_scope_port_bundle(core=core, flow=flow)

        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertTrue(torch.equal(bundle.entries["source"], torch.tensor([[1.0, 2.0], [3.0, 4.0]])))
        self.assertTrue(torch.equal(bundle.entries["target"], torch.tensor([[5.0, 6.0], [7.0, 8.0]])))

    def test_build_step_scope_port_bundle_rejects_inconsistent_decode_metadata(self) -> None:
        core = self._core()
        core.RUNTIME.decode_request_ids = ["reqA"]
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                RequestMeta.read(),
            ),
            writes=(),
            window="out_of_band_train",
            bundle_key=("engine_step_id", "phase"),
        )

        with self.assertRaisesRegex(RuntimeError, "decode runtime metadata is inconsistent"):
            residual_bundle_dispatch.build_step_scope_port_bundle(core=core, flow=flow)


if __name__ == "__main__":
    unittest.main()
