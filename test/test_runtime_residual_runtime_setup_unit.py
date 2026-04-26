#!/usr/bin/env python3
"""Unit tests for generic residual runtime setup extraction."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.ports.base import ConsumerFlow
from tllm.ports.request_meta import RequestMeta
from tllm.ports.residual_stream import ResidualStream
from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.ports import residual_runtime_setup


class RuntimeResidualRuntimeSetupUnitTest(unittest.TestCase):
    def _core(self):
        runtime = type("Runtime", (), {})()
        runtime.config = type(
            "Config",
            (),
            {
                "tap_layer_paths": [],
                "source_layer_path": "model.model.layers[0].input_layernorm",
                "target_layer_path": "model.model.layers[-1].input_layernorm",
                "graph_scratch_rows": 8,
                "compact_capture_lane": False,
                "enable_distiller_intervention": False,
            },
        )()
        runtime.residual_raw_paths = {
            ResidualLocator(layer=0, site="block_output", phase="decode"): "model.model.layers[0].input_layernorm",
            ResidualLocator(layer=-1, site="block_output", phase="decode"): "model.model.layers[-1].input_layernorm",
        }
        runtime.dispatch_plan = None
        runtime.tap_layers = {}
        runtime.tap_scratch = {}
        runtime.tap_decode_hidden = {}
        runtime.residual_bindings = {}
        runtime.source_resolved_path = ""
        runtime.target_resolved_path = ""
        runtime.launch_consumer_from_hooks = True
        runtime.decode_row_idx = None
        runtime.decode_valid_mask = None
        return type(
            "Core",
            (),
            {
                "RUNTIME": runtime,
                "_resolve_module_by_path_with_fallback": staticmethod(lambda model, raw_path: (model.mapping[raw_path], model.alias[raw_path])),
                "_infer_hidden_dtype": staticmethod(lambda layer: torch.float32),
                "_runner_uses_compilation_or_cudagraph": staticmethod(lambda runner: False),
            },
        )()

    def _model(self):
        layer0 = torch.nn.Linear(4, 4)
        layer1 = torch.nn.Linear(4, 4)
        model = type("Model", (), {})()
        model.mapping = {
            "model.model.layers[0].input_layernorm": layer0,
            "model.model.layers[-1].input_layernorm": layer1,
        }
        model.alias = {
            "model.model.layers[0].input_layernorm": "layers.0",
            "model.model.layers[-1].input_layernorm": "layers.1",
        }
        model.config = type("Cfg", (), {"hidden_size": 4})()
        return model, layer0, layer1

    def test_resolve_runtime_setup_builds_bindings_and_sizes(self) -> None:
        core = self._core()
        model, _, _ = self._model()
        runner = type("Runner", (), {"model": model, "device": torch.device("cpu"), "max_num_reqs": 8})()

        setup = residual_runtime_setup.resolve_runtime_setup(core=core, runner=runner)

        self.assertEqual(setup.rows, 8)
        self.assertEqual(setup.hidden_size, 4)
        self.assertEqual(setup.hidden_dtype, torch.float32)
        self.assertEqual(setup.source_resolved, "layers.0")
        self.assertEqual(setup.target_resolved, "layers.1")
        self.assertEqual(tuple(sorted(setup.resolved_layers.keys())), ("layers.0", "layers.1"))

    def test_resolve_runtime_setup_preserves_runner_capacity_when_residual_flows_are_capped(self) -> None:
        core = self._core()
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="hidden"),
                RequestMeta.read(),
            ),
            writes=(),
            window="background",
            bundle_key=("engine_step_id", "phase"),
            max_bundle_rows=1,
        )
        consumer = mock.Mock()
        consumer.flows.return_value = [flow]
        consumer.consumer_id = "row_cap"
        core.RUNTIME.dispatch_plan = DispatchPlan.build([consumer])
        core.RUNTIME.config.compact_capture_lane = True
        model, _, _ = self._model()
        runner = type("Runner", (), {"model": model, "device": torch.device("cpu"), "max_num_reqs": 8})()

        setup = residual_runtime_setup.resolve_runtime_setup(core=core, runner=runner)

        self.assertEqual(setup.rows, 8)

    def test_apply_runtime_setup_allocates_compact_capture_buffers_for_compact_flow(self) -> None:
        core = self._core()
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="hidden"),
                RequestMeta.read(),
            ),
            writes=(),
            window="background",
            bundle_key=("engine_step_id", "phase"),
            row_compaction="first_per_prompt",
            max_bundle_rows=2,
        )
        consumer = mock.Mock()
        consumer.flows.return_value = [flow]
        consumer.consumer_id = "compact_rows"
        core.RUNTIME.dispatch_plan = DispatchPlan.build([consumer])
        core.RUNTIME.config.compact_capture_lane = True
        model, _, _ = self._model()
        runner = type("Runner", (), {"model": model, "device": torch.device("cpu"), "max_num_reqs": 8})()
        setup = residual_runtime_setup.resolve_runtime_setup(core=core, runner=runner)

        with mock.patch.object(residual_runtime_setup._residual_capture_hooks, "install_layer_forward_taps") as p_install:
            residual_runtime_setup.apply_runtime_setup(core=core, runner=runner, setup=setup)

        self.assertEqual(tuple(core.RUNTIME.decode_compact_row_idx.shape), (2,))
        self.assertEqual(core.RUNTIME.decode_compact_count, 0)
        self.assertIn("layers.0", core.RUNTIME.tap_decode_hidden_compact)
        self.assertEqual(tuple(core.RUNTIME.tap_decode_hidden_compact["layers.0"].shape), (2, 4))
        p_install.assert_called_once()

    def test_apply_runtime_setup_populates_runtime_state_and_installs_hooks(self) -> None:
        core = self._core()
        model, _, _ = self._model()
        runner = type("Runner", (), {"model": model, "device": torch.device("cpu"), "max_num_reqs": 8})()
        setup = residual_runtime_setup.resolve_runtime_setup(core=core, runner=runner)

        with mock.patch.object(residual_runtime_setup._residual_capture_hooks, "install_layer_forward_taps") as p_install:
            residual_runtime_setup.apply_runtime_setup(core=core, runner=runner, setup=setup)

        self.assertEqual(core.RUNTIME.source_resolved_path, "layers.0")
        self.assertEqual(core.RUNTIME.target_resolved_path, "layers.1")
        self.assertIn("layers.0", core.RUNTIME.residual_bindings)
        self.assertEqual(tuple(core.RUNTIME.decode_row_idx.shape), (8,))
        self.assertEqual(tuple(core.RUNTIME.decode_valid_mask.shape), (8, 1))
        p_install.assert_called_once()


if __name__ == "__main__":
    unittest.main()
