#!/usr/bin/env python3
"""Unit tests for bridging current runtime state into port frames."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tllm.consumers.base import BaseConsumer
from tllm.ports.base import ConsumerFlow
from tllm.ports.residual_stream import ResidualLocator
from tllm.ports.request_meta import RequestMeta
from tllm.ports.residual_stream import ResidualStream
from tllm.runtime import residual_runtime as esamp_runtime
from tllm.runtime.vllm_patch import port_runtime_hooks
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.ports.assembler import BundleAssembler
from tllm.runtime.ports.residual_bindings import ResidualPathBinding


class RuntimePortBridgeUnitTest(unittest.TestCase):
    class _FlowConsumer(BaseConsumer):
        @property
        def consumer_id(self) -> str:
            return "flow_consumer"

        def __init__(self) -> None:
            self.seen = []
            self.feedback_calls = 0

        def flows(self):
            return [
                ConsumerFlow(
                    reads=(
                        ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                        ResidualStream.read(layer=-1, site="block_output", phase="decode", role="target"),
                        RequestMeta.read(),
                    ),
                    writes=(),
                    window="background",
                )
            ]

        def consume(self, batch, ctx) -> None:
            _ = (batch, ctx)

        def consume_bundle(self, bundle, ctx) -> None:
            self.seen.append(bundle)

        def on_tick(self, event_name, ctx) -> None:
            _ = (event_name, ctx)

        def on_step_end(self, ctx) -> None:
            _ = ctx
            self.feedback_calls += 1

    class _StepFlowConsumer(BaseConsumer):
        @property
        def consumer_id(self) -> str:
            return "step_flow_consumer"

        def __init__(self) -> None:
            self.seen = []
            self.feedback_calls = 0

        def flows(self):
            return [
                ConsumerFlow(
                    reads=(
                        ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                        ResidualStream.read(layer=-1, site="block_output", phase="decode", role="target"),
                        RequestMeta.read(),
                    ),
                    writes=(),
                    window="out_of_band",
                    bundle_key=("engine_step_id", "phase"),
                )
            ]

        def consume(self, batch, ctx) -> None:
            _ = (batch, ctx)

        def consume_bundle(self, bundle, ctx) -> None:
            self.seen.append(bundle)

        def on_tick(self, event_name, ctx) -> None:
            _ = (event_name, ctx)

        def on_step_end(self, ctx) -> None:
            _ = ctx
            self.feedback_calls += 1

    class _StepFlowConsumerWithoutFeedback(BaseConsumer):
        @property
        def consumer_id(self) -> str:
            return "step_flow_no_feedback"

        def __init__(self) -> None:
            self.seen = []

        def flows(self):
            return [
                ConsumerFlow(
                    reads=(
                        ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                        RequestMeta.read(),
                    ),
                    writes=(),
                    window="out_of_band",
                    bundle_key=("engine_step_id", "phase"),
                )
            ]

        def consume_bundle(self, bundle, ctx) -> None:
            _ = ctx
            self.seen.append(bundle)

    def test_runtime_hook_install_api_name_matches_conditional_behavior(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "tllm" / "runtime" / "vllm_patch" / "port_runtime_hooks.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("def ensure_runtime_hooks", text)

    def test_execute_model_wrapper_records_cpu_path_hotspots_when_enabled(self) -> None:
        runtime = SimpleNamespace(
            path_hotspot_enabled=True,
            launch_consumer_from_hooks=False,
            event_step_id=1,
        )
        core = SimpleNamespace(
            RUNTIME=runtime,
            MODEL_HOOK_FLAG="_installed",
            _ORIG_EXECUTE_MODEL=mock.Mock(return_value="ok"),
            _runner_uses_compilation_or_cudagraph=mock.Mock(return_value=True),
            record_path_hotspot_cpu=mock.Mock(),
        )
        runner = SimpleNamespace(model=SimpleNamespace(_installed=True))

        with mock.patch.object(port_runtime_hooks.active_targets, "runtime_has_active_targets", return_value=True), mock.patch.object(
            port_runtime_hooks._common_hooks, "dispatch_runtime_event"
        ), mock.patch.object(port_runtime_hooks, "maybe_launch_post_logits_decode_work"), mock.patch.object(
            port_runtime_hooks, "dispatch_decode_port_bundles", return_value=1
        ), mock.patch.object(port_runtime_hooks._hidden_bridge, "dispatch_deferred_layer_batches"), mock.patch.object(
            port_runtime_hooks, "time"
        ) as time_mod:
            time_mod.perf_counter.side_effect = [10.0, 10.5, 10.6, 10.7, 10.8]

            out = port_runtime_hooks.wrapped_execute_model(core=core, runner=runner, args=(), kwargs={})

        self.assertEqual(out, "ok")
        recorded = {call.args[0]: call.args[1] for call in core.record_path_hotspot_cpu.call_args_list}
        self.assertAlmostEqual(recorded["execute_model.forward_cpu"], 500.0)
        self.assertAlmostEqual(recorded["execute_model.post_logits_cpu"], 100.0)
        self.assertAlmostEqual(recorded["execute_model.dispatch_bundles_cpu"], 100.0)
        self.assertAlmostEqual(recorded["execute_model.deferred_layers_cpu"], 100.0)

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
        runtime.source_resolved_path = "layers.0"
        runtime.target_resolved_path = "layers.1"
        runtime.event_step_id = 9
        runtime.dispatch_plan = None
        return type("Core", (), {"RUNTIME": runtime})()

    def test_decode_runtime_state_bridges_into_complete_source_target_bundles(self) -> None:
        core = self._core()

        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                ResidualStream.read(layer=-1, site="block_output", phase="decode", role="target"),
                RequestMeta.read(),
            ),
            writes=(),
            window="out_of_band",
        )
        assembler = BundleAssembler(flow)

        source_frames = port_runtime_hooks.build_decode_port_frames(core=core, layer_path="layers.0")
        target_frames = port_runtime_hooks.build_decode_port_frames(core=core, layer_path="layers.1")

        bundles = []
        for frame in source_frames + target_frames:
            bundles.extend(assembler.push(frame))

        self.assertEqual(len(bundles), 2)
        by_request = {bundle.key.request_id: bundle for bundle in bundles}
        self.assertEqual(tuple(by_request["reqA"].entries["source"].tolist()), (1.0, 2.0))
        self.assertEqual(tuple(by_request["reqA"].entries["target"].tolist()), (5.0, 6.0))
        self.assertEqual(by_request["reqA"].entries["request_meta"]["prompt_idx"], 10)
        self.assertEqual(tuple(by_request["reqB"].entries["source"].tolist()), (3.0, 4.0))
        self.assertEqual(tuple(by_request["reqB"].entries["target"].tolist()), (7.0, 8.0))
        self.assertEqual(by_request["reqB"].entries["request_meta"]["sample_idx"], 1)

    def test_runtime_bridge_can_dispatch_bundles_to_flow_consumers(self) -> None:
        core = self._core()
        consumer = self._FlowConsumer()
        core.RUNTIME.dispatch_plan = DispatchPlan.build([consumer])
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        dispatched = port_runtime_hooks.dispatch_decode_port_bundles(core=core, runner=runner)

        self.assertEqual(dispatched, 2)
        self.assertEqual(len(consumer.seen), 2)
        self.assertEqual(consumer.feedback_calls, 0)
        by_request = {bundle.key.request_id: bundle for bundle in consumer.seen}
        self.assertEqual(tuple(by_request["reqA"].entries["source"].tolist()), (1.0, 2.0))
        self.assertEqual(tuple(by_request["reqB"].entries["target"].tolist()), (7.0, 8.0))

    def test_runtime_bridge_can_dispatch_step_scope_aggregated_bundle(self) -> None:
        core = self._core()
        consumer = self._StepFlowConsumer()
        core.RUNTIME.dispatch_plan = DispatchPlan.build([consumer])
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        dispatched = port_runtime_hooks.dispatch_decode_port_bundles(core=core, runner=runner)

        self.assertEqual(dispatched, 1)
        self.assertEqual(len(consumer.seen), 1)
        self.assertEqual(consumer.feedback_calls, 1)
        bundle = consumer.seen[0]
        self.assertTrue(torch.equal(bundle.entries["source"], torch.tensor([[1.0, 2.0], [3.0, 4.0]])))
        self.assertTrue(torch.equal(bundle.entries["target"], torch.tensor([[5.0, 6.0], [7.0, 8.0]])))
        self.assertEqual(bundle.entries["request_meta"], [
            {"request_id": "reqA", "prompt_idx": 10, "sample_idx": 0, "phase": "decode", "engine_step_id": 9},
            {"request_id": "reqB", "prompt_idx": 11, "sample_idx": 1, "phase": "decode", "engine_step_id": 9},
        ])

    def test_runtime_bridge_skips_row_frame_construction_for_direct_step_scope_flow(self) -> None:
        core = self._core()
        consumer = self._StepFlowConsumer()
        core.RUNTIME.dispatch_plan = DispatchPlan.build([consumer])
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        with mock.patch.object(port_runtime_hooks, "build_decode_port_frames", side_effect=AssertionError("frames should not be built")):
            dispatched = port_runtime_hooks.dispatch_decode_port_bundles(core=core, runner=runner)

        self.assertEqual(dispatched, 1)

    def test_runtime_bridge_skips_sparse_step_flow_before_building_context_or_bundle(self) -> None:
        core = self._core()
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                RequestMeta.read(),
            ),
            writes=(),
            window="background",
            bundle_key=("engine_step_id", "phase"),
            dispatch_every_n_steps=256,
        )
        consumer = mock.Mock()
        consumer.flows.return_value = [flow]
        consumer.consumer_id = "sparse"
        core.RUNTIME.dispatch_plan = DispatchPlan.build([consumer])
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        with mock.patch.object(port_runtime_hooks, "build_step_scope_port_bundle", side_effect=AssertionError("bundle should not be built")), mock.patch(
            "tllm.runtime.ports.residual_bundle_dispatch._common_hooks.build_runtime_context",
            side_effect=AssertionError("context should not be built"),
        ):
            dispatched = port_runtime_hooks.dispatch_decode_port_bundles(core=core, runner=runner)

        self.assertEqual(dispatched, 0)
        consumer.consume_bundle.assert_not_called()

    def test_runtime_reports_direct_step_scope_row_cap_overflow(self) -> None:
        core = self._core()
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                RequestMeta.read(),
            ),
            writes=(),
            window="background",
            bundle_key=("engine_step_id", "phase"),
            max_bundle_rows=1,
        )

        with self.assertRaisesRegex(RuntimeError, "flow row cap exceeded"):
            port_runtime_hooks.build_step_scope_port_bundle(core=core, flow=flow)

    def test_runtime_bridge_does_not_require_feedback_hook_for_step_scope_flow(self) -> None:
        core = self._core()
        consumer = self._StepFlowConsumerWithoutFeedback()
        core.RUNTIME.dispatch_plan = DispatchPlan.build([consumer])
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        dispatched = port_runtime_hooks.dispatch_decode_port_bundles(core=core, runner=runner)

        self.assertEqual(dispatched, 1)
        self.assertEqual(len(consumer.seen), 1)

    def test_runtime_can_build_direct_step_scope_bundle_for_source_target_flow(self) -> None:
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

        bundle = port_runtime_hooks.build_step_scope_port_bundle(core=core, flow=flow)

        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertTrue(torch.equal(bundle.entries["source"], torch.tensor([[1.0, 2.0], [3.0, 4.0]])))
        self.assertTrue(torch.equal(bundle.entries["target"], torch.tensor([[5.0, 6.0], [7.0, 8.0]])))
        self.assertEqual(bundle.entries["request_meta"], [
            {"request_id": "reqA", "prompt_idx": 10, "sample_idx": 0, "phase": "decode", "engine_step_id": 9},
            {"request_id": "reqB", "prompt_idx": 11, "sample_idx": 1, "phase": "decode", "engine_step_id": 9},
        ])

    def test_runtime_bridge_skips_unneeded_target_layer_for_single_source_flow(self) -> None:
        class _SourceOnlyConsumer(BaseConsumer):
            @property
            def consumer_id(self) -> str:
                return "source_only"

            def __init__(self) -> None:
                self.seen = []

            def flows(self):
                return [
                    ConsumerFlow(
                        reads=(
                            ResidualStream.read(layer=0, site="block_output", phase="decode", role="hidden"),
                            RequestMeta.read(),
                        ),
                        writes=(),
                        window="background",
                        bundle_key=("engine_step_id", "phase"),
                    )
                ]

            def consume(self, batch, ctx) -> None:
                _ = (batch, ctx)

            def consume_bundle(self, bundle, ctx) -> None:
                self.seen.append(bundle)

            def on_tick(self, event_name, ctx) -> None:
                _ = (event_name, ctx)

            def on_step_end(self, ctx) -> None:
                _ = ctx

        core = self._core()
        consumer = _SourceOnlyConsumer()
        core.RUNTIME.dispatch_plan = DispatchPlan.build([consumer])
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        dispatched = port_runtime_hooks.dispatch_decode_port_bundles(core=core, runner=runner)

        self.assertEqual(dispatched, 1)
        self.assertEqual(len(consumer.seen), 1)
        bundle = consumer.seen[0]
        self.assertIn("hidden", bundle.entries)
        self.assertIn("request_meta", bundle.entries)
        self.assertNotIn("target", bundle.entries)

    def test_prepare_decode_localization_raises_when_active_runtime_lacks_decode_scratch(self) -> None:
        core = type("Core", (), {})()
        runtime = type("Runtime", (), {})()
        runtime.decode_row_idx = None
        runtime.decode_valid_mask = None
        runtime.decode_count = 0
        runtime.decode_prompt_idxs = []
        runtime.decode_sample_idxs = []
        runtime.decode_request_ids = []
        core.RUNTIME = runtime
        core.pick_common_attn_metadata = staticmethod(lambda attn_metadata, spec_decode_common: attn_metadata)
        core.compute_decode_localization = staticmethod(
            lambda **kwargs: (torch.tensor([0], dtype=torch.long), [0], [0], [0])
        )
        core._resolve_prompt_sample_for_req_id = staticmethod(lambda req_id: (0, 0))

        runner = type("Runner", (), {})()
        runner._scheduler_output = object()
        runner.input_batch = type(
            "InputBatch",
            (),
            {
                "req_ids": ["reqA"],
                "num_reqs": 1,
                "req_id_to_index": {"reqA": 0},
                "num_prompt_tokens": [1],
                "num_computed_tokens_cpu": [1],
            },
        )()
        view = type(
            "View",
            (),
            {
                "logits_indices": torch.tensor([0], dtype=torch.long),
                "attn_metadata": type("Common", (), {"num_actual_tokens": 1})(),
                "spec_decode_common": None,
            },
        )()

        with self.assertRaisesRegex(RuntimeError, "decode scratch|decode_row_idx|decode_valid_mask"):
            port_runtime_hooks.prepare_decode_localization(core=core, runner=runner, out=(None, None), prepare_inputs_view=view)

    def test_prepare_decode_localization_keeps_flow_row_cap_out_of_global_decode_state(self) -> None:
        core = type("Core", (), {})()
        runtime = type("Runtime", (), {})()
        runtime.decode_row_idx = torch.zeros((3,), dtype=torch.long)
        runtime.decode_valid_mask = torch.zeros((3, 1), dtype=torch.float32)
        runtime.decode_prompt_idx_buf = torch.full((3,), -1, dtype=torch.long)
        runtime.decode_sample_idx_buf = torch.full((3,), -1, dtype=torch.long)
        runtime.decode_count = 0
        runtime.decode_prompt_idxs = []
        runtime.decode_sample_idxs = []
        runtime.decode_request_ids = []
        core.RUNTIME = runtime
        core.pick_common_attn_metadata = staticmethod(lambda attn_metadata, spec_decode_common: attn_metadata)
        seen_max_rows: list[int] = []

        def _compute_decode_localization(**kwargs):
            seen_max_rows.append(int(kwargs["max_decode_rows"]))
            return torch.tensor([4, 5, 6], dtype=torch.long), [10, 11, 12], [0, 0, 0], [0, 1, 2]

        core.compute_decode_localization = staticmethod(_compute_decode_localization)
        core._resolve_prompt_sample_for_req_id = staticmethod(lambda req_id: (0, 0))

        runner = type("Runner", (), {})()
        runner.input_batch = type(
            "InputBatch",
            (),
            {
                "req_ids": ["reqA", "reqB", "reqC"],
                "num_reqs": 3,
                "req_id_to_index": {"reqA": 0, "reqB": 1, "reqC": 2},
                "num_prompt_tokens": [1, 1, 1],
                "num_computed_tokens_cpu": [1, 1, 1],
            },
        )()
        view = type(
            "View",
            (),
            {
                "logits_indices": torch.tensor([4, 5, 6], dtype=torch.long),
                "attn_metadata": type("Common", (), {"num_actual_tokens": 8})(),
                "spec_decode_common": None,
            },
        )()

        port_runtime_hooks.prepare_decode_localization(core=core, runner=runner, out=(None, None), prepare_inputs_view=view)

        self.assertEqual(runtime.decode_count, 3)
        self.assertEqual(runtime.decode_request_ids, ["reqA", "reqB", "reqC"])
        self.assertEqual(runtime.decode_prompt_idxs, [10, 11, 12])
        self.assertEqual(runtime.decode_sample_idxs, [0, 0, 0])
        self.assertEqual(tuple(runtime.decode_row_idx.tolist()), (4, 5, 6))
        self.assertEqual(tuple(runtime.decode_valid_mask[:, 0].tolist()), (1.0, 1.0, 1.0))
        self.assertEqual(seen_max_rows, [0])

    def test_prepare_decode_localization_scans_all_decode_requests_independent_of_flow_cap(self) -> None:
        core = type("Core", (), {})()
        runtime = type("Runtime", (), {})()
        runtime.decode_row_idx = torch.zeros((3,), dtype=torch.long)
        runtime.decode_valid_mask = torch.zeros((3, 1), dtype=torch.float32)
        runtime.decode_prompt_idx_buf = torch.full((3,), -1, dtype=torch.long)
        runtime.decode_sample_idx_buf = torch.full((3,), -1, dtype=torch.long)
        runtime.decode_count = 0
        runtime.decode_prompt_idxs = []
        runtime.decode_sample_idxs = []
        runtime.decode_request_ids = []
        core.RUNTIME = runtime
        core.pick_common_attn_metadata = staticmethod(lambda attn_metadata, spec_decode_common: attn_metadata)
        seen_req_ids: list[object] = []

        def _compute_decode_localization(**kwargs):
            seen_req_ids.extend(kwargs["req_ids"])
            return torch.tensor([4, 5, 6], dtype=torch.long), [10, 11, 12], [0, 0, 0], [0, 1, 2]

        core.compute_decode_localization = staticmethod(_compute_decode_localization)
        core._resolve_prompt_sample_for_req_id = staticmethod(lambda req_id: (10, 0))

        class _CountingTokens:
            def __init__(self) -> None:
                self.seen: list[int] = []

            def __getitem__(self, index):
                self.seen.append(int(index))
                return 1

        prompt_tokens = _CountingTokens()
        computed_tokens = _CountingTokens()

        runner = type("Runner", (), {})()
        runner.input_batch = type(
            "InputBatch",
            (),
            {
                "req_ids": ["reqA", "reqB", "reqC"],
                "num_reqs": 3,
                "req_id_to_index": {"reqA": 0, "reqB": 1, "reqC": 2},
                "num_prompt_tokens": prompt_tokens,
                "num_computed_tokens_cpu": computed_tokens,
            },
        )()
        view = type(
            "View",
            (),
            {
                "logits_indices": torch.tensor([4, 5, 6], dtype=torch.long),
                "attn_metadata": type("Common", (), {"num_actual_tokens": 8})(),
                "spec_decode_common": None,
            },
        )()

        port_runtime_hooks.prepare_decode_localization(core=core, runner=runner, out=(None, None), prepare_inputs_view=view)

        self.assertEqual(runtime.decode_count, 3)
        self.assertEqual(seen_req_ids, ["reqA", "reqB", "reqC"])
        self.assertEqual(prompt_tokens.seen, [0, 1, 2])
        self.assertEqual(computed_tokens.seen, [0, 1, 2])

    def test_prepare_decode_localization_skips_device_metadata_when_no_consumer_needs_it(self) -> None:
        core = type("Core", (), {})()
        runtime = type("Runtime", (), {})()
        runtime.decode_row_idx = torch.zeros((2,), dtype=torch.long)
        runtime.decode_valid_mask = torch.zeros((2, 1), dtype=torch.float32)
        runtime.decode_prompt_idx_buf = torch.full((2,), 77, dtype=torch.long)
        runtime.decode_sample_idx_buf = torch.full((2,), 88, dtype=torch.long)
        runtime.decode_count = 0
        runtime.decode_prompt_idxs = []
        runtime.decode_sample_idxs = []
        runtime.decode_request_ids = []
        runtime.decode_prompt_idx_tensor = torch.tensor([99], dtype=torch.long)
        runtime.decode_sample_idx_tensor = torch.tensor([98], dtype=torch.long)
        runtime.dispatch_plan = SimpleNamespace(requires_device_decode_metadata=lambda: False)
        runtime.consumer = None
        core.RUNTIME = runtime
        core.pick_common_attn_metadata = staticmethod(lambda attn_metadata, spec_decode_common: attn_metadata)
        core.compute_decode_localization = staticmethod(
            lambda **_: (torch.tensor([4, 5], dtype=torch.long), [10, 11], [0, 1], [0, 1])
        )
        core._resolve_prompt_sample_for_req_id = staticmethod(lambda req_id: (0, 0))

        runner = type("Runner", (), {})()
        runner.input_batch = type(
            "InputBatch",
            (),
            {
                "req_ids": ["reqA", "reqB"],
                "num_reqs": 2,
                "req_id_to_index": {"reqA": 0, "reqB": 1},
                "num_prompt_tokens": [1, 1],
                "num_computed_tokens_cpu": [1, 1],
            },
        )()
        view = type(
            "View",
            (),
            {
                "logits_indices": torch.tensor([4, 5], dtype=torch.long),
                "attn_metadata": type("Common", (), {"num_actual_tokens": 8})(),
                "spec_decode_common": None,
            },
        )()

        port_runtime_hooks.prepare_decode_localization(core=core, runner=runner, out=(None, None), prepare_inputs_view=view)

        self.assertEqual(runtime.decode_prompt_idxs, [10, 11])
        self.assertEqual(runtime.decode_sample_idxs, [0, 1])
        self.assertIsNone(runtime.decode_prompt_idx_tensor)
        self.assertIsNone(runtime.decode_sample_idx_tensor)
        self.assertEqual(tuple(runtime.decode_prompt_idx_buf.tolist()), (77, 77))
        self.assertEqual(tuple(runtime.decode_sample_idx_buf.tolist()), (88, 88))
        self.assertEqual(tuple(runtime.decode_valid_mask[:, 0].tolist()), (0.0, 0.0))

    def test_prepare_decode_localization_keeps_first_per_prompt_compaction_flow_local(self) -> None:
        core = type("Core", (), {})()
        runtime = type("Runtime", (), {})()
        runtime.decode_row_idx = torch.zeros((4,), dtype=torch.long)
        runtime.decode_valid_mask = torch.zeros((4, 1), dtype=torch.float32)
        runtime.decode_prompt_idx_buf = torch.full((4,), -1, dtype=torch.long)
        runtime.decode_sample_idx_buf = torch.full((4,), -1, dtype=torch.long)
        runtime.decode_count = 0
        runtime.decode_prompt_idxs = []
        runtime.decode_sample_idxs = []
        runtime.decode_request_ids = []
        runtime.dispatch_plan = SimpleNamespace()
        core.RUNTIME = runtime
        core.pick_common_attn_metadata = staticmethod(lambda attn_metadata, spec_decode_common: attn_metadata)
        seen_max_rows: list[int] = []

        def _compute_decode_localization(**kwargs):
            seen_max_rows.append(int(kwargs["max_decode_rows"]))
            return torch.tensor([4, 5, 6, 7], dtype=torch.long), [10, 10, 11, 11], [0, 1, 0, 1], [0, 1, 2, 3]

        core.compute_decode_localization = staticmethod(_compute_decode_localization)
        core._resolve_prompt_sample_for_req_id = staticmethod(lambda req_id: (0, 0))

        runner = type("Runner", (), {})()
        runner.input_batch = type(
            "InputBatch",
            (),
            {
                "req_ids": ["reqA0", "reqA1", "reqB0", "reqB1"],
                "num_reqs": 4,
                "req_id_to_index": {"reqA0": 0, "reqA1": 1, "reqB0": 2, "reqB1": 3},
                "num_prompt_tokens": [1, 1, 1, 1],
                "num_computed_tokens_cpu": [1, 1, 1, 1],
            },
        )()
        view = type(
            "View",
            (),
            {
                "logits_indices": torch.tensor([4, 5, 6, 7], dtype=torch.long),
                "attn_metadata": type("Common", (), {"num_actual_tokens": 8})(),
                "spec_decode_common": None,
            },
        )()

        port_runtime_hooks.prepare_decode_localization(core=core, runner=runner, out=(None, None), prepare_inputs_view=view)

        self.assertEqual(seen_max_rows, [0])
        self.assertEqual(runtime.decode_count, 4)
        self.assertEqual(runtime.decode_request_ids, ["reqA0", "reqA1", "reqB0", "reqB1"])
        self.assertEqual(runtime.decode_prompt_idxs, [10, 10, 11, 11])
        self.assertEqual(runtime.decode_sample_idxs, [0, 1, 0, 1])
        self.assertEqual(tuple(runtime.decode_row_idx.tolist()), (4, 5, 6, 7))
        self.assertEqual(tuple(runtime.decode_valid_mask[:, 0].tolist()), (1.0, 1.0, 1.0, 1.0))

    def test_prepare_decode_localization_uses_strided_compact_rows_for_regular_prompt_groups(self) -> None:
        core = type("Core", (), {})()
        runtime = type("Runtime", (), {})()
        runtime.decode_row_idx = torch.zeros((6,), dtype=torch.long)
        runtime.decode_valid_mask = torch.zeros((6, 1), dtype=torch.float32)
        runtime.decode_prompt_idx_buf = torch.full((6,), -1, dtype=torch.long)
        runtime.decode_sample_idx_buf = torch.full((6,), -1, dtype=torch.long)
        runtime.decode_compact_row_idx = torch.full((3,), -1, dtype=torch.long)
        runtime.decode_count = 0
        runtime.decode_prompt_idxs = []
        runtime.decode_sample_idxs = []
        runtime.decode_request_ids = []
        runtime.decode_compact_count = 0
        runtime.decode_compact_row_ids = ()
        runtime.dispatch_plan = SimpleNamespace()
        core.RUNTIME = runtime
        core.pick_common_attn_metadata = staticmethod(lambda attn_metadata, spec_decode_common: attn_metadata)
        row_idx = torch.tensor([4, 5, 8, 9, 12, 13], dtype=torch.long)

        def _compute_decode_localization(**kwargs):
            return row_idx, [10, 10, 11, 11, 12, 12], [0, 1, 0, 1, 0, 1], list(range(6))

        core.compute_decode_localization = staticmethod(_compute_decode_localization)
        core._resolve_prompt_sample_for_req_id = staticmethod(lambda req_id: (0, 0))

        runner = type("Runner", (), {})()
        runner.input_batch = type(
            "InputBatch",
            (),
            {
                "req_ids": ["a0", "a1", "b0", "b1", "c0", "c1"],
                "num_reqs": 6,
                "req_id_to_index": {"a0": 0, "a1": 1, "b0": 2, "b1": 3, "c0": 4, "c1": 5},
                "num_prompt_tokens": [1, 1, 1, 1, 1, 1],
                "num_computed_tokens_cpu": [1, 1, 1, 1, 1, 1],
            },
        )()
        view = type(
            "View",
            (),
            {
                "logits_indices": torch.tensor([4, 5, 8, 9, 12, 13], dtype=torch.long),
                "attn_metadata": type("Common", (), {"num_actual_tokens": 16})(),
                "spec_decode_common": None,
            },
        )()

        with mock.patch("torch.as_tensor", wraps=torch.as_tensor) as as_tensor:
            port_runtime_hooks.prepare_decode_localization(core=core, runner=runner, out=(None, None), prepare_inputs_view=view)

        compact_as_tensor_calls = [
            call for call in as_tensor.call_args_list if list(call.args[0]) == [0, 2, 4]
        ]
        self.assertEqual(compact_as_tensor_calls, [])
        self.assertEqual(runtime.decode_compact_count, 3)
        self.assertEqual(runtime.decode_compact_row_ids, (0, 2, 4))
        self.assertEqual(tuple(runtime.decode_compact_row_idx.tolist()), (4, 8, 12))

    def test_prepare_decode_localization_leaves_inactive_tail_undefined(self) -> None:
        core = type("Core", (), {})()
        runtime = type("Runtime", (), {})()
        runtime.decode_row_idx = torch.full((3,), 99, dtype=torch.long)
        runtime.decode_valid_mask = torch.full((3, 1), 7.0, dtype=torch.float32)
        runtime.decode_prompt_idx_buf = torch.full((3,), 77, dtype=torch.long)
        runtime.decode_sample_idx_buf = torch.full((3,), 88, dtype=torch.long)
        runtime.decode_count = 0
        runtime.decode_prompt_idxs = []
        runtime.decode_sample_idxs = []
        runtime.decode_request_ids = []
        runtime.dispatch_plan = None
        core.RUNTIME = runtime
        core.pick_common_attn_metadata = staticmethod(lambda attn_metadata, spec_decode_common: attn_metadata)
        core.compute_decode_localization = staticmethod(
            lambda **kwargs: (torch.tensor([4], dtype=torch.long), [10], [0], [0])
        )
        core._resolve_prompt_sample_for_req_id = staticmethod(lambda req_id: (10, 0))
        runner = type("Runner", (), {})()
        runner.input_batch = type(
            "InputBatch",
            (),
            {
                "req_ids": ["reqA"],
                "num_reqs": 1,
                "req_id_to_index": {"reqA": 0},
                "num_prompt_tokens": [1],
                "num_computed_tokens_cpu": [1],
            },
        )()
        view = type(
            "View",
            (),
            {
                "logits_indices": torch.tensor([4], dtype=torch.long),
                "attn_metadata": type("Common", (), {"num_actual_tokens": 5})(),
                "spec_decode_common": None,
            },
        )()

        port_runtime_hooks.prepare_decode_localization(core=core, runner=runner, out=(None, None), prepare_inputs_view=view)

        self.assertEqual(runtime.decode_count, 1)
        self.assertEqual(tuple(runtime.decode_row_idx.tolist()), (4, 99, 99))
        self.assertEqual(tuple(runtime.decode_valid_mask[:, 0].tolist()), (1.0, 7.0, 7.0))
        self.assertEqual(tuple(runtime.decode_prompt_idx_buf.tolist()), (10, 77, 77))
        self.assertEqual(tuple(runtime.decode_sample_idx_buf.tolist()), (0, 88, 88))

    def test_build_step_scope_port_bundle_raises_when_active_flow_entry_is_missing(self) -> None:
        core = self._core()
        core.RUNTIME.tap_decode_hidden.pop("layers.1")
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

        with self.assertRaisesRegex(RuntimeError, "missing.*target|target.*missing"):
            port_runtime_hooks.build_step_scope_port_bundle(core=core, flow=flow)

    def test_build_tap_path_list_uses_flow_required_residual_layers(self) -> None:
        class _SourceOnlyPlan:
            def has_active_targets(self):
                return True

            def required_residual_layers(self):
                return {(0, "block_output", "decode")}

        runtime = type("Runtime", (), {})()
        runtime.config = type(
            "Config",
            (),
            {
                "tap_layer_paths": [],
                "source_layer_path": "model.model.layers[0].input_layernorm",
                "target_layer_path": "model.model.layers[-1].input_layernorm",
            },
        )()
        runtime.dispatch_plan = _SourceOnlyPlan()
        core = type("Core", (), {"RUNTIME": runtime})()

        paths = port_runtime_hooks.build_tap_path_list(core=core)

        self.assertEqual(paths, ["model.model.layers[0].input_layernorm"])

    def test_build_tap_path_list_prefers_runtime_residual_raw_paths_over_cfg_names(self) -> None:
        class _SourceOnlyPlan:
            def has_active_targets(self):
                return True

            def required_residual_layers(self):
                return {(0, "block_output", "decode")}

        runtime = type("Runtime", (), {})()
        runtime.config = type(
            "Config",
            (),
            {
                "tap_layer_paths": [],
                "source_layer_path": "wrong.source.path",
                "target_layer_path": "wrong.target.path",
            },
        )()
        runtime.residual_raw_paths = {
            ResidualLocator(layer=0, site="block_output", phase="decode"): "right.source.path",
            ResidualLocator(layer=-1, site="block_output", phase="decode"): "right.target.path",
        }
        runtime.dispatch_plan = _SourceOnlyPlan()
        core = type("Core", (), {"RUNTIME": runtime})()

        paths = port_runtime_hooks.build_tap_path_list(core=core)

        self.assertEqual(paths, ["right.source.path"])

    def test_setup_runtime_hooks_if_active_allows_missing_target_for_source_only_flow(self) -> None:
        class _SourceOnlyPlan:
            def has_active_targets(self):
                return True

            def required_residual_layers(self):
                return {(0, "block_output", "decode")}

        class _Layer(torch.nn.Module):
            def forward(self, x):
                return x

        model = type("Model", (), {})()
        layer0 = _Layer()
        setattr(model, "model", type("Inner", (), {"layers": [type("Block", (), {"input_layernorm": layer0})()]})())
        setattr(model, "config", type("Cfg", (), {"hidden_size": 4})())

        runtime = type("Runtime", (), {})()
        runtime.config = type(
            "Config",
            (),
            {
                "tap_layer_paths": [],
                "source_layer_path": "model.model.layers[0].input_layernorm",
                "target_layer_path": "model.model.layers[-1].input_layernorm",
                "graph_scratch_rows": 8,
                "distiller_hidden_dim": 8,
                "distiller_lr": 1e-3,
                "per_request_models": False,
                "per_request_model_bank": False,
                "model_bank_slots": 0,
                "model_bank_flush_interval": 1,
                "model_bank_rank": 8,
                "model_bank_use_output_layernorm": True,
                "model_bank_initializer": None,
                "model_bank_train_cudagraph": False,
                "enable_esamp_training": False,
            },
        )()
        runtime.decode_row_idx = None
        runtime.decode_valid_mask = None
        runtime.tap_layers = {}
        runtime.tap_scratch = {}
        runtime.tap_decode_hidden = {}
        runtime.source_resolved_path = ""
        runtime.target_resolved_path = ""
        runtime.launch_consumer_from_hooks = True
        runtime.consumer = None
        runtime.dispatch_plan = _SourceOnlyPlan()
        core = type(
            "Core",
            (),
            {
                "RUNTIME": runtime,
                "MODEL_HOOK_FLAG": "esamp_hook_installed",
                "MODEL_HOOK_SPEC_ATTR": "esamp_hook_spec",
                "_resolve_module_by_path_with_fallback": staticmethod(esamp_runtime.resolve_module_by_path_with_fallback),
                "_infer_hidden_dtype": staticmethod(lambda layer: torch.float32),
                "_runner_uses_compilation_or_cudagraph": staticmethod(lambda runner: False),
            },
        )()
        runner = type("Runner", (), {"model": model, "device": torch.device("cpu"), "max_num_reqs": 8})()

        port_runtime_hooks.setup_runtime_hooks_if_active(core=core, runner=runner)

        self.assertTrue(core.RUNTIME.source_resolved_path.endswith("layers[0].input_layernorm"))
        self.assertEqual(core.RUNTIME.target_resolved_path, "")

    def test_setup_runtime_hooks_if_active_noops_when_no_active_targets_exist(self) -> None:
        class _Layer(torch.nn.Module):
            def forward(self, x):
                return x

        model = type("Model", (), {})()
        layer0 = _Layer()
        setattr(model, "model", type("Inner", (), {"layers": [type("Block", (), {"input_layernorm": layer0})()]})())
        setattr(model, "config", type("Cfg", (), {"hidden_size": 4})())

        runtime = type("Runtime", (), {})()
        runtime.config = type(
            "Config",
            (),
            {
                "tap_layer_paths": [],
                "source_layer_path": "model.model.layers[0].input_layernorm",
                "target_layer_path": "model.model.layers[-1].input_layernorm",
                "graph_scratch_rows": 8,
                "distiller_hidden_dim": 8,
                "distiller_lr": 1e-3,
                "per_request_models": False,
                "per_request_model_bank": False,
                "model_bank_slots": 0,
                "model_bank_flush_interval": 1,
                "model_bank_rank": 8,
                "model_bank_use_output_layernorm": True,
                "model_bank_initializer": None,
                "model_bank_train_cudagraph": False,
                "enable_esamp_training": False,
            },
        )()
        runtime.residual_raw_paths = {}
        runtime.decode_row_idx = None
        runtime.decode_valid_mask = None
        runtime.tap_layers = {}
        runtime.tap_scratch = {}
        runtime.tap_decode_hidden = {}
        runtime.residual_bindings = {}
        runtime.source_resolved_path = ""
        runtime.target_resolved_path = ""
        runtime.launch_consumer_from_hooks = True
        runtime.consumer = None
        runtime.dispatch_plan = None
        core = type(
            "Core",
            (),
            {
                "RUNTIME": runtime,
                "MODEL_HOOK_FLAG": "esamp_hook_installed",
                "MODEL_HOOK_SPEC_ATTR": "esamp_hook_spec",
                "_resolve_module_by_path_with_fallback": staticmethod(esamp_runtime.resolve_module_by_path_with_fallback),
                "_infer_hidden_dtype": staticmethod(lambda layer: torch.float32),
                "_runner_uses_compilation_or_cudagraph": staticmethod(lambda runner: False),
            },
        )()
        runner = type("Runner", (), {"model": model, "device": torch.device("cpu"), "max_num_reqs": 8})()

        port_runtime_hooks.setup_runtime_hooks_if_active(core=core, runner=runner)

        self.assertFalse(getattr(model, core.MODEL_HOOK_FLAG, False))
        self.assertEqual(core.RUNTIME.tap_layers, {})
        self.assertEqual(core.RUNTIME.tap_decode_hidden, {})

    def test_setup_runtime_hooks_if_active_does_not_reensure_resources_when_hooks_already_installed(self) -> None:
        model = type("Model", (), {})()
        setattr(model, "esamp_hook_installed", True)
        setattr(model, "esamp_hook_spec", ("hook",))
        setattr(model, "_tllm_compute_logits_wrapped", True)
        consumer = mock.Mock()
        runtime = type("Runtime", (), {})()
        runtime.config = type(
            "Config",
            (),
            {
                "tap_layer_paths": [],
                "source_layer_path": "model.model.layers[0].input_layernorm",
                "target_layer_path": "model.model.layers[-1].input_layernorm",
                "graph_scratch_rows": 8,
                "distiller_hidden_dim": 8,
                "distiller_lr": 1e-3,
                "per_request_models": False,
                "per_request_model_bank": False,
                "model_bank_slots": 0,
                "model_bank_flush_interval": 1,
                "model_bank_rank": 8,
                "model_bank_use_output_layernorm": True,
                "model_bank_initializer": None,
                "model_bank_train_cudagraph": False,
                "enable_esamp_training": False,
            },
        )()
        runtime.launch_consumer_from_hooks = True
        runtime.consumer = consumer
        runtime.dispatch_plan = SimpleNamespace(has_active_targets=lambda: True)
        core = type(
            "Core",
            (),
            {
                "RUNTIME": runtime,
                "MODEL_HOOK_FLAG": "esamp_hook_installed",
                "MODEL_HOOK_SPEC_ATTR": "esamp_hook_spec",
                "_runner_uses_compilation_or_cudagraph": staticmethod(lambda runner: True),
            },
        )()
        runner = type("Runner", (), {"model": model, "device": torch.device("cpu"), "max_num_reqs": 8})()

        with mock.patch.object(port_runtime_hooks._residual_runtime_setup, "resolve_runtime_setup") as p_setup, mock.patch.object(
            port_runtime_hooks._sampler_patch,
            "ensure_sampler_precompute_buffers",
        ) as p_ensure_buffers:
            p_setup.return_value = SimpleNamespace(
                resolved_layers={},
                target_resolved="",
                hook_spec=("hook",),
            )
            port_runtime_hooks.setup_runtime_hooks_if_active(core=core, runner=runner)

        consumer._ensure_runtime_resources.assert_not_called()
        p_ensure_buffers.assert_not_called()
        self.assertFalse(runtime.launch_consumer_from_hooks)


if __name__ == "__main__":
    unittest.main()
