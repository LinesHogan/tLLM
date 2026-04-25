#!/usr/bin/env python3
"""Unit tests for the current ESamp public consumer contract."""

from __future__ import annotations

from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig
from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.runtime_context import RuntimeContext
from tllm.ports.base import PortKind
from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime import residual_runtime as esamp_runtime
from tllm.workflows import side_train_support


class ESampConsumerContractUnitTest(unittest.TestCase):
    def tearDown(self) -> None:
        esamp_runtime.clear_dispatch_consumers()

    @staticmethod
    def _ctx(*, runner_max_num_reqs: int = 8, model=None) -> RuntimeContext:
        return RuntimeContext(
            runner=SimpleNamespace(max_num_reqs=runner_max_num_reqs),
            model=model,
            device=torch.device("cpu"),
            main_stream=None,
            is_compiling=False,
            uses_cudagraph=False,
            event_name="flow:decode",
        )

    def test_flows_cover_source_target_and_request_metadata(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(), engine=mock.Mock())
        flows = consumer.flows()

        self.assertEqual(len(flows), 1)
        flow = flows[0]
        self.assertEqual(flow.window, "out_of_band_train")
        self.assertEqual(
            [read.kind for read in flow.reads],
            [PortKind.RESIDUAL_STREAM, PortKind.RESIDUAL_STREAM, PortKind.REQUEST_META],
        )
        self.assertEqual([read.role for read in flow.reads[:2]], ["source", "target"])
        self.assertEqual(flow.bundle_key, ("engine_step_id", "phase"))

    def test_flows_are_empty_when_side_train_is_disabled(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(enable_side_train=False), engine=mock.Mock())

        self.assertEqual(tuple(consumer.flows()), ())

    def test_consumer_uses_single_formal_engine_without_env_switch(self) -> None:
        config = ESampConsumerConfig()
        consumer = ESampConsumer(config)

        self.assertEqual(type(consumer._engine).__name__, "ESampTrainEngine")

    def test_update_config_reuses_existing_engine_and_calls_configure(self) -> None:
        class _Engine:
            def __init__(self) -> None:
                self.calls: list[ESampConsumerConfig] = []

            def configure(self, config: ESampConsumerConfig) -> None:
                self.calls.append(config)

        engine = _Engine()
        initial = ESampConsumerConfig(side_hidden_dim=32, enable_side_train=True)
        updated = ESampConsumerConfig(side_hidden_dim=16, enable_side_train=False)
        consumer = ESampConsumer(initial, engine=engine)

        consumer.update_config(updated)

        self.assertIs(consumer._engine, engine)
        self.assertIs(consumer.config, updated)
        self.assertEqual(engine.calls, [updated])

    def test_esamp_workflows_no_longer_import_consumer_runtime_adapter(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        for rel in (
            "tllm/workflows/benchmarks/side_train_benchmark.py",
            "tllm/workflows/benchmarks/per_request_side_train_benchmark.py",
            "tllm/workflows/repro/repro_side_train_loss.py",
        ):
            text = (repo_root / rel).read_text(encoding="utf-8")
            self.assertNotIn("tllm.consumers.esamp.runtime_adapter", text, msg=rel)
            self.assertIn("residual_runtime", text, msg=rel)

    def test_consume_bundle_routes_source_target_and_feedback_into_engine(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=4), engine=engine)
        source_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        target_hidden = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
        ctx = self._ctx()
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="decode", request_id="reqA", sample_idx=0),
            entries={
                "source": source_hidden,
                "target": target_hidden,
                "request_meta": [
                    {"prompt_idx": 7, "sample_idx": 0},
                    {"prompt_idx": 9, "sample_idx": 1},
                ],
            },
        )

        with mock.patch.object(consumer, "_ensure_runtime_resources", return_value=None) as p_prepare:
            consumer._source_stage = torch.zeros((4, 2), dtype=torch.float32)
            consumer._target_stage = torch.zeros((4, 2), dtype=torch.float32)
            consumer.consume_bundle(bundle, ctx)

        p_prepare.assert_called_once_with(ctx, rows_hidden=source_hidden)
        self.assertEqual(engine.launch_forward.call_count, 1)
        self.assertEqual(engine.launch_target.call_count, 1)
        engine.launch_delayed_backward.assert_not_called()

        consumer.apply_feedback(ctx)
        engine.launch_delayed_backward.assert_called_once_with(2, prompt_idxs=[7, 9])

    def test_model_bank_consume_bundle_stages_one_row_per_prompt(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(
            ESampConsumerConfig(
                graph_scratch_rows=6,
                per_request_models=True,
                per_request_model_bank=True,
            ),
            engine=engine,
        )
        source_hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        target_hidden = source_hidden + 100.0
        ctx = self._ctx()
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="decode", request_id="reqA", sample_idx=0),
            entries={
                "source": source_hidden,
                "target": target_hidden,
                "request_meta": [
                    {"prompt_idx": 7, "sample_idx": 0},
                    {"prompt_idx": 7, "sample_idx": 1},
                    {"prompt_idx": 7, "sample_idx": 2},
                    {"prompt_idx": 9, "sample_idx": 0},
                    {"prompt_idx": 9, "sample_idx": 1},
                    {"prompt_idx": 9, "sample_idx": 2},
                ],
            },
        )

        with mock.patch.object(consumer, "_ensure_runtime_resources", return_value=None):
            consumer._source_stage = torch.zeros((6, 2), dtype=torch.float32)
            consumer._target_stage = torch.zeros((6, 2), dtype=torch.float32)
            consumer.consume_bundle(bundle, ctx)

        staged_source = engine.launch_forward.call_args.args[0]
        staged_target = engine.launch_target.call_args.args[0]
        self.assertTrue(torch.equal(staged_source[:2], source_hidden[[0, 3]]))
        self.assertTrue(torch.equal(staged_target[:2], target_hidden[[0, 3]]))

        consumer.apply_feedback(ctx)
        engine.launch_delayed_backward.assert_called_once_with(2, prompt_idxs=[7, 9])

    def test_consume_bundle_rejects_mismatched_source_target_row_counts(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=4), engine=engine)
        ctx = self._ctx()
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="decode", request_id="reqA", sample_idx=0),
            entries={
                "source": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
                "target": torch.tensor([[5.0, 6.0]], dtype=torch.float32),
                "request_meta": [{"prompt_idx": 7, "sample_idx": 0}],
            },
        )

        with self.assertRaisesRegex(RuntimeError, "row count"):
            consumer.consume_bundle(bundle, ctx)

    def test_consumer_ensure_runtime_resources_raises_when_hidden_metadata_cannot_be_inferred(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(), engine=mock.Mock())

        with self.assertRaisesRegex(RuntimeError, "hidden_size|hidden_dtype"):
            consumer._ensure_runtime_resources(self._ctx(model=None), rows_hidden=None)

    def test_consumer_ensure_runtime_resources_raises_when_rows_cannot_be_resolved(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=0), engine=mock.Mock())
        rows_hidden = torch.zeros((0, 8), dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "graph_scratch_rows|rows"):
            consumer._ensure_runtime_resources(self._ctx(runner_max_num_reqs=0, model=None), rows_hidden=rows_hidden)

    def test_consumer_ensure_runtime_resources_keeps_initializer_optional(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=4), engine=engine)
        rows_hidden = torch.randn((2, 8), dtype=torch.float32)
        model = SimpleNamespace(config=SimpleNamespace(hidden_size=0))

        with mock.patch.object(
            consumer,
            "_resolve_layer_with_fallback",
            side_effect=[(torch.nn.Identity(), "source.path"), (torch.nn.Identity(), "target.path")],
        ), mock.patch.object(consumer, "_maybe_prepare_initializer", wraps=consumer._maybe_prepare_initializer) as p_prepare:
            consumer._ensure_runtime_resources(self._ctx(model=model), rows_hidden=rows_hidden)

        engine.ensure_resources.assert_called_once()
        p_prepare.assert_called_once()

    def test_consumer_sync_config_no_longer_mutates_engine_state_directly(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "consumer.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("self._engine.state.model_bank_initializer", text)

    def test_stage_rows_requires_prepared_stage_buffer(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(), engine=mock.Mock())
        rows_hidden = torch.randn((2, 8), dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "stage buffer"):
            consumer._stage_rows(rows_hidden, stage=None)

    def test_stage_rows_rejects_shape_mismatch_after_resources_are_prepared(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(), engine=mock.Mock())
        rows_hidden = torch.randn((2, 8), dtype=torch.float32)
        wrong_stage = torch.zeros((1, 8), dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "stage buffer"):
            consumer._stage_rows(rows_hidden, stage=wrong_stage)

    def test_configure_esamp_runtime_installs_esamp_consumer(self) -> None:
        side_train_support.configure_esamp_runtime(
            graph_scratch_rows=64,
            tap_layer_paths=["model.model.layers[0]", "model.model.layers[-1]"],
            source_layer_path="model.model.layers[0]",
            target_layer_path="model.model.layers[-1]",
            enable_side_train=True,
            side_hidden_dim=128,
            side_lr=1e-3,
        )
        self.assertIsInstance(esamp_runtime.RUNTIME.consumer, ESampConsumer)
        self.assertIsNotNone(esamp_runtime.RUNTIME.dispatch_plan)
        self.assertEqual(
            esamp_runtime.RUNTIME.residual_raw_paths,
            {
                ResidualLocator(layer=0, site="block_output", phase="decode"): "model.model.layers[0]",
                ResidualLocator(layer=-1, site="block_output", phase="decode"): "model.model.layers[-1]",
            },
        )
        targets = esamp_runtime.RUNTIME.dispatch_plan.flow_targets()
        self.assertTrue(targets)
        self.assertIs(targets[0].consumer, esamp_runtime.RUNTIME.consumer)

    def test_configure_esamp_runtime_with_disabled_side_train_leaves_no_active_dispatch_plan(self) -> None:
        side_train_support.configure_esamp_runtime(
            graph_scratch_rows=64,
            tap_layer_paths=["model.model.layers[0]", "model.model.layers[-1]"],
            source_layer_path="model.model.layers[0]",
            target_layer_path="model.model.layers[-1]",
            enable_side_train=False,
            side_hidden_dim=128,
            side_lr=1e-3,
        )

        self.assertIsInstance(esamp_runtime.RUNTIME.consumer, ESampConsumer)
        self.assertIsNone(esamp_runtime.RUNTIME.dispatch_plan)

    def test_configure_esamp_runtime_keeps_single_formal_engine_type(self) -> None:
        side_train_support.configure_esamp_runtime(
            graph_scratch_rows=64,
            tap_layer_paths=["model.model.layers[0]", "model.model.layers[-1]"],
            source_layer_path="model.model.layers[0]",
            target_layer_path="model.model.layers[-1]",
            enable_side_train=True,
            side_hidden_dim=128,
            side_lr=1e-3,
        )
        self.assertEqual(type(esamp_runtime.RUNTIME.consumer._engine).__name__, "ESampTrainEngine")

        side_train_support.configure_esamp_runtime(
            graph_scratch_rows=64,
            tap_layer_paths=["model.model.layers[0]", "model.model.layers[-1]"],
            source_layer_path="model.model.layers[0]",
            target_layer_path="model.model.layers[-1]",
            enable_side_train=True,
            side_hidden_dim=128,
            side_lr=1e-3,
        )

        self.assertEqual(type(esamp_runtime.RUNTIME.consumer._engine).__name__, "ESampTrainEngine")

    def test_configure_esamp_runtime_synchronizes_existing_consumer_before_update(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig())
        calls: list[str] = []
        old_consumer = esamp_runtime.RUNTIME.consumer
        try:
            esamp_runtime.RUNTIME.consumer = consumer
            with mock.patch.object(consumer, "synchronize", side_effect=lambda: calls.append("sync")), mock.patch.object(
                consumer, "update_config", side_effect=lambda config: calls.append(f"update:{config.enable_side_train}")
            ), mock.patch.object(side_train_support.runtime, "clear_dispatch_consumers"), mock.patch.object(
                side_train_support.runtime, "set_runtime_consumer"
            ), mock.patch.object(side_train_support.runtime, "configure_runtime"):
                side_train_support.configure_esamp_runtime(
                    graph_scratch_rows=64,
                    tap_layer_paths=["model.model.layers[0]", "model.model.layers[-1]"],
                    source_layer_path="model.model.layers[0]",
                    target_layer_path="model.model.layers[-1]",
                    enable_side_train=False,
                    side_hidden_dim=128,
                    side_lr=1e-3,
                )
        finally:
            esamp_runtime.RUNTIME.consumer = old_consumer

        self.assertEqual(calls, ["sync", "update:False"])


if __name__ == "__main__":
    unittest.main()
