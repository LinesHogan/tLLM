#!/usr/bin/env python3
"""Unit tests for the dummy consumer extension template."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.hidden_batch import HiddenBatch
from tllm.contracts.runtime_context import RuntimeContext
from tllm.consumers.dummy.config import DummyConsumerConfig
from tllm.consumers.dummy.consumer import DummyConsumer
from tllm.consumers.dummy.worker import DummyCpuWorker
from tllm.ports.base import PortKind


class DummyConsumerUnitTest(unittest.TestCase):
    def test_dummy_declares_background_flow_over_residual_and_request_meta(self) -> None:
        c = DummyConsumer(DummyConsumerConfig())
        flows = c.flows()

        self.assertEqual(len(flows), 1)
        self.assertFalse(hasattr(c, "consume"))
        self.assertFalse(hasattr(c, "on_tick"))
        flow = flows[0]
        self.assertEqual(flow.window, "background")
        self.assertEqual([read.kind for read in flow.reads], [PortKind.RESIDUAL_STREAM, PortKind.REQUEST_META])
        self.assertEqual([write.kind for write in flow.writes], [PortKind.CPU_EXPORT])
        self.assertEqual(flow.bundle_key, ("engine_step_id", "phase"))

    def test_dummy_can_disable_cpu_export_for_pure_counting_baseline(self) -> None:
        c = DummyConsumer(DummyConsumerConfig(export_to_cpu=False))
        flow = c.flows()[0]

        self.assertEqual(tuple(flow.writes), ())

    def test_dummy_default_path_stages_cpu_without_hot_path_drain(self) -> None:
        c = DummyConsumer(DummyConsumerConfig(export_every_n_steps=1))
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="decode", request_id="reqA", sample_idx=0),
            entries={
                "hidden": torch.randn((2, 4), dtype=torch.float32),
                "request_meta": {"request_id": "reqA", "prompt_idx": 0, "sample_idx": 0},
            },
        )
        ctx = RuntimeContext(
            runner=None,
            model=None,
            device=torch.device("cpu"),
            main_stream=None,
            is_compiling=False,
            uses_cudagraph=False,
            event_name="flow:background",
        )

        with mock.patch("tllm.consumers.dummy.consumer.clone_hidden_to_cpu", wraps=lambda x: x.detach().to("cpu")) as p_clone, mock.patch(
            "builtins.print"
        ) as p_print:
            c.consume_bundle(bundle, ctx)
            hot_path_stats = c.read_stats()

        self.assertTrue(p_clone.called)
        self.assertFalse(p_print.called)
        self.assertEqual(hot_path_stats["consumed_batches"], 1.0)
        self.assertEqual(hot_path_stats["consumed_rows"], 2.0)
        self.assertEqual(hot_path_stats["processed_batches"], 0.0)
        self.assertEqual(hot_path_stats["pending"], 1.0)

        with mock.patch("builtins.print") as p_print:
            c.synchronize()

        stats = c.read_stats()
        self.assertTrue(p_print.called)
        self.assertEqual(stats["processed_batches"], 1.0)
        self.assertEqual(stats["pending"], 0.0)

    def test_dummy_default_cpu_export_stages_only_a_small_hidden_sample(self) -> None:
        c = DummyConsumer(DummyConsumerConfig(export_every_n_steps=1))
        hidden = torch.randn((4, 32), dtype=torch.float32)
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="decode", request_id="reqA", sample_idx=0),
            entries={
                "hidden": hidden,
                "request_meta": {"request_id": "reqA", "prompt_idx": 0, "sample_idx": 0},
            },
        )
        ctx = RuntimeContext(
            runner=None,
            model=None,
            device=torch.device("cpu"),
            main_stream=None,
            is_compiling=False,
            uses_cudagraph=False,
            event_name="flow:background",
        )

        seen_shapes: list[tuple[int, ...]] = []

        def _stage(x: torch.Tensor) -> torch.Tensor:
            seen_shapes.append(tuple(x.shape))
            return x.detach().to("cpu")

        with mock.patch("tllm.consumers.dummy.consumer.clone_hidden_to_cpu", wraps=_stage):
            c.consume_bundle(bundle, ctx)

        self.assertEqual(seen_shapes, [(1, 16)])
        self.assertEqual(c.read_stats()["consumed_rows"], 4.0)

    def test_dummy_default_export_stride_is_sparse_for_generation_throughput(self) -> None:
        c = DummyConsumer(DummyConsumerConfig())
        self.assertEqual(c.config.export_every_n_steps, 256)

    def test_dummy_default_path_drops_cpu_export_when_queue_is_full_instead_of_draining(self) -> None:
        c = DummyConsumer(DummyConsumerConfig(export_every_n_steps=1, max_queue_size=1))
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="decode", request_id="reqA", sample_idx=0),
            entries={
                "hidden": torch.randn((2, 4), dtype=torch.float32),
                "request_meta": {"request_id": "reqA", "prompt_idx": 0, "sample_idx": 0},
            },
        )
        ctx = RuntimeContext(
            runner=None,
            model=None,
            device=torch.device("cpu"),
            main_stream=None,
            is_compiling=False,
            uses_cudagraph=False,
            event_name="flow:background",
        )

        with mock.patch("tllm.consumers.dummy.consumer.clone_hidden_to_cpu", wraps=lambda x: x.detach().to("cpu")) as p_clone, mock.patch(
            "tllm.consumers.dummy.consumer.DummyCpuWorker.drain"
        ) as p_drain:
            c.consume_bundle(bundle, ctx)
            c.consume_bundle(bundle, ctx)

        self.assertEqual(p_clone.call_count, 1)
        p_drain.assert_not_called()
        stats = c.read_stats()
        self.assertEqual(stats["pending"], 1.0)
        self.assertEqual(stats["dropped_batches"], 1.0)

    def test_dummy_can_skip_intermediate_steps_via_export_stride(self) -> None:
        c = DummyConsumer(DummyConsumerConfig(enable_async=False, export_every_n_steps=4))
        ctx = RuntimeContext(
            runner=None,
            model=None,
            device=torch.device("cpu"),
            main_stream=None,
            is_compiling=False,
            uses_cudagraph=False,
            event_name="flow:background",
        )
        skipped_bundle = PortBundle(
            key=BundleKey(engine_step_id=3, phase="decode", request_id="reqA", sample_idx=0),
            entries={"hidden": torch.randn((2, 4), dtype=torch.float32), "request_meta": {}},
        )
        taken_bundle = PortBundle(
            key=BundleKey(engine_step_id=4, phase="decode", request_id="reqA", sample_idx=0),
            entries={"hidden": torch.randn((2, 4), dtype=torch.float32), "request_meta": {}},
        )

        c.consume_bundle(skipped_bundle, ctx)
        c.consume_bundle(taken_bundle, ctx)

        stats = c.read_stats()
        self.assertEqual(stats["consumed_batches"], 1.0)

    def test_dummy_consume_bundle_records_stats(self) -> None:
        c = DummyConsumer(DummyConsumerConfig(enable_async=False, export_every_n_steps=1))
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="decode", request_id="reqA", sample_idx=0),
            entries={
                "hidden": torch.randn((2, 4), dtype=torch.float32),
                "request_meta": {"request_id": "reqA", "prompt_idx": 0, "sample_idx": 0},
            },
        )
        ctx = RuntimeContext(
            runner=None,
            model=None,
            device=torch.device("cpu"),
            main_stream=None,
            is_compiling=False,
            uses_cudagraph=False,
            event_name="flow:background",
        )
        c.consume_bundle(bundle, ctx)
        c.on_step_end(ctx)

        stats = c.read_stats()
        self.assertEqual(stats["consumed_batches"], 1)
        self.assertEqual(stats["feedback_calls"], 1)

    def test_dummy_async_bundle_path_stages_cpu_and_prints_noise_summary_when_enabled(self) -> None:
        c = DummyConsumer(DummyConsumerConfig(enable_async=True, export_to_cpu=True, export_every_n_steps=1, export_max_rows=0, export_max_cols=0))
        bundle = PortBundle(
            key=BundleKey(engine_step_id=2, phase="decode", request_id="reqA", sample_idx=0),
            entries={
                "hidden": torch.full((2, 3), 1.0, dtype=torch.float32),
                "request_meta": [
                    {"request_id": "reqA", "prompt_idx": 0, "sample_idx": 0},
                    {"request_id": "reqB", "prompt_idx": 0, "sample_idx": 1},
                ],
            },
        )
        ctx = RuntimeContext(
            runner=None,
            model=None,
            device=torch.device("cpu"),
            main_stream=None,
            is_compiling=False,
            uses_cudagraph=False,
            event_name="flow:background",
        )

        with mock.patch("tllm.consumers.dummy.consumer.clone_hidden_to_cpu", wraps=lambda x: x.detach().to("cpu")) as p_clone, mock.patch(
            "builtins.print"
        ) as p_print, mock.patch("torch.randn_like", return_value=torch.ones((2, 3), dtype=torch.float32)):
            c.consume_bundle(bundle, ctx)
            c.synchronize()

        self.assertTrue(p_clone.called)
        self.assertTrue(p_print.called)
        stats = c.read_stats()
        self.assertEqual(stats["consumed_batches"], 1)
        self.assertEqual(stats["processed_batches"], 1)
        self.assertEqual(stats["processed_rows"], 2)
        self.assertGreater(stats["last_noise_std"], 0.0)

    def test_dummy_async_flow_bundle_drains_on_feedback_when_cpu_export_enabled(self) -> None:
        c = DummyConsumer(
            DummyConsumerConfig(enable_async=True, export_to_cpu=True, feedback_interval=1, export_every_n_steps=1, export_max_rows=0, export_max_cols=0)
        )
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="decode", request_id="reqA", sample_idx=0),
            entries={
                "hidden": torch.full((2, 3), 1.0, dtype=torch.float32),
                "request_meta": {"request_id": "reqA", "prompt_idx": 0, "sample_idx": 0},
            },
        )
        ctx = RuntimeContext(
            runner=None,
            model=None,
            device=torch.device("cpu"),
            main_stream=None,
            is_compiling=False,
            uses_cudagraph=False,
            event_name="flow:background",
        )

        with mock.patch("tllm.consumers.dummy.consumer.clone_hidden_to_cpu", wraps=lambda x: x.detach().to("cpu")) as p_clone, mock.patch(
            "builtins.print"
        ) as p_print, mock.patch("torch.randn_like", return_value=torch.ones((2, 3), dtype=torch.float32)):
            c.consume_bundle(bundle, ctx)
            before = c.read_stats()
            c.on_step_end(ctx)
            after = c.read_stats()

        self.assertTrue(p_clone.called)
        self.assertEqual(before["pending"], 1.0)
        self.assertEqual(after["pending"], 0.0)
        self.assertEqual(after["processed_batches"], 1.0)
        self.assertTrue(p_print.called)

    def test_dummy_step_scope_bundle_emits_one_summary_for_aggregated_rows_when_cpu_export_enabled(self) -> None:
        c = DummyConsumer(
            DummyConsumerConfig(enable_async=True, export_to_cpu=True, feedback_interval=1, export_every_n_steps=1, export_max_rows=0, export_max_cols=0)
        )
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="decode", request_id="reqA", sample_idx=0),
            entries={
                "hidden": torch.ones((8, 4), dtype=torch.float32),
                "request_meta": [
                    {"request_id": "reqA", "prompt_idx": 0, "sample_idx": 0},
                    {"request_id": "reqB", "prompt_idx": 1, "sample_idx": 1},
                ],
            },
        )
        ctx = RuntimeContext(
            runner=None,
            model=None,
            device=torch.device("cpu"),
            main_stream=None,
            is_compiling=False,
            uses_cudagraph=False,
            event_name="flow:background",
        )

        with mock.patch("builtins.print") as p_print, mock.patch("torch.randn_like", return_value=torch.ones((8, 4), dtype=torch.float32)):
            c.consume_bundle(bundle, ctx)
            c.on_step_end(ctx)

        stats = c.read_stats()
        self.assertEqual(stats["consumed_batches"], 1.0)
        self.assertEqual(stats["processed_batches"], 1.0)
        self.assertEqual(stats["processed_rows"], 8.0)
        self.assertEqual(p_print.call_count, 1)

    def test_dummy_async_feedback_interval_reduces_drain_frequency_when_cpu_export_enabled(self) -> None:
        c = DummyConsumer(
            DummyConsumerConfig(enable_async=True, export_to_cpu=True, feedback_interval=4, export_every_n_steps=1, export_max_rows=0, export_max_cols=0)
        )
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="decode", request_id="reqA", sample_idx=0),
            entries={
                "hidden": torch.ones((2, 3), dtype=torch.float32),
                "request_meta": {"request_id": "reqA", "prompt_idx": 0, "sample_idx": 0},
            },
        )
        ctx = RuntimeContext(
            runner=None,
            model=None,
            device=torch.device("cpu"),
            main_stream=None,
            is_compiling=False,
            uses_cudagraph=False,
            event_name="flow:background",
        )

        with mock.patch("torch.randn_like", return_value=torch.ones((2, 3), dtype=torch.float32)):
            for _ in range(3):
                c.consume_bundle(bundle, ctx)
                c.on_step_end(ctx)
            mid = c.read_stats()
            c.consume_bundle(bundle, ctx)
            c.on_step_end(ctx)
            end = c.read_stats()

        self.assertEqual(mid["pending"], 3.0)
        self.assertEqual(mid["processed_batches"], 0.0)
        self.assertEqual(end["pending"], 0.0)
        self.assertEqual(end["processed_batches"], 4.0)

    def test_dummy_async_can_defer_all_drain_until_synchronize_when_cpu_export_enabled(self) -> None:
        c = DummyConsumer(
            DummyConsumerConfig(
                enable_async=True,
                export_to_cpu=True,
                feedback_interval=0,
                max_queue_size=4096,
                export_every_n_steps=1,
                export_max_rows=0,
                export_max_cols=0,
            )
        )
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="decode", request_id="reqA", sample_idx=0),
            entries={
                "hidden": torch.ones((2, 3), dtype=torch.float32),
                "request_meta": {"request_id": "reqA", "prompt_idx": 0, "sample_idx": 0},
            },
        )
        ctx = RuntimeContext(
            runner=None,
            model=None,
            device=torch.device("cpu"),
            main_stream=None,
            is_compiling=False,
            uses_cudagraph=False,
            event_name="flow:background",
        )

        with mock.patch("builtins.print") as p_print, mock.patch("torch.randn_like", return_value=torch.ones((2, 3), dtype=torch.float32)):
            for _ in range(4):
                c.consume_bundle(bundle, ctx)
                c.on_step_end(ctx)
            mid = c.read_stats()
            c.synchronize()
            end = c.read_stats()

        self.assertEqual(mid["pending"], 4.0)
        self.assertEqual(mid["processed_batches"], 0.0)
        self.assertEqual(end["pending"], 0.0)
        self.assertEqual(end["processed_batches"], 4.0)
        self.assertEqual(p_print.call_count, 1)

    def test_dummy_cpu_worker_injects_tiny_noise_and_prints_summary(self) -> None:
        worker = DummyCpuWorker()
        hidden = torch.ones((2, 4), dtype=torch.float32)
        worker.enqueue(hidden)

        with mock.patch("builtins.print") as p_print, mock.patch("torch.randn_like", return_value=torch.ones_like(hidden)):
            worker.drain(limit=0)

        self.assertTrue(p_print.called)
        self.assertTrue(torch.allclose(hidden, torch.ones_like(hidden)))
        stats = worker.stats()
        self.assertEqual(stats["processed_batches"], 1.0)
        self.assertEqual(stats["processed_rows"], 2.0)
        self.assertAlmostEqual(stats["last_noise_std"], 1e-3, places=6)

    def test_dummy_cpu_worker_can_drain_inference_tensor_without_inplace_error(self) -> None:
        worker = DummyCpuWorker()
        with torch.inference_mode():
            hidden = torch.ones((2, 4), dtype=torch.float32)
            cpu_hidden = hidden.detach().to("cpu")
        worker.enqueue(cpu_hidden)

        with mock.patch("builtins.print") as p_print, mock.patch("torch.randn_like", return_value=torch.ones_like(cpu_hidden)):
            worker.drain(limit=0)

        self.assertTrue(p_print.called)
        stats = worker.stats()
        self.assertEqual(stats["processed_batches"], 1.0)


if __name__ == "__main__":
    unittest.main()
