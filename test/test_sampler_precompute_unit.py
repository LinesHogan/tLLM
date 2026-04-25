#!/usr/bin/env python3
"""Unit tests for distiller precompute scheduling."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig
from tllm.runtime.vllm_patch import sampler_patch
from tllm.runtime.sampler_bridge.types import SamplerStepView


class SamplerPrecomputeUnitTest(unittest.TestCase):
    def tearDown(self) -> None:
        sampler_patch._ORIG_VLLM_SAMPLER_SAMPLE = None

    def _view(self, *, runner, model=None) -> SamplerStepView:
        return SamplerStepView(
            engine_step_id=5,
            phase="decode",
            logits=torch.randn((2, 3), dtype=torch.float32),
            sampling_metadata=object(),
            decode_count=2,
            request_ids=("reqA", "reqB"),
            prompt_idxs=(7, 8),
            sample_idxs=(0, 1),
            prompt_idx_tensor=torch.tensor([7, 8], dtype=torch.long),
            sample_idx_tensor=torch.tensor([0, 1], dtype=torch.long),
            source_hidden=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            device=torch.device("cpu"),
            model=model or SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((3, 2)), bias=None)),
            runner=runner,
        )

    def test_provider_prepare_step_prefers_runtime_precomputed_cache(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(
            ESampConsumerConfig(enable_esamp_training=True, enable_distiller_intervention=True, distiller_beta=0.5),
            engine=engine,
        )
        runtime = SimpleNamespace(
            distiller_port_publish_step_id=5,
            sampler_precomputed_step_id=5,
            sampler_precomputed_row_ids=torch.tensor([1], dtype=torch.long),
            sampler_precomputed_pred_hidden=torch.tensor([[9.0, 10.0]], dtype=torch.float32),
            sampler_precomputed_dense_logits=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            sampler_precompute_event=None,
        )
        runner = SimpleNamespace(_tllm_runtime=runtime)
        model = SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((3, 2)), bias=None))

        state = consumer.sampler_modifier_provider().prepare_step(self._view(runner=runner, model=model))

        assert state is not None
        engine.predict_hidden_for_sampling.assert_not_called()
        self.assertTrue(torch.equal(state.affected_row_ids, torch.tensor([1], dtype=torch.long)))
        self.assertTrue(torch.equal(state.pred_hidden, torch.tensor([[9.0, 10.0]], dtype=torch.float32)))
        self.assertTrue(torch.equal(state.precomputed_dense_logits, torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)))

    def test_provider_prepare_step_prefers_compact_cache_over_full_buffer_mask_scan(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(
            ESampConsumerConfig(enable_esamp_training=True, enable_distiller_intervention=True, distiller_beta=0.5),
            engine=engine,
        )
        runtime = SimpleNamespace(
            distiller_port_publish_step_id=5,
            sampler_precomputed_step_id=5,
            sampler_precompute_event=None,
            sampler_precomputed_row_ids=torch.tensor([1], dtype=torch.long),
            sampler_precomputed_pred_hidden=torch.tensor([[9.0, 10.0]], dtype=torch.float32),
            sampler_precomputed_dense_logits=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            sampler_precomputed_pred_hidden_full=torch.full((2, 2), -999.0, dtype=torch.float32),
            sampler_precomputed_valid_mask=torch.tensor([False, False]),
            sampler_precomputed_dense_logits_full=torch.full((2, 3), -999.0, dtype=torch.float32),
            sampler_precomputed_all_rows=True,
            sampler_source_capture_step_id=-1,
        )
        runner = SimpleNamespace(_tllm_runtime=runtime)
        model = SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((3, 2)), bias=None))

        with mock.patch.object(torch.Tensor, "nonzero", side_effect=AssertionError("compact cache path should not scan full valid mask")):
            state = consumer.sampler_modifier_provider().prepare_step(self._view(runner=runner, model=model))

        assert state is not None
        self.assertTrue(torch.equal(state.affected_row_ids, torch.tensor([1], dtype=torch.long)))
        self.assertTrue(torch.equal(state.pred_hidden, torch.tensor([[9.0, 10.0]], dtype=torch.float32)))
        self.assertTrue(torch.equal(state.precomputed_dense_logits, torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)))

    def test_maybe_schedule_sampler_precompute_populates_runtime_cache(self) -> None:
        engine = mock.Mock()
        engine.predict_hidden_for_sampling.return_value = (
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[5.0, 6.0]], dtype=torch.float32),
        )
        consumer = ESampConsumer(
            ESampConsumerConfig(
                enable_esamp_training=True,
                enable_distiller_intervention=True,
                distiller_beta=0.5,
            ),
            engine=engine,
        )
        runtime = SimpleNamespace(
            consumer=consumer,
            event_step_id=11,
            decode_count=2,
            decode_prompt_idxs=[7, 8],
            source_resolved_path="layers.0",
            tap_decode_hidden={"layers.0": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)},
            sampler_precompute_stream=None,
            sampler_precompute_event=None,
            sampler_precomputed_step_id=-1,
            sampler_precomputed_row_ids=None,
            sampler_precomputed_pred_hidden=None,
            sampler_precomputed_dense_logits=None,
        )
        runner = SimpleNamespace(
            _tllm_runtime=runtime,
            model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((3, 2)), bias=None)),
        )

        sampler_patch.maybe_schedule_sampler_precompute(runtime=runtime, runner=runner, layer_path="layers.0")

        engine.predict_hidden_for_sampling.assert_called_once()
        self.assertEqual(runtime.sampler_precomputed_step_id, 11)
        self.assertTrue(torch.equal(runtime.sampler_precomputed_row_ids, torch.tensor([0], dtype=torch.long)))
        self.assertTrue(torch.equal(runtime.sampler_precomputed_pred_hidden, torch.tensor([[5.0, 6.0]], dtype=torch.float32)))
        self.assertIsNone(runtime.sampler_precomputed_dense_logits)

    def test_maybe_schedule_sampler_precompute_ignores_non_source_layer(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(
            ESampConsumerConfig(
                enable_esamp_training=True,
                enable_distiller_intervention=True,
                distiller_beta=0.5,
            ),
            engine=engine,
        )
        runtime = SimpleNamespace(
            consumer=consumer,
            event_step_id=11,
            decode_count=2,
            decode_prompt_idxs=[7, 8],
            source_resolved_path="layers.0",
            tap_decode_hidden={"layers.0": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)},
            sampler_precompute_stream=None,
            sampler_precompute_event=None,
            sampler_precomputed_step_id=-1,
            sampler_precomputed_row_ids=None,
            sampler_precomputed_pred_hidden=None,
            sampler_precomputed_dense_logits=None,
        )
        runner = SimpleNamespace(_tllm_runtime=runtime)

        sampler_patch.maybe_schedule_sampler_precompute(runtime=runtime, runner=runner, layer_path="layers.1")

        engine.predict_hidden_for_sampling.assert_not_called()
        self.assertEqual(runtime.sampler_precomputed_step_id, -1)

    def test_dense_precompute_populates_runtime_dense_logits_cache(self) -> None:
        engine = mock.Mock()
        engine.predict_hidden_for_sampling.return_value = (
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[5.0, 6.0]], dtype=torch.float32),
        )
        consumer = ESampConsumer(
            ESampConsumerConfig(
                enable_esamp_training=True,
                enable_distiller_intervention=True,
                distiller_sampler_backend="pre_filter_dense",
                distiller_beta=0.5,
            ),
            engine=engine,
        )
        model = SimpleNamespace(
            lm_head=SimpleNamespace(
                weight=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
                bias=torch.tensor([0.5, -0.5], dtype=torch.float32),
            )
        )
        runtime = SimpleNamespace(
            consumer=consumer,
            event_step_id=12,
            decode_count=2,
            decode_prompt_idxs=[7, 8],
            source_resolved_path="layers.0",
            tap_decode_hidden={"layers.0": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)},
            sampler_precompute_stream=None,
            sampler_precompute_event=None,
            sampler_precomputed_step_id=-1,
            sampler_precomputed_row_ids=None,
            sampler_precomputed_pred_hidden=None,
            sampler_precomputed_dense_logits=None,
        )
        runner = SimpleNamespace(_tllm_runtime=runtime, model=model)

        sampler_patch.maybe_schedule_sampler_precompute(runtime=runtime, runner=runner, layer_path="layers.0")

        self.assertTrue(torch.equal(runtime.sampler_precomputed_dense_logits, torch.tensor([[5.5, 5.5]], dtype=torch.float32)))

    def test_prepare_decode_step_enables_source_time_precompute_for_shared_mode(self) -> None:
        engine = mock.Mock()
        engine.state = SimpleNamespace(per_request_models=False)
        consumer = ESampConsumer(
            ESampConsumerConfig(
                enable_esamp_training=True,
                enable_distiller_intervention=True,
                distiller_beta=0.5,
            ),
            engine=engine,
        )
        runtime = SimpleNamespace(
            consumer=consumer,
            event_step_id=12,
            decode_count=2,
            decode_prompt_idx_tensor=torch.tensor([7, 8], dtype=torch.long),
            source_resolved_path="layers.0",
            tap_decode_hidden={"layers.0": torch.empty((4, 2), dtype=torch.float32)},
            sampler_precomputed_pred_hidden_full=None,
            sampler_precomputed_valid_mask=None,
        )
        runner = SimpleNamespace(_tllm_runtime=runtime, model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((3, 2)), bias=None)))

        sampler_patch.maybe_prepare_sampler_decode_step(runtime=runtime, runner=runner)

        self.assertTrue(runtime.sampler_source_precompute_enabled)
        self.assertEqual(runtime.sampler_precomputed_step_id, 12)
        self.assertEqual(tuple(runtime.sampler_precomputed_pred_hidden_full.shape), (4, 2))
        self.assertEqual(tuple(runtime.sampler_precomputed_valid_mask.shape), (4,))
        self.assertTrue(torch.equal(runtime.sampler_precomputed_all_row_ids, torch.arange(4, dtype=torch.long)))

    def test_prepare_decode_step_uses_host_prompt_list_for_model_bank(self) -> None:
        engine = mock.Mock()
        engine.state = SimpleNamespace(per_request_models=True)
        engine.using_model_bank = True
        engine.prepare_sampling_slots_for_step.return_value = True
        consumer = ESampConsumer(
            ESampConsumerConfig(
                enable_esamp_training=True,
                enable_distiller_intervention=True,
                distiller_beta=0.5,
            ),
            engine=engine,
        )
        runtime = SimpleNamespace(
            consumer=consumer,
            event_step_id=12,
            decode_count=2,
            decode_prompt_idxs=[7, 8],
            decode_prompt_idx_tensor=torch.tensor([7, 8], dtype=torch.long),
            source_resolved_path="layers.0",
            tap_decode_hidden={"layers.0": torch.empty((4, 2), dtype=torch.float32)},
            sampler_precomputed_pred_hidden_full=None,
            sampler_precomputed_valid_mask=None,
        )
        runner = SimpleNamespace(_tllm_runtime=runtime, model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((3, 2)), bias=None)))

        sampler_patch.maybe_prepare_sampler_decode_step(runtime=runtime, runner=runner)

        engine.prepare_sampling_slots_for_step.assert_called_once_with([7, 8])
        self.assertTrue(runtime.sampler_source_precompute_enabled)
        self.assertTrue(runtime.sampler_precomputed_all_rows)

    def test_prepare_decode_step_disables_distiller_port_when_provider_is_inactive(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(
            ESampConsumerConfig(
                enable_esamp_training=True,
                enable_distiller_intervention=True,
                distiller_beta=0.0,
            ),
            engine=engine,
        )
        runtime = SimpleNamespace(
            consumer=consumer,
            event_step_id=12,
            sampler_source_precompute_enabled=True,
            sampler_precomputed_dense_logits=torch.tensor([[1.0]], dtype=torch.float32),
            sampler_precomputed_dense_logits_full=torch.tensor([[1.0]], dtype=torch.float32),
            sampler_precomputed_pred_hidden=torch.tensor([[1.0]], dtype=torch.float32),
            sampler_precomputed_row_ids=torch.tensor([0], dtype=torch.long),
            sampler_precomputed_all_rows=True,
        )
        runner = SimpleNamespace(_tllm_runtime=runtime, model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((3, 2)), bias=None)))

        sampler_patch.maybe_prepare_sampler_decode_step(runtime=runtime, runner=runner)

        self.assertFalse(runtime.distiller_port_enabled)
        self.assertFalse(runtime.sampler_source_precompute_enabled)
        self.assertEqual(runtime.sampler_precomputed_step_id, 12)
        self.assertIsNone(runtime.sampler_precomputed_dense_logits)
        self.assertIsNone(runtime.sampler_precomputed_dense_logits_full)
        self.assertIsNone(runtime.sampler_precomputed_pred_hidden)
        self.assertIsNone(runtime.sampler_precomputed_row_ids)
        self.assertFalse(runtime.sampler_precomputed_all_rows)

    def test_prepare_step_full_row_cache_avoids_nonzero(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(
            ESampConsumerConfig(enable_esamp_training=True, enable_distiller_intervention=True, distiller_beta=0.5),
            engine=engine,
        )
        runtime = SimpleNamespace(
            distiller_port_publish_step_id=5,
            sampler_precomputed_step_id=5,
            sampler_source_capture_step_id=5,
            sampler_precomputed_pred_hidden_full=torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32),
            sampler_precomputed_valid_mask=torch.tensor([True, True]),
            sampler_precomputed_dense_logits_full=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
            sampler_precomputed_all_rows=True,
            sampler_precompute_event=None,
        )
        runner = SimpleNamespace(_tllm_runtime=runtime)
        model = SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((3, 2)), bias=None))

        with mock.patch.object(torch.Tensor, "nonzero", side_effect=AssertionError("nonzero should not be used")):
            state = consumer.sampler_modifier_provider().prepare_step(self._view(runner=runner, model=model))

        assert state is not None
        self.assertTrue(torch.equal(state.affected_row_ids, torch.tensor([0, 1], dtype=torch.long)))
        self.assertTrue(torch.equal(state.pred_hidden, runtime.sampler_precomputed_pred_hidden_full))
        self.assertTrue(torch.equal(state.precomputed_dense_logits, runtime.sampler_precomputed_dense_logits_full))

    def test_schedule_precompute_uses_full_buffer_cache_without_count_nonzero(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(
            ESampConsumerConfig(
                enable_esamp_training=True,
                enable_distiller_intervention=True,
                distiller_sampler_backend="pre_filter_dense",
                distiller_beta=0.5,
            ),
            engine=engine,
        )
        runtime = SimpleNamespace(
            consumer=consumer,
            event_step_id=17,
            decode_count=2,
            source_resolved_path="layers.0",
            sampler_source_precompute_enabled=True,
            sampler_source_capture_step_id=17,
            sampler_precomputed_step_id=-1,
            sampler_precompute_event=None,
            sampler_precomputed_pred_hidden_full=torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
            sampler_precomputed_valid_mask=torch.tensor([True, True]),
            sampler_precomputed_dense_logits_full=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            sampler_precomputed_row_ids=torch.tensor([1], dtype=torch.long),
            sampler_precomputed_pred_hidden=torch.tensor([[9.0, 9.0]], dtype=torch.float32),
            sampler_precomputed_dense_logits=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            sampler_precomputed_all_rows=True,
            tap_decode_hidden={"layers.0": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)},
        )
        runner = SimpleNamespace(_tllm_runtime=runtime, model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((2, 2)), bias=None)))

        with mock.patch.object(torch.Tensor, "count_nonzero", side_effect=AssertionError("count_nonzero should not be used")):
            sampler_patch.maybe_schedule_sampler_precompute(runtime=runtime, runner=runner, layer_path="layers.0")

        engine.predict_hidden_for_sampling.assert_not_called()
        self.assertEqual(runtime.sampler_precomputed_step_id, 17)
        self.assertIsNone(runtime.sampler_precomputed_row_ids)
        self.assertIsNone(runtime.sampler_precomputed_pred_hidden)
        self.assertTrue(torch.equal(runtime.sampler_precomputed_dense_logits_full, torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)))

    def test_schedule_precompute_ignores_stale_full_buffer_without_fresh_capture(self) -> None:
        engine = mock.Mock()
        engine.predict_hidden_for_sampling.return_value = (
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[5.0, 6.0]], dtype=torch.float32),
        )
        consumer = ESampConsumer(
            ESampConsumerConfig(
                enable_esamp_training=True,
                enable_distiller_intervention=True,
                distiller_beta=0.5,
            ),
            engine=engine,
        )
        runtime = SimpleNamespace(
            consumer=consumer,
            event_step_id=17,
            decode_count=2,
            source_resolved_path="layers.0",
            sampler_source_precompute_enabled=True,
            sampler_source_capture_step_id=16,
            sampler_precomputed_step_id=-1,
            sampler_precompute_event=None,
            sampler_precomputed_pred_hidden_full=torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
            sampler_precomputed_valid_mask=torch.tensor([True, True]),
            sampler_precomputed_dense_logits_full=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            sampler_precomputed_row_ids=None,
            sampler_precomputed_pred_hidden=None,
            sampler_precomputed_dense_logits=None,
            sampler_precomputed_all_rows=True,
            tap_decode_hidden={"layers.0": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)},
            decode_prompt_idx_tensor=torch.tensor([7, 8], dtype=torch.long),
            decode_prompt_idxs=[7, 8],
        )
        runner = SimpleNamespace(_tllm_runtime=runtime, model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((2, 2)), bias=None)))

        sampler_patch.maybe_schedule_sampler_precompute(runtime=runtime, runner=runner, layer_path="layers.0")

        engine.predict_hidden_for_sampling.assert_called_once()
        self.assertEqual(runtime.sampler_precomputed_step_id, 17)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_capture_source_precompute_uses_cuda_stream_and_records_event(self) -> None:
        engine = mock.Mock()

        def _capture(source_hidden, prompt_idxs, *, out_pred_hidden, out_valid_mask):
            out_pred_hidden.copy_(source_hidden + 1.0)
            out_valid_mask.fill_(True)
            return True

        engine.predict_hidden_for_sampling_capture.side_effect = _capture
        consumer = ESampConsumer(
            ESampConsumerConfig(
                enable_esamp_training=True,
                enable_distiller_intervention=True,
                distiller_sampler_backend="pre_filter_dense",
                distiller_beta=0.5,
            ),
            engine=engine,
        )
        source_hidden = torch.randn((4, 2), device="cuda", dtype=torch.float32)
        runtime = SimpleNamespace(
            consumer=consumer,
            source_resolved_path="layers.0",
            tap_decode_hidden={"layers.0": source_hidden},
            decode_prompt_idx_buf=torch.tensor([7, 8, 9, 10], device="cuda", dtype=torch.long),
            sampler_allow_source_capture=True,
            sampler_allow_source_async=True,
            sampler_precompute_stream=None,
            sampler_precompute_event=None,
            sampler_precompute_source_hidden_full=None,
            sampler_precompute_prompt_idx_full=None,
            sampler_precomputed_pred_hidden_full=torch.empty_like(source_hidden),
            sampler_precomputed_valid_mask=torch.zeros((4,), device="cuda", dtype=torch.bool),
            sampler_precomputed_dense_logits_full=None,
        )
        runner = SimpleNamespace(
            _tllm_runtime=runtime,
            model=SimpleNamespace(
                lm_head=SimpleNamespace(
                    weight=torch.randn((3, 2), device="cuda", dtype=torch.float32),
                    bias=torch.randn((3,), device="cuda", dtype=torch.float32),
                )
            ),
        )

        sampler_patch.ensure_sampler_precompute_buffers(runtime=runtime, runner=runner)
        consumer.sampler_modifier_provider().maybe_capture_source_precompute(
            runtime=runtime,
            runner=runner,
            layer_path="layers.0",
        )
        torch.cuda.current_stream(device=source_hidden.device).wait_event(runtime.sampler_precompute_event)
        torch.cuda.synchronize(device=source_hidden.device)

        self.assertIsNotNone(runtime.sampler_precompute_stream)
        self.assertIsNotNone(runtime.sampler_precompute_event)
        self.assertTrue(torch.allclose(runtime.sampler_precompute_source_hidden_full, source_hidden))
        self.assertTrue(torch.equal(runtime.sampler_precompute_prompt_idx_full, torch.tensor([7, 8, 9, 10], device="cuda", dtype=torch.long)))
        self.assertTrue(torch.allclose(runtime.sampler_precomputed_pred_hidden_full, source_hidden + 1.0))
        self.assertTrue(torch.equal(runtime.sampler_precomputed_valid_mask, torch.tensor([True, True, True, True], device="cuda")))
        self.assertIsInstance(runtime.sampler_precomputed_dense_logits_full, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
