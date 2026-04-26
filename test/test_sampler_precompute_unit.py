#!/usr/bin/env python3
"""Unit tests for distiller precompute scheduling."""

from __future__ import annotations

from dataclasses import fields
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig
from tllm.runtime.residual_runtime import (
    SamplerPrecomputeBuffers,
    SamplerPrecomputeState,
)
from tllm.runtime.vllm_patch import sampler_patch
from tllm.runtime.sampler_bridge.types import SamplerStepView


class SamplerPrecomputeUnitTest(unittest.TestCase):
    def tearDown(self) -> None:
        sampler_patch._ORIG_VLLM_SAMPLER_SAMPLE = None

    def test_sampler_precompute_state_replaces_loose_nullable_tensor_fields(self) -> None:
        field_names = {field.name for field in fields(SamplerPrecomputeState)}
        self.assertIn("buffers", field_names)
        self.assertIn("cache", field_names)
        self.assertFalse(
            {
                "row_ids",
                "pred_hidden",
                "pred_hidden_row_map",
                "dense_logits",
                "all_row_ids",
                "pred_hidden_full",
                "source_hidden_full",
                "prompt_idx_full",
                "dense_logits_full",
                "valid_mask",
            }
            & field_names
        )

    def test_captured_rows_for_step_derives_device_from_owned_buffers(self) -> None:
        params = inspect.signature(SamplerPrecomputeState.captured_rows_for_step).parameters

        self.assertNotIn("device", params)

    def test_sampler_provider_prepare_step_uses_typed_precompute_views(self) -> None:
        provider_text = Path("tllm/consumers/esamp/sampler_provider.py").read_text(encoding="utf-8")
        prepare_step_body = provider_text.split("    def prepare_step(", 1)[1]
        self.assertNotRegex(prepare_step_body, r"isinstance\([^\n]*torch\.Tensor|not isinstance\([^\n]*torch\.Tensor")
        self.assertIn("cache_for_step", prepare_step_body)
        self.assertIn("captured_rows_for_step", prepare_step_body)

    def _runtime(self, **kwargs):
        kwargs.setdefault("sampler_precompute", SamplerPrecomputeState())
        return SimpleNamespace(**kwargs)

    def _cache_state(
        self,
        *,
        step_id: int,
        row_ids: torch.Tensor,
        pred_hidden: torch.Tensor,
        dense_logits: torch.Tensor | None = None,
        all_rows: bool = False,
        source_enabled: bool = False,
        port_publish_step_id: int | None = None,
    ) -> SamplerPrecomputeState:
        state = SamplerPrecomputeState(
            port_publish_step_id=step_id if port_publish_step_id is None else int(port_publish_step_id),
            precomputed_step_id=step_id,
            event=None,
        )
        state.store_cache(
            step_id=step_id,
            row_ids=row_ids,
            pred_hidden=pred_hidden,
            dense_logits=dense_logits,
            all_rows=all_rows,
        )
        return state

    def _buffer_state(
        self,
        *,
        step_id: int,
        pred_hidden: torch.Tensor,
        valid_mask: torch.Tensor,
        dense_logits: torch.Tensor | None = None,
        all_rows: bool = False,
        source_enabled: bool = False,
        port_publish_step_id: int | None = None,
    ) -> SamplerPrecomputeState:
        rows = int(pred_hidden.shape[0])
        state = SamplerPrecomputeState(
            port_publish_step_id=step_id if port_publish_step_id is None else int(port_publish_step_id),
            precomputed_step_id=step_id,
            source_capture_step_id=step_id,
            source_enabled=source_enabled,
            all_rows=all_rows,
            event=None,
            buffers=SamplerPrecomputeBuffers(
                pred_hidden=pred_hidden,
                valid_mask=valid_mask,
                source_hidden=torch.empty_like(pred_hidden),
                prompt_idx=torch.arange(rows, device=pred_hidden.device, dtype=torch.long),
                all_row_ids=torch.arange(rows, device=pred_hidden.device, dtype=torch.long),
                dense_logits=dense_logits,
            ),
        )
        return state

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
        runtime = self._runtime(
            sampler_precompute=self._cache_state(
                step_id=5,
                row_ids=torch.tensor([1], dtype=torch.long),
                pred_hidden=torch.tensor([[9.0, 10.0]], dtype=torch.float32),
                dense_logits=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            ),
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
        runtime = self._runtime(
            sampler_precompute=self._cache_state(
                step_id=5,
                row_ids=torch.tensor([1], dtype=torch.long),
                pred_hidden=torch.tensor([[9.0, 10.0]], dtype=torch.float32),
                dense_logits=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
                all_rows=True,
            ),
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
        runtime = self._runtime(
            consumer=consumer,
            event_step_id=11,
            decode_count=2,
            decode_prompt_idxs=[7, 8],
            source_resolved_path="layers.0",
            tap_decode_hidden={"layers.0": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)},
            sampler_precompute=SamplerPrecomputeState(stream=None, event=None, precomputed_step_id=-1),
        )
        runner = SimpleNamespace(
            _tllm_runtime=runtime,
            model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((3, 2)), bias=None)),
        )

        sampler_patch.maybe_schedule_sampler_precompute(runtime=runtime, runner=runner, layer_path="layers.0")

        engine.predict_hidden_for_sampling.assert_called_once()
        self.assertEqual(runtime.sampler_precompute.precomputed_step_id, 11)
        cache = runtime.sampler_precompute.cache_for_step(11)
        assert cache is not None
        self.assertTrue(torch.equal(cache.row_ids, torch.tensor([0], dtype=torch.long)))
        self.assertTrue(torch.equal(cache.pred_hidden, torch.tensor([[5.0, 6.0]], dtype=torch.float32)))
        self.assertIsNone(cache.dense_logits)

    def test_schedule_precompute_normalizes_host_prompt_list_before_engine_call(self) -> None:
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
        source_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        runtime = self._runtime(
            consumer=consumer,
            event_step_id=11,
            decode_count=2,
            decode_prompt_idxs=[7, 8],
            source_resolved_path="layers.0",
            tap_decode_hidden={"layers.0": source_hidden},
            sampler_precompute=SamplerPrecomputeState(stream=None, event=None, precomputed_step_id=-1),
        )
        runner = SimpleNamespace(
            _tllm_runtime=runtime,
            model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((3, 2)), bias=None)),
        )

        sampler_patch.maybe_schedule_sampler_precompute(runtime=runtime, runner=runner, layer_path="layers.0")

        prompt_arg = engine.predict_hidden_for_sampling.call_args.args[1]
        self.assertIsInstance(prompt_arg, torch.Tensor)
        self.assertTrue(torch.equal(prompt_arg, torch.tensor([7, 8], dtype=torch.long)))

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
        runtime = self._runtime(
            consumer=consumer,
            event_step_id=11,
            decode_count=2,
            decode_prompt_idxs=[7, 8],
            source_resolved_path="layers.0",
            tap_decode_hidden={"layers.0": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)},
            sampler_precompute=SamplerPrecomputeState(stream=None, event=None, precomputed_step_id=-1),
        )
        runner = SimpleNamespace(_tllm_runtime=runtime)

        sampler_patch.maybe_schedule_sampler_precompute(runtime=runtime, runner=runner, layer_path="layers.1")

        engine.predict_hidden_for_sampling.assert_not_called()
        self.assertEqual(runtime.sampler_precompute.precomputed_step_id, -1)

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
        runtime = self._runtime(
            consumer=consumer,
            event_step_id=12,
            decode_count=2,
            decode_prompt_idxs=[7, 8],
            source_resolved_path="layers.0",
            tap_decode_hidden={"layers.0": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)},
            sampler_precompute=SamplerPrecomputeState(stream=None, event=None, precomputed_step_id=-1),
        )
        runner = SimpleNamespace(_tllm_runtime=runtime, model=model)

        sampler_patch.maybe_schedule_sampler_precompute(runtime=runtime, runner=runner, layer_path="layers.0")

        cache = runtime.sampler_precompute.cache_for_step(12)
        assert cache is not None
        self.assertTrue(torch.equal(cache.dense_logits, torch.tensor([[5.5, 5.5]], dtype=torch.float32)))

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
        runtime = self._runtime(
            consumer=consumer,
            event_step_id=12,
            decode_count=2,
            decode_prompt_idx_tensor=torch.tensor([7, 8], dtype=torch.long),
            source_resolved_path="layers.0",
            tap_decode_hidden={"layers.0": torch.empty((4, 2), dtype=torch.float32)},
            sampler_precompute=SamplerPrecomputeState(),
        )
        runner = SimpleNamespace(_tllm_runtime=runtime, model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((3, 2)), bias=None)))

        sampler_patch.maybe_prepare_sampler_decode_step(runtime=runtime, runner=runner)

        self.assertTrue(runtime.sampler_precompute.source_enabled)
        self.assertEqual(runtime.sampler_precompute.precomputed_step_id, 12)
        assert runtime.sampler_precompute.buffers is not None
        self.assertEqual(tuple(runtime.sampler_precompute.buffers.pred_hidden.shape), (4, 2))
        self.assertEqual(tuple(runtime.sampler_precompute.buffers.valid_mask.shape), (4,))
        self.assertTrue(torch.equal(runtime.sampler_precompute.buffers.all_row_ids, torch.arange(4, dtype=torch.long)))

    def test_prepare_decode_step_uses_host_prompt_list_for_model_bank(self) -> None:
        engine = mock.Mock()
        engine.state = SimpleNamespace(per_request_models=True)
        engine.using_model_bank = True
        engine.assign_sampling_model_bank_slots.return_value = True
        consumer = ESampConsumer(
            ESampConsumerConfig(
                enable_esamp_training=True,
                enable_distiller_intervention=True,
                distiller_beta=0.5,
            ),
            engine=engine,
        )
        runtime = self._runtime(
            consumer=consumer,
            event_step_id=12,
            decode_count=2,
            decode_prompt_idxs=[7, 8],
            decode_prompt_idx_tensor=torch.tensor([7, 8], dtype=torch.long),
            source_resolved_path="layers.0",
            tap_decode_hidden={"layers.0": torch.empty((4, 2), dtype=torch.float32)},
            sampler_precompute=SamplerPrecomputeState(),
        )
        runner = SimpleNamespace(_tllm_runtime=runtime, model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((3, 2)), bias=None)))

        sampler_patch.maybe_prepare_sampler_decode_step(runtime=runtime, runner=runner)

        prompt_arg = engine.assign_sampling_model_bank_slots.call_args.args[0]
        self.assertIsInstance(prompt_arg, torch.Tensor)
        self.assertTrue(torch.equal(prompt_arg, torch.tensor([7, 8], dtype=torch.long)))
        self.assertTrue(runtime.sampler_precompute.source_enabled)
        self.assertTrue(runtime.sampler_precompute.all_rows)

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
        runtime = self._runtime(
            consumer=consumer,
            event_step_id=12,
            sampler_precompute=SamplerPrecomputeState(
                source_enabled=True,
                cache=None,
                all_rows=True,
            ),
        )
        runner = SimpleNamespace(_tllm_runtime=runtime, model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((3, 2)), bias=None)))

        sampler_patch.maybe_prepare_sampler_decode_step(runtime=runtime, runner=runner)

        self.assertFalse(runtime.sampler_precompute.port_enabled)
        self.assertFalse(runtime.sampler_precompute.source_enabled)
        self.assertEqual(runtime.sampler_precompute.precomputed_step_id, 12)
        self.assertIsNone(runtime.sampler_precompute.cache)
        self.assertFalse(runtime.sampler_precompute.all_rows)

    def test_prepare_step_full_row_cache_avoids_nonzero(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(
            ESampConsumerConfig(enable_esamp_training=True, enable_distiller_intervention=True, distiller_beta=0.5),
            engine=engine,
        )
        full_pred_hidden = torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
        full_dense_logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        runtime = self._runtime(
            sampler_precompute=self._buffer_state(
                step_id=5,
                pred_hidden=full_pred_hidden,
                valid_mask=torch.tensor([True, True]),
                dense_logits=full_dense_logits,
                all_rows=True,
            ),
        )
        runner = SimpleNamespace(_tllm_runtime=runtime)
        model = SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((3, 2)), bias=None))

        with mock.patch.object(torch.Tensor, "nonzero", side_effect=AssertionError("nonzero should not be used")):
            state = consumer.sampler_modifier_provider().prepare_step(self._view(runner=runner, model=model))

        assert state is not None
        self.assertTrue(torch.equal(state.affected_row_ids, torch.tensor([0, 1], dtype=torch.long)))
        self.assertTrue(torch.equal(state.pred_hidden, full_pred_hidden))
        self.assertTrue(torch.equal(state.precomputed_dense_logits, full_dense_logits))

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
        runtime = self._runtime(
            consumer=consumer,
            event_step_id=17,
            decode_count=2,
            source_resolved_path="layers.0",
            sampler_precompute=self._buffer_state(
                step_id=17,
                source_enabled=True,
                pred_hidden=torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
                valid_mask=torch.tensor([True, True]),
                dense_logits=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
                all_rows=True,
            ),
            tap_decode_hidden={"layers.0": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)},
        )
        runner = SimpleNamespace(_tllm_runtime=runtime, model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((2, 2)), bias=None)))

        with mock.patch.object(torch.Tensor, "count_nonzero", side_effect=AssertionError("count_nonzero should not be used")):
            sampler_patch.maybe_schedule_sampler_precompute(runtime=runtime, runner=runner, layer_path="layers.0")

        engine.predict_hidden_for_sampling.assert_not_called()
        self.assertEqual(runtime.sampler_precompute.precomputed_step_id, 17)
        cache = runtime.sampler_precompute.cache_for_step(17)
        assert cache is not None
        self.assertTrue(torch.equal(cache.row_ids, torch.tensor([0, 1], dtype=torch.long)))
        self.assertTrue(torch.equal(cache.dense_logits, torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)))

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
        runtime = self._runtime(
            consumer=consumer,
            event_step_id=17,
            decode_count=2,
            source_resolved_path="layers.0",
            sampler_precompute=self._buffer_state(
                step_id=16,
                source_enabled=True,
                pred_hidden=torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
                valid_mask=torch.tensor([True, True]),
                dense_logits=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
                all_rows=True,
            ),
            tap_decode_hidden={"layers.0": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)},
            decode_prompt_idx_tensor=torch.tensor([7, 8], dtype=torch.long),
            decode_prompt_idxs=[7, 8],
        )
        runner = SimpleNamespace(_tllm_runtime=runtime, model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((2, 2)), bias=None)))

        sampler_patch.maybe_schedule_sampler_precompute(runtime=runtime, runner=runner, layer_path="layers.0")

        engine.predict_hidden_for_sampling.assert_called_once()
        self.assertEqual(runtime.sampler_precompute.precomputed_step_id, 17)

    def test_capture_source_precompute_owns_buffer_lifecycle(self) -> None:
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
                distiller_beta=0.5,
            ),
            engine=engine,
        )
        source_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        runtime = self._runtime(
            consumer=consumer,
            event_step_id=21,
            source_resolved_path="layers.0",
            tap_decode_hidden={"layers.0": source_hidden},
            decode_prompt_idx_buf=torch.tensor([7, 8], dtype=torch.long),
            sampler_allow_source_capture=True,
            sampler_allow_source_async=False,
            sampler_precompute=SamplerPrecomputeState(stream=None, event=None),
        )
        runner = SimpleNamespace(
            _tllm_runtime=runtime,
            model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((3, 2)), bias=None)),
        )

        consumer.sampler_modifier_provider().maybe_capture_source_precompute(
            runtime=runtime,
            runner=runner,
            layer_path="layers.0",
        )

        buffers = runtime.sampler_precompute.buffers
        assert buffers is not None
        self.assertTrue(torch.equal(buffers.source_hidden, source_hidden))
        self.assertTrue(torch.equal(buffers.prompt_idx, torch.tensor([7, 8], dtype=torch.long)))
        self.assertTrue(torch.equal(buffers.pred_hidden, source_hidden + 1.0))
        self.assertTrue(torch.equal(buffers.valid_mask, torch.tensor([True, True])))
        self.assertEqual(runtime.sampler_precompute.source_capture_step_id, 21)

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
        runtime = self._runtime(
            consumer=consumer,
            source_resolved_path="layers.0",
            tap_decode_hidden={"layers.0": source_hidden},
            decode_prompt_idx_buf=torch.tensor([7, 8, 9, 10], device="cuda", dtype=torch.long),
            sampler_allow_source_capture=True,
            sampler_allow_source_async=True,
            sampler_precompute=SamplerPrecomputeState(stream=None, event=None),
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
        torch.cuda.current_stream(device=source_hidden.device).wait_event(runtime.sampler_precompute.event)
        torch.cuda.synchronize(device=source_hidden.device)

        self.assertIsNotNone(runtime.sampler_precompute.stream)
        self.assertIsNotNone(runtime.sampler_precompute.event)
        assert runtime.sampler_precompute.buffers is not None
        buffers = runtime.sampler_precompute.buffers
        self.assertTrue(torch.allclose(buffers.source_hidden, source_hidden))
        self.assertTrue(torch.equal(buffers.prompt_idx, torch.tensor([7, 8, 9, 10], device="cuda", dtype=torch.long)))
        self.assertTrue(torch.allclose(buffers.pred_hidden, source_hidden + 1.0))
        self.assertTrue(torch.equal(buffers.valid_mask, torch.tensor([True, True, True, True], device="cuda")))
        self.assertIsInstance(buffers.dense_logits, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
