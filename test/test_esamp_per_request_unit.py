#!/usr/bin/env python3
"""Unit tests for per-request ESamp helpers."""

from __future__ import annotations

from collections import deque
import unittest
from unittest import mock
import contextlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch

from tllm.consumers.esamp import engine as engine_module
from tllm.consumers.esamp.config import AdaptationStreamMode, ESampConsumerConfig
from tllm.consumers.esamp.engine import ESampTrainEngine, copy_active_rows_into_buffer, group_row_indices_by_prompt
from tllm.consumers.esamp.initializers.svd import SVDModelBankInitializerConfig, build_model_bank_initializer
from tllm.consumers.base import BaseConsumer
from tllm.contracts.hidden_batch import HiddenBatch
from tllm.contracts.runtime_context import RuntimeContext
from tllm.contracts.subscription import ConsumerSubscription
from tllm.runtime import residual_runtime as esamp_runtime
from tllm.runtime.vllm_patch import port_runtime_hooks


class ESampPerRequestUnitTest(unittest.TestCase):
    class _ReplayFailGraph:
        def replay(self) -> None:
            raise RuntimeError("inference tensor update outside InferenceMode")

    class _ReplayNoopGraph:
        def replay(self) -> None:
            return None

    def test_runtime_dispatches_events_in_stable_order(self) -> None:
        observed: list[str] = []

        class _TickConsumer(BaseConsumer):
            @property
            def consumer_id(self) -> str:
                return "tick_consumer"

            def subscriptions(self):
                return [
                    ConsumerSubscription(consumer_id="tick_consumer", event_name="prepare_inputs.post"),
                    ConsumerSubscription(consumer_id="tick_consumer", event_name="execute_model.post"),
                ]

            def consume(self, batch: HiddenBatch, ctx: RuntimeContext) -> None:
                _ = (batch, ctx)

            def on_tick(self, event_name: str, ctx: RuntimeContext) -> None:
                _ = ctx
                observed.append(event_name)

            def on_step_end(self, ctx: RuntimeContext) -> None:
                _ = ctx

        esamp_runtime.clear_dispatch_consumers()
        esamp_runtime.register_dispatch_consumer(_TickConsumer())

        class _Runner:
            def __init__(self) -> None:
                self.device = torch.device("cpu")
                self.model = object()

        runner = _Runner()

        out_tuple = (None, None, None, None, None, None)
        with mock.patch.object(port_runtime_hooks, "setup_runtime_hooks_if_active") as p_ensure, mock.patch.object(
            port_runtime_hooks, "prepare_decode_localization"
        ) as p_decode:
            p_ensure.return_value = None
            p_decode.return_value = None
            with mock.patch.object(esamp_runtime, "_ORIG_PREPARE_INPUTS", return_value=out_tuple), mock.patch.object(
                esamp_runtime, "_ORIG_EXECUTE_MODEL", return_value={"ok": True}
            ):
                port_runtime_hooks.wrapped_prepare_inputs(
                    core=esamp_runtime,
                    runner=runner,
                    scheduler_output=object(),
                )
                port_runtime_hooks.wrapped_execute_model(
                    core=esamp_runtime,
                    runner=runner,
                    args=tuple(),
                    kwargs={},
                )

        self.assertEqual(observed, ["prepare_inputs.post", "execute_model.post"])
        esamp_runtime.clear_dispatch_consumers()

    def test_synchronize_esamp_also_flushes_registered_dispatch_consumers(self) -> None:
        observed: list[str] = []

        class _SyncConsumer(BaseConsumer):
            @property
            def consumer_id(self) -> str:
                return "sync_consumer"

            def subscriptions(self):
                return []

            def consume(self, batch: HiddenBatch, ctx: RuntimeContext) -> None:
                _ = (batch, ctx)

            def on_tick(self, event_name: str, ctx: RuntimeContext) -> None:
                _ = (event_name, ctx)

            def on_step_end(self, ctx: RuntimeContext) -> None:
                _ = ctx

            def synchronize(self) -> None:
                observed.append("dispatch_sync")

        class _RuntimeConsumer:
            def synchronize(self) -> None:
                observed.append("runtime_sync")

        esamp_runtime.clear_dispatch_consumers()
        esamp_runtime.register_dispatch_consumer(_SyncConsumer())
        old_runtime_consumer = esamp_runtime.RUNTIME.consumer
        esamp_runtime.RUNTIME.consumer = _RuntimeConsumer()
        try:
            esamp_runtime.synchronize_esamp()
        finally:
            esamp_runtime.RUNTIME.consumer = old_runtime_consumer
            esamp_runtime.clear_dispatch_consumers()

        self.assertEqual(observed, ["runtime_sync", "dispatch_sync"])

    def test_execute_model_dispatches_deferred_layer_batches_when_hooks_are_disabled(self) -> None:
        observed: list[tuple[str, str, int, list[int]]] = []

        class _CaptureConsumer(BaseConsumer):
            @property
            def consumer_id(self) -> str:
                return "capture_consumer"

            def subscriptions(self):
                return [
                    ConsumerSubscription(
                        consumer_id="capture_consumer",
                        event_name="layer.post",
                        phase_filter="decode",
                    ),
                    ConsumerSubscription(consumer_id="capture_consumer", event_name="execute_model.post"),
                ]

            def consume(self, batch: HiddenBatch, ctx: RuntimeContext) -> None:
                observed.append((ctx.event_name, batch.layer_path, int(batch.rows_hidden.shape[0]), batch.metadata["prompt_idxs"]))

            def on_tick(self, event_name: str, ctx: RuntimeContext) -> None:
                _ = ctx
                if event_name == "execute_model.post":
                    observed.append((event_name, "", 0, []))

            def on_step_end(self, ctx: RuntimeContext) -> None:
                _ = ctx

        esamp_runtime.clear_dispatch_consumers()
        esamp_runtime.register_dispatch_consumer(_CaptureConsumer())
        esamp_runtime.RUNTIME.config.consumer_implementation = "base_consumer"
        esamp_runtime.RUNTIME.launch_consumer_from_hooks = False
        esamp_runtime.RUNTIME.event_step_id = 9
        esamp_runtime.RUNTIME.decode_count = 2
        esamp_runtime.RUNTIME.decode_row_idx = torch.tensor([4, 7, 0], dtype=torch.long)
        esamp_runtime.RUNTIME.decode_valid_mask = torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float32)
        esamp_runtime.RUNTIME.decode_prompt_idxs = [3, 5]
        esamp_runtime.RUNTIME.decode_sample_idxs = [0, 1]
        esamp_runtime.RUNTIME.decode_request_ids = ["reqA", "reqB"]
        esamp_runtime.RUNTIME.source_resolved_path = "layers.0"
        esamp_runtime.RUNTIME.target_resolved_path = "layers.1"
        esamp_runtime.RUNTIME.tap_decode_hidden = {
            "layers.0": torch.tensor([[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]], dtype=torch.float32),
            "layers.1": torch.tensor([[5.0, 6.0], [7.0, 8.0], [88.0, 88.0]], dtype=torch.float32),
        }

        class _Runner:
            def __init__(self) -> None:
                self.device = torch.device("cpu")
                self.model = object()

        runner = _Runner()
        launched_step_id = -1
        try:
            with mock.patch.object(port_runtime_hooks, "setup_runtime_hooks_if_active") as p_ensure, mock.patch.object(
                esamp_runtime, "_ORIG_EXECUTE_MODEL", return_value={"ok": True}
            ):
                p_ensure.return_value = None
                port_runtime_hooks.wrapped_execute_model(
                    core=esamp_runtime,
                    runner=runner,
                    args=tuple(),
                    kwargs={},
                )
        finally:
            esamp_runtime.clear_dispatch_consumers()
            esamp_runtime.RUNTIME.tap_decode_hidden = {}
            esamp_runtime.RUNTIME.decode_row_idx = None
            esamp_runtime.RUNTIME.decode_valid_mask = None
            esamp_runtime.RUNTIME.decode_prompt_idxs = []
            esamp_runtime.RUNTIME.decode_sample_idxs = []
            esamp_runtime.RUNTIME.decode_request_ids = []
            esamp_runtime.RUNTIME.decode_count = 0
            esamp_runtime.RUNTIME.launch_consumer_from_hooks = True
            esamp_runtime.RUNTIME.source_resolved_path = ""
            esamp_runtime.RUNTIME.target_resolved_path = ""

        self.assertEqual(
            observed,
            [
                ("layer.post", "layers.0", 2, [3, 5]),
                ("layer.post", "layers.1", 2, [3, 5]),
                ("execute_model.post", "", 0, []),
            ],
        )

    def test_group_rows_by_prompt(self) -> None:
        prompt_idxs = [2, 2, -1, 5, 2, 5]
        grouped = group_row_indices_by_prompt(prompt_idxs, active_rows=6)
        self.assertEqual(grouped, {2: [0, 1, 4], 5: [3, 5]})

    def test_wrapped_execute_model_launches_same_step_decode_work_once_as_fallback(self) -> None:
        esamp_runtime.RUNTIME.launch_consumer_from_hooks = False
        esamp_runtime.RUNTIME.dispatch_plan = SimpleNamespace(
            has_active_targets=lambda: True,
            select=lambda **kwargs: [],
        )
        esamp_runtime.RUNTIME.event_step_id = 9
        esamp_runtime.RUNTIME.sampler_precompute.port_enabled = True
        esamp_runtime.RUNTIME.decode_post_logits_launched_step_id = -1
        esamp_runtime.RUNTIME.decode_count = 2
        esamp_runtime.RUNTIME.decode_row_idx = torch.tensor([4, 7, 0], dtype=torch.long)
        esamp_runtime.RUNTIME.decode_valid_mask = torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float32)
        esamp_runtime.RUNTIME.decode_prompt_idxs = [3, 5]
        esamp_runtime.RUNTIME.decode_sample_idxs = [0, 1]
        esamp_runtime.RUNTIME.decode_request_ids = ["reqA", "reqB"]
        esamp_runtime.RUNTIME.source_resolved_path = "layers.0"
        esamp_runtime.RUNTIME.target_resolved_path = "layers.1"
        esamp_runtime.RUNTIME.tap_decode_hidden = {
            "layers.0": torch.tensor([[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]], dtype=torch.float32),
            "layers.1": torch.tensor([[5.0, 6.0], [7.0, 8.0], [88.0, 88.0]], dtype=torch.float32),
        }

        class _Runner:
            def __init__(self) -> None:
                self.device = torch.device("cpu")
                self.model = object()

        runner = _Runner()
        try:
            with mock.patch.object(port_runtime_hooks, "setup_runtime_hooks_if_active") as p_ensure, mock.patch.object(
                esamp_runtime, "_ORIG_EXECUTE_MODEL", return_value={"ok": True}
            ), mock.patch.object(
                port_runtime_hooks,
                "dispatch_decode_port_bundles",
                return_value=1,
            ) as p_dispatch, mock.patch.object(
                port_runtime_hooks._sampler_patch,
                "maybe_schedule_sampler_precompute",
            ) as p_schedule:
                p_ensure.return_value = None
                port_runtime_hooks.wrapped_execute_model(
                    core=esamp_runtime,
                    runner=runner,
                    args=tuple(),
                    kwargs={},
                )
                launched_step_id = int(esamp_runtime.RUNTIME.decode_post_logits_launched_step_id)
        finally:
            esamp_runtime.RUNTIME.dispatch_plan = None
            esamp_runtime.RUNTIME.sampler_precompute.port_enabled = True
            esamp_runtime.RUNTIME.decode_post_logits_launched_step_id = -1
            esamp_runtime.RUNTIME.tap_decode_hidden = {}
            esamp_runtime.RUNTIME.decode_row_idx = None
            esamp_runtime.RUNTIME.decode_valid_mask = None
            esamp_runtime.RUNTIME.decode_prompt_idxs = []
            esamp_runtime.RUNTIME.decode_sample_idxs = []
            esamp_runtime.RUNTIME.decode_request_ids = []
            esamp_runtime.RUNTIME.decode_count = 0
            esamp_runtime.RUNTIME.launch_consumer_from_hooks = True
            esamp_runtime.RUNTIME.source_resolved_path = ""
            esamp_runtime.RUNTIME.target_resolved_path = ""

        p_dispatch.assert_called_once_with(core=esamp_runtime, runner=runner)
        p_schedule.assert_called_once_with(
            runtime=esamp_runtime.RUNTIME,
            runner=runner,
            layer_path="layers.0",
        )
        self.assertEqual(launched_step_id, 9)

    def test_wrapped_execute_model_skips_fallback_when_compute_logits_hook_is_present(self) -> None:
        esamp_runtime.RUNTIME.launch_consumer_from_hooks = False
        esamp_runtime.RUNTIME.dispatch_plan = SimpleNamespace(
            has_active_targets=lambda: True,
            select=lambda **kwargs: [],
        )
        esamp_runtime.RUNTIME.event_step_id = 9
        esamp_runtime.RUNTIME.decode_post_logits_launched_step_id = -1
        esamp_runtime.RUNTIME.decode_count = 2
        esamp_runtime.RUNTIME.source_resolved_path = "layers.0"

        class _Runner:
            def __init__(self) -> None:
                self.device = torch.device("cpu")
                self.model = SimpleNamespace(_tllm_compute_logits_wrapped=True)

        runner = _Runner()
        try:
            with mock.patch.object(port_runtime_hooks, "setup_runtime_hooks_if_active") as p_ensure, mock.patch.object(
                esamp_runtime, "_ORIG_EXECUTE_MODEL", return_value={"ok": True}
            ), mock.patch.object(
                port_runtime_hooks,
                "dispatch_decode_port_bundles",
                return_value=1,
            ) as p_dispatch, mock.patch.object(
                port_runtime_hooks._sampler_patch,
                "maybe_schedule_sampler_precompute",
            ) as p_schedule:
                p_ensure.return_value = None
                port_runtime_hooks.wrapped_execute_model(
                    core=esamp_runtime,
                    runner=runner,
                    args=tuple(),
                    kwargs={},
                )
        finally:
            esamp_runtime.RUNTIME.dispatch_plan = None
            esamp_runtime.RUNTIME.decode_post_logits_launched_step_id = -1
            esamp_runtime.RUNTIME.decode_count = 0
            esamp_runtime.RUNTIME.launch_consumer_from_hooks = True
            esamp_runtime.RUNTIME.source_resolved_path = ""

        p_dispatch.assert_called_once_with(core=esamp_runtime, runner=runner)
        p_schedule.assert_not_called()
        self.assertEqual(esamp_runtime.RUNTIME.decode_post_logits_launched_step_id, -1)

    def test_wrapped_execute_model_skips_reinstall_when_model_hooks_already_exist(self) -> None:
        esamp_runtime.RUNTIME.launch_consumer_from_hooks = True
        esamp_runtime.RUNTIME.dispatch_plan = SimpleNamespace(
            has_active_targets=lambda: True,
            select=lambda **kwargs: [],
        )

        class _Runner:
            def __init__(self) -> None:
                self.device = torch.device("cpu")
                self.model = SimpleNamespace(
                    esamp_hook_installed=True,
                    _tllm_compute_logits_wrapped=True,
                )

        runner = _Runner()
        try:
            with mock.patch.object(port_runtime_hooks, "setup_runtime_hooks_if_active") as p_ensure, mock.patch.object(
                esamp_runtime, "_ORIG_EXECUTE_MODEL", return_value={"ok": True}
            ), mock.patch.object(
                port_runtime_hooks,
                "dispatch_decode_port_bundles",
                return_value=1,
            ):
                port_runtime_hooks.wrapped_execute_model(
                    core=esamp_runtime,
                    runner=runner,
                    args=tuple(),
                    kwargs={},
                )
        finally:
            esamp_runtime.RUNTIME.dispatch_plan = None
            esamp_runtime.RUNTIME.launch_consumer_from_hooks = True

        p_ensure.assert_not_called()

    def test_post_logits_decode_work_is_idempotent_within_step(self) -> None:
        core = SimpleNamespace(
            RUNTIME=SimpleNamespace(
                event_step_id=13,
                decode_count=2,
                decode_post_logits_launched_step_id=-1,
                source_resolved_path="layers.0",
                sampler_precompute=esamp_runtime.SamplerPrecomputeState(),
            ),
        )
        runner = SimpleNamespace(model=object())

        with mock.patch.object(
            port_runtime_hooks._sampler_patch,
            "maybe_schedule_sampler_precompute",
        ) as p_schedule:
            port_runtime_hooks.maybe_launch_post_logits_decode_work(core=core, runner=runner)
            port_runtime_hooks.maybe_launch_post_logits_decode_work(core=core, runner=runner)

        p_schedule.assert_called_once_with(
            runtime=core.RUNTIME,
            runner=runner,
            layer_path="layers.0",
        )
        self.assertEqual(core.RUNTIME.decode_post_logits_launched_step_id, 13)


    def test_group_rows_respects_active_rows(self) -> None:
        prompt_idxs = [0, 1, 2, 3]
        grouped = group_row_indices_by_prompt(prompt_idxs, active_rows=2)
        self.assertEqual(grouped, {0: [0], 1: [1]})

    def test_copy_active_rows_leaves_inactive_tail_untouched(self) -> None:
        dst = torch.full((4, 3), fill_value=-1.0, dtype=torch.float32)
        src = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=torch.float32,
        )

        copied = copy_active_rows_into_buffer(dst, src)

        self.assertEqual(copied, 2)
        self.assertTrue(torch.equal(dst[:2], src))
        self.assertTrue(torch.equal(dst[2:], torch.full((2, 3), fill_value=-1.0, dtype=torch.float32)))

    def test_runtime_req_id_resolver_for_n_greater_than_one(self) -> None:
        esamp_runtime.RUNTIME.reqid_to_promptidx = {"reqA": 7, "reqB": 11}
        esamp_runtime.RUNTIME.reqid_to_sampleidx = {}
        self.assertEqual(esamp_runtime._resolve_prompt_sample_for_req_id("reqA"), (7, 0))
        self.assertEqual(esamp_runtime._resolve_prompt_sample_for_req_id("3_reqA"), (7, 3))
        self.assertEqual(esamp_runtime._resolve_prompt_sample_for_req_id("1_reqB"), (11, 1))
        self.assertEqual(esamp_runtime._resolve_prompt_sample_for_req_id("unknown"), (-1, -1))

    def test_runtime_req_id_resolver_prefers_direct_sample_mapping(self) -> None:
        esamp_runtime.RUNTIME.reqid_to_promptidx = {"reqA": 7}
        esamp_runtime.RUNTIME.reqid_to_sampleidx = {"reqA": 5}
        self.assertEqual(esamp_runtime._resolve_prompt_sample_for_req_id("reqA"), (7, 5))

    def test_runner_compile_detection_uses_use_cuda_graph_flag(self) -> None:
        class _Runner:
            use_cuda_graph = True
            compilation_config = None

        self.assertTrue(esamp_runtime.runner_uses_compilation_or_cudagraph(_Runner()))

    def test_wrapped_prepare_inputs_bypasses_runtime_hook_path_when_no_active_targets(self) -> None:
        class _Runner:
            def __init__(self) -> None:
                self.device = torch.device("cpu")
                self.model = object()

        runner = _Runner()
        old_consumer = esamp_runtime.RUNTIME.consumer
        old_plan = esamp_runtime.RUNTIME.dispatch_plan
        try:
            esamp_runtime.RUNTIME.consumer = None
            esamp_runtime.RUNTIME.dispatch_plan = None
            out_tuple = (None, None, None, None, None, None)
            with mock.patch.object(port_runtime_hooks, "setup_runtime_hooks_if_active") as p_ensure, mock.patch.object(
                port_runtime_hooks._common_hooks, "dispatch_runtime_event"
            ) as p_dispatch, mock.patch.object(esamp_runtime, "_ORIG_PREPARE_INPUTS", return_value=out_tuple):
                out = port_runtime_hooks.wrapped_prepare_inputs(
                    core=esamp_runtime,
                    runner=runner,
                    scheduler_output=object(),
                )
            self.assertIs(out, out_tuple)
            p_ensure.assert_not_called()
            p_dispatch.assert_not_called()
        finally:
            esamp_runtime.RUNTIME.consumer = old_consumer
            esamp_runtime.RUNTIME.dispatch_plan = old_plan

    def test_wrapped_execute_model_bypasses_runtime_hook_path_when_no_active_targets(self) -> None:
        class _Runner:
            def __init__(self) -> None:
                self.device = torch.device("cpu")
                self.model = object()

        runner = _Runner()
        old_consumer = esamp_runtime.RUNTIME.consumer
        old_plan = esamp_runtime.RUNTIME.dispatch_plan
        try:
            esamp_runtime.RUNTIME.consumer = None
            esamp_runtime.RUNTIME.dispatch_plan = None
            with mock.patch.object(port_runtime_hooks, "setup_runtime_hooks_if_active") as p_ensure, mock.patch.object(
                port_runtime_hooks, "dispatch_decode_port_bundles"
            ) as p_bundles, mock.patch.object(
                port_runtime_hooks._common_hooks, "dispatch_runtime_event"
            ) as p_dispatch, mock.patch.object(esamp_runtime, "_ORIG_EXECUTE_MODEL", return_value={"ok": True}) as p_orig:
                out = port_runtime_hooks.wrapped_execute_model(
                    core=esamp_runtime,
                    runner=runner,
                    args=tuple(),
                    kwargs={},
                )
            self.assertEqual(out, {"ok": True})
            p_orig.assert_called_once()
            p_ensure.assert_not_called()
            p_bundles.assert_not_called()
            p_dispatch.assert_not_called()
        finally:
            esamp_runtime.RUNTIME.consumer = old_consumer
            esamp_runtime.RUNTIME.dispatch_plan = old_plan

    def test_configure_runtime_accepts_optional_model_bank_initializer(self) -> None:
        esamp_runtime.configure_runtime(
            graph_scratch_rows=64,
            tap_layer_paths=["model.model.layers[0]", "model.model.layers[-1]"],
            source_layer_path="model.model.layers[0]",
            target_layer_path="model.model.layers[-1]",
            enable_esamp_training=True,
            distiller_hidden_dim=256,
            distiller_lr=1e-3,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=16,
            model_bank_flush_interval=4,
            adaptation_pipeline_slots=8,
            adaptation_stream_mode="single",
            adaptation_stream_priority=-1,
            model_bank_rank=32,
            model_bank_use_output_layernorm=False,
            model_bank_initializer=SVDModelBankInitializerConfig(
                method="ridge_svd",
                ridge_lambda=5e-3,
                min_rows=24,
                max_wait_steps=3,
            ),
        )
        cfg = esamp_runtime.RUNTIME.config
        self.assertTrue(cfg.per_request_models)
        self.assertTrue(cfg.per_request_model_bank)
        self.assertEqual(cfg.model_bank_slots, 16)
        self.assertEqual(cfg.model_bank_flush_interval, 4)
        self.assertEqual(cfg.adaptation_pipeline_slots, 8)
        self.assertEqual(cfg.adaptation_stream_mode, "single")
        self.assertEqual(cfg.adaptation_stream_priority, -1)
        self.assertEqual(cfg.model_bank_rank, 32)
        self.assertFalse(cfg.model_bank_use_output_layernorm)
        assert cfg.model_bank_initializer is not None
        self.assertEqual(cfg.model_bank_initializer.method, "ridge_svd")
        self.assertAlmostEqual(cfg.model_bank_initializer.ridge_lambda, 5e-3)
        self.assertEqual(cfg.model_bank_initializer.min_rows, 24)
        self.assertEqual(cfg.model_bank_initializer.max_wait_steps, 3)

    def test_adaptation_stream_mode_accepts_only_canonical_names(self) -> None:
        for mode in ("dual", "single", "serial"):
            with self.subTest(mode=mode):
                cfg = ESampConsumerConfig(adaptation_stream_mode=mode)
                self.assertEqual(cfg.adaptation_stream_mode, mode)

        cfg = ESampConsumerConfig(adaptation_stream_mode=AdaptationStreamMode.SINGLE)
        self.assertEqual(cfg.adaptation_stream_mode, AdaptationStreamMode.SINGLE)
        self.assertEqual(str(cfg.adaptation_stream_mode), "single")

        for mode in ("async", "async-dual", "dual-stream", "current", "main", "single-stream"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "adaptation_stream_mode"):
                    ESampConsumerConfig(adaptation_stream_mode=mode)

    def test_model_bank_ridge_svd_init_reduces_start_mse(self) -> None:
        device = torch.device("cpu")
        dtype = torch.float32
        hidden = 12
        rank = 4
        rows = 32
        g = torch.Generator(device=device)
        g.manual_seed(1234)

        consumer = ESampTrainEngine(
            hidden_dim=rank,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=1,
            model_bank_rank=rank,
            model_bank_use_output_layernorm=False,
            model_bank_initializer=build_model_bank_initializer(
                SVDModelBankInitializerConfig(
                    method="ridge_svd",
                    ridge_lambda=1e-2,
                    min_rows=rows,
                    max_wait_steps=1,
                )
            ),
        )
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=device,
                rows=rows,
                hidden_size=hidden,
                hidden_dtype=dtype,
            )
        slot = consumer._assign_model_bank_slot(0)

        src = torch.randn((rows, hidden), generator=g, device=device, dtype=dtype)
        teacher_u = torch.randn((hidden, rank), generator=g, device=device, dtype=dtype) * 0.2
        teacher_v = torch.randn((rank, hidden), generator=g, device=device, dtype=dtype) * 0.2
        tgt = src + torch.matmul(torch.matmul(src, teacher_u), teacher_v)
        slot_ids = torch.full((rows,), int(slot), device=device, dtype=torch.long)

        pred_before = consumer._model_bank_forward_locked(slot_ids, src)
        mse_before = float(torch.mean((pred_before - tgt) ** 2).item())
        assert consumer.state.model_bank_initializer is not None
        consumer.state.model_bank_initializer.maybe_prepare_slots(consumer, slot_ids, src, tgt)
        pred_after = consumer._model_bank_forward_locked(slot_ids, src)
        mse_after = float(torch.mean((pred_after - tgt) ** 2).item())

        self.assertTrue(consumer.state.model_bank_initializer.state.slot_init_done.get(slot, False))
        self.assertLess(mse_after, mse_before)

    def test_model_bank_ffn_fast_svd_template_init(self) -> None:
        device = torch.device("cpu")
        dtype = torch.float32
        hidden = 10
        rank = 3
        rows = 16

        consumer = ESampTrainEngine(
            hidden_dim=rank,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=1,
            model_bank_rank=rank,
            model_bank_use_output_layernorm=False,
            model_bank_initializer=build_model_bank_initializer(
                SVDModelBankInitializerConfig(method="ffn_fast_svd")
            ),
        )
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=device,
                rows=rows,
                hidden_size=hidden,
                hidden_dtype=dtype,
            )

        g = torch.Generator(device=device)
        g.manual_seed(7)
        a = torch.randn((hidden, rank), generator=g, device=device, dtype=dtype) * 0.1
        b = torch.randn((rank, hidden), generator=g, device=device, dtype=dtype) * 0.1
        assert consumer.state.model_bank_initializer is not None
        consumer.state.model_bank_initializer.set_template(a=a, b=b, key="unit")
        slot = consumer._assign_model_bank_slot(0)
        self.assertTrue(consumer.state.model_bank_initializer.state.slot_init_done.get(slot, False))

        src = torch.randn((rows, hidden), generator=g, device=device, dtype=dtype)
        slot_ids = torch.full((rows,), int(slot), device=device, dtype=torch.long)
        pred = consumer._model_bank_forward_locked(slot_ids, src)
        delta = pred - src

        gate_const = float(torch.nn.functional.silu(torch.ones(())))
        ref = torch.matmul(torch.matmul(src, a), b) * gate_const
        mse = float(torch.mean((delta - ref) ** 2).item())
        self.assertLess(mse, 1e-6)

    def test_model_bank_ffn_fast_svd_templates_are_cpu_detached(self) -> None:
        hidden = 8
        rank = 2
        mlp = torch.nn.Module()
        mlp.up_proj = torch.nn.Linear(hidden, hidden * 2, bias=False)
        mlp.down_proj = torch.nn.Linear(hidden * 2, hidden, bias=False)

        initializer = build_model_bank_initializer(SVDModelBankInitializerConfig(method="ffn_fast_svd"))
        consumer = ESampTrainEngine(
            hidden_dim=rank,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=1,
            model_bank_rank=rank,
            model_bank_initializer=initializer,
        )

        assert initializer is not None
        initializer.prepare_from_model(
            engine=consumer,
            model=mlp,
            target_layer=mlp,
            target_resolved="mlp",
            hidden_size=hidden,
        )

        self.assertIsNotNone(initializer.state.template_a)
        self.assertIsNotNone(initializer.state.template_b)
        assert initializer.state.template_a is not None
        assert initializer.state.template_b is not None
        self.assertEqual(initializer.state.template_a.device.type, "cpu")
        self.assertEqual(initializer.state.template_b.device.type, "cpu")
        self.assertFalse(initializer.state.template_a.requires_grad)
        self.assertFalse(initializer.state.template_b.requires_grad)

    def test_model_bank_ffn_fast_svd_failed_template_does_not_mark_slot_initialized(self) -> None:
        initializer = build_model_bank_initializer(SVDModelBankInitializerConfig(method="ffn_fast_svd"))
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=1,
            model_bank_rank=2,
            model_bank_initializer=initializer,
        )

        assert initializer is not None
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=2,
                hidden_size=4,
                hidden_dtype=torch.float32,
            )
        slot = consumer._assign_model_bank_slot(0)

        self.assertFalse(initializer.state.slot_init_done.get(slot, True))

    def test_engine_configure_preserves_initializer_instance_when_config_is_unchanged(self) -> None:
        initializer_cfg = SVDModelBankInitializerConfig(method="ffn_fast_svd")
        consumer = ESampTrainEngine(
            hidden_dim=4,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_initializer=build_model_bank_initializer(initializer_cfg),
        )
        before = consumer.state.model_bank_initializer
        assert before is not None

        consumer.configure(
            type(
                "Cfg",
                (),
                {
                    "distiller_hidden_dim": 4,
                    "distiller_lr": 1e-3,
                    "enable_esamp_training": True,
                    "per_request_models": True,
                    "per_request_model_bank": True,
                    "model_bank_slots": 0,
                    "model_bank_flush_interval": 1,
                    "model_bank_rank": 64,
                    "model_bank_use_output_layernorm": True,
                    "model_bank_initializer": initializer_cfg,
                    "model_bank_train_cudagraph": False,
                    "trace_per_request_losses": False,
                    "trace_interval": 1,
                    "trace_max_points": 0,
                },
            )()
        )

        self.assertIs(consumer.state.model_bank_initializer, before)

    def test_engine_configure_invalidates_shared_resources_when_mode_changes(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=4, lr=1e-3, enabled=True)
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=8,
                hidden_dtype=torch.float32,
            )

        self.assertIsNotNone(consumer.state.shared)
        consumer.state.graphs["shared"].capture_state = "captured"

        consumer.configure(
            SimpleNamespace(
                distiller_hidden_dim=4,
                distiller_lr=1e-3,
                enable_esamp_training=True,
                per_request_models=True,
                per_request_model_bank=False,
                model_bank_slots=0,
                model_bank_flush_interval=1,
                model_bank_rank=64,
                model_bank_use_output_layernorm=True,
                model_bank_initializer=None,
                model_bank_train_cudagraph=False,
                trace_per_request_losses=False,
                trace_interval=1,
                trace_max_points=0,
            )
        )

        self.assertIsNone(consumer.state.shared)
        self.assertEqual(consumer.state.graphs["shared"].capture_state, "uncaptured")

    def test_engine_configure_invalidates_model_bank_resources_when_bank_layout_changes(self) -> None:
        consumer = ESampTrainEngine(
            hidden_dim=4,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=2,
            model_bank_rank=4,
        )
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=8,
                hidden_dtype=torch.float32,
            )

        self.assertIsNotNone(consumer.state.model_bank)
        self.assertIsNotNone(consumer.state.model_bank_optimizer)
        consumer.state.graphs["model_bank"].capture_state = "captured"
        consumer.state.prompt_to_slot[7] = 1
        consumer.state.slot_to_prompt[1] = 7

        consumer.configure(
            SimpleNamespace(
                distiller_hidden_dim=4,
                distiller_lr=1e-3,
                enable_esamp_training=True,
                per_request_models=True,
                per_request_model_bank=True,
                model_bank_slots=2,
                model_bank_flush_interval=1,
                model_bank_rank=2,
                model_bank_use_output_layernorm=True,
                model_bank_initializer=None,
                model_bank_train_cudagraph=False,
                trace_per_request_losses=False,
                trace_interval=1,
                trace_max_points=0,
            )
        )

        self.assertIsNone(consumer.state.model_bank)
        self.assertIsNone(consumer.state.model_bank_optimizer)
        self.assertEqual(consumer.state.prompt_to_slot, {})
        self.assertEqual(consumer.state.slot_to_prompt, {})
        self.assertEqual(consumer.state.graphs["model_bank"].capture_state, "uncaptured")

    def test_engine_configure_clears_in_flight_step_state_even_for_hot_updates(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=4, lr=1e-3, enabled=True)
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=8,
                hidden_dtype=torch.float32,
            )

        consumer.state.current_step_slot = 1
        consumer.state.ready_step_slot = 2
        consumer.state.pending_train_queue.append((2, 4, [0, 1, 2, 3]))

        consumer.configure(
            SimpleNamespace(
                distiller_hidden_dim=4,
                distiller_lr=1e-3,
                enable_esamp_training=False,
                per_request_models=False,
                per_request_model_bank=False,
                model_bank_slots=0,
                model_bank_flush_interval=1,
                model_bank_rank=64,
                model_bank_use_output_layernorm=True,
                model_bank_initializer=None,
                model_bank_train_cudagraph=False,
                trace_per_request_losses=False,
                trace_interval=1,
                trace_max_points=0,
            )
        )

        self.assertIsNone(consumer.state.current_step_slot)
        self.assertIsNone(consumer.state.ready_step_slot)
        self.assertEqual(len(consumer.state.pending_train_queue), 0)

    def test_shared_graph_replay_failure_disables_graph_and_falls_back(self) -> None:
        device = torch.device("cpu")
        dtype = torch.float32
        hidden = 4
        rows = 3
        consumer = ESampTrainEngine(hidden_dim=2, lr=1e-3, enabled=True)
        shared_graph = consumer.state.graphs["shared"]
        shared_graph.capture_state = "captured"
        shared_graph.replay_graph = self._ReplayFailGraph()
        shared_graph.rows = rows
        shared_graph.buffers = {
            "src": torch.zeros((rows, hidden), device=device, dtype=dtype),
            "tgt": torch.zeros((rows, hidden), device=device, dtype=dtype),
            "valid": torch.zeros((rows,), device=device, dtype=torch.float32),
            "loss": torch.zeros((1,), device=device, dtype=torch.float32),
        }
        consumer.state.stats = engine_module._StatsBuffers(
            loss_sum=torch.zeros((1,), device=device, dtype=torch.float32),
            loss_count=torch.zeros((1,), device=device, dtype=torch.int64),
        )
        src = torch.randn((rows, hidden), device=device, dtype=dtype)
        tgt = torch.randn((rows, hidden), device=device, dtype=dtype)

        ok = consumer._train_shared_batch_graph_locked(src, tgt, active_rows=rows)
        self.assertFalse(ok)
        self.assertEqual(consumer.state.graphs["shared"].capture_state, "disabled")
        self.assertIn("InferenceMode", consumer.state.graphs["shared"].disable_reason)

    def test_model_bank_graph_replay_failure_disables_graph_and_falls_back(self) -> None:
        device = torch.device("cpu")
        dtype = torch.float32
        hidden = 4
        rows = 3
        slots = 2
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
        )
        bank_graph = consumer.state.graphs["model_bank"]
        bank_graph.capture_state = "captured"
        bank_graph.replay_graph = self._ReplayFailGraph()
        bank_graph.rows = rows
        bank_graph.buffers = {
            "slot_ids": torch.zeros((rows,), device=device, dtype=torch.long),
            "src": torch.zeros((rows, hidden), device=device, dtype=dtype),
            "tgt": torch.zeros((rows, hidden), device=device, dtype=dtype),
            "valid": torch.zeros((rows,), device=device, dtype=torch.float32),
            "slot_sum": torch.zeros((slots,), device=device, dtype=torch.float32),
            "slot_cnt": torch.zeros((slots,), device=device, dtype=torch.float32),
            "loss": torch.zeros((1,), device=device, dtype=torch.float32),
        }
        consumer.state.slot_to_prompt = {0: 0}
        consumer.state.stats = engine_module._StatsBuffers(
            loss_sum=torch.zeros((1,), device=device, dtype=torch.float32),
            loss_count=torch.zeros((1,), device=device, dtype=torch.int64),
        )
        slot_ids = torch.tensor([0, 0, 0], device=device, dtype=torch.long)
        src = torch.randn((rows, hidden), device=device, dtype=dtype)
        tgt = torch.randn((rows, hidden), device=device, dtype=dtype)

        ok = consumer._train_model_bank_batch_graph_locked(slot_ids, src, tgt)
        self.assertFalse(ok)
        self.assertEqual(consumer.state.graphs["model_bank"].capture_state, "disabled")
        self.assertIn("InferenceMode", consumer.state.graphs["model_bank"].disable_reason)

    def test_model_bank_graph_replay_counts_active_slots_not_total_assigned_slots(self) -> None:
        device = torch.device("cpu")
        dtype = torch.float32
        hidden = 4
        rows = 3
        slots = 4
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
        )
        bank_graph = consumer.state.graphs["model_bank"]
        bank_graph.capture_state = "captured"
        bank_graph.replay_graph = self._ReplayNoopGraph()
        bank_graph.rows = rows
        bank_graph.buffers = {
            "slot_ids": torch.zeros((rows,), device=device, dtype=torch.long),
            "src": torch.zeros((rows, hidden), device=device, dtype=dtype),
            "tgt": torch.zeros((rows, hidden), device=device, dtype=dtype),
            "valid": torch.zeros((rows,), device=device, dtype=torch.float32),
            "slot_sum": torch.zeros((slots,), device=device, dtype=torch.float32),
            "slot_cnt": torch.zeros((slots,), device=device, dtype=torch.float32),
            "loss": torch.ones((1,), device=device, dtype=torch.float32),
        }
        consumer.state.slot_to_prompt = {0: 0, 1: 1, 2: 2, 3: 3}
        consumer.state.stats = engine_module._StatsBuffers(
            loss_sum=torch.zeros((1,), device=device, dtype=torch.float32),
            loss_count=torch.zeros((1,), device=device, dtype=torch.int64),
        )
        slot_ids = torch.tensor([0, 0, 1], device=device, dtype=torch.long)
        src = torch.randn((rows, hidden), device=device, dtype=dtype)
        tgt = torch.randn((rows, hidden), device=device, dtype=dtype)

        ok = consumer._train_model_bank_batch_graph_locked(slot_ids, src, tgt)

        self.assertTrue(ok)
        self.assertEqual(int(consumer.state.stats.loss_count.item()), 2)

    def test_model_bank_graph_replay_uses_host_active_slot_count(self) -> None:
        device = torch.device("cpu")
        dtype = torch.float32
        hidden = 4
        rows = 3
        slots = 4
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
        )
        bank_graph = consumer.state.graphs["model_bank"]
        bank_graph.capture_state = "captured"
        bank_graph.replay_graph = self._ReplayNoopGraph()
        bank_graph.rows = rows
        bank_graph.buffers = {
            "slot_ids": torch.zeros((rows,), device=device, dtype=torch.long),
            "src": torch.zeros((rows, hidden), device=device, dtype=dtype),
            "tgt": torch.zeros((rows, hidden), device=device, dtype=dtype),
            "valid": torch.zeros((rows,), device=device, dtype=torch.float32),
            "slot_sum": torch.zeros((slots,), device=device, dtype=torch.float32),
            "slot_cnt": torch.zeros((slots,), device=device, dtype=torch.float32),
            "loss": torch.ones((1,), device=device, dtype=torch.float32),
        }
        consumer.state.slot_to_prompt = {0: 0, 1: 1, 2: 2, 3: 3}
        consumer.state.stats = engine_module._StatsBuffers(
            loss_sum=torch.zeros((1,), device=device, dtype=torch.float32),
            loss_count=torch.zeros((1,), device=device, dtype=torch.int64),
        )
        slot_ids = torch.tensor([0, 0, 1], device=device, dtype=torch.long)
        src = torch.randn((rows, hidden), device=device, dtype=dtype)
        tgt = torch.randn((rows, hidden), device=device, dtype=dtype)

        with mock.patch.object(consumer, "_graph_enabled", return_value=True), mock.patch.object(
            consumer,
            "_count_active_model_bank_slots",
            side_effect=AssertionError("unexpected gpu unique"),
        ):
            ok = consumer._maybe_run_model_bank_graph(
                src=src,
                tgt=tgt,
                slot_tensor=slot_ids,
                active_slot_count=2,
            )

        self.assertTrue(ok)
        self.assertEqual(int(consumer.state.stats.loss_count.item()), 2)

    def test_graph_input_staging_only_clears_previous_active_prefix(self) -> None:
        device = torch.device("cpu")
        dtype = torch.float32
        hidden = 2
        rows = 5
        consumer = ESampTrainEngine(hidden_dim=2, lr=1e-3, enabled=True)
        graph = consumer.state.graphs["shared"]
        graph.rows = rows
        graph.active_rows = 2
        graph.buffers = {
            "src": torch.zeros((rows, hidden), device=device, dtype=dtype),
            "tgt": torch.zeros((rows, hidden), device=device, dtype=dtype),
            "valid": torch.tensor([1.0, 1.0, 9.0, 9.0, 9.0], device=device, dtype=torch.float32),
            "loss": torch.zeros((1,), device=device, dtype=torch.float32),
        }
        src = torch.ones((1, hidden), device=device, dtype=dtype)
        tgt = torch.full((1, hidden), 2.0, device=device, dtype=dtype)

        ok = consumer._stage_graph_inputs("shared", src=src, tgt=tgt, active_rows=1)

        self.assertTrue(ok)
        self.assertEqual(graph.active_rows, 1)
        self.assertTrue(torch.equal(graph.buffers["valid"], torch.tensor([1.0, 0.0, 9.0, 9.0, 9.0])))

    def test_external_graph_input_staging_keeps_source_target_aliases(self) -> None:
        device = torch.device("cpu")
        dtype = torch.float32
        hidden = 2
        rows = 3
        consumer = ESampTrainEngine(hidden_dim=2, lr=1e-3, enabled=True)
        graph = consumer.state.graphs["model_bank"]
        graph.rows = rows
        graph.slots = 2
        graph.external_inputs = True
        external_src = torch.full((rows, hidden), 5.0, device=device, dtype=dtype)
        external_tgt = torch.full((rows, hidden), 6.0, device=device, dtype=dtype)
        graph.buffers = {
            "slot_ids": torch.zeros((rows,), device=device, dtype=torch.long),
            "src": external_src,
            "tgt": external_tgt,
            "valid": torch.zeros((rows,), device=device, dtype=torch.float32),
            "slot_sum": torch.zeros((2,), device=device, dtype=torch.float32),
            "slot_cnt": torch.zeros((2,), device=device, dtype=torch.float32),
            "loss": torch.zeros((1,), device=device, dtype=torch.float32),
        }
        src = torch.ones((rows, hidden), device=device, dtype=dtype)
        tgt = torch.ones((rows, hidden), device=device, dtype=dtype)
        slot_ids = torch.tensor([0, 1], device=device, dtype=torch.long)

        ok = consumer._stage_graph_inputs(
            "model_bank",
            src=src,
            tgt=tgt,
            active_rows=2,
            slot_ids=slot_ids,
            graph=graph,
        )

        self.assertTrue(ok)
        self.assertTrue(torch.equal(graph.buffers["src"], external_src))
        self.assertTrue(torch.equal(graph.buffers["tgt"], external_tgt))
        self.assertTrue(torch.equal(graph.buffers["slot_ids"][:2], slot_ids))
        self.assertTrue(torch.equal(graph.buffers["valid"], torch.tensor([1.0, 1.0, 0.0])))

    def test_read_and_reset_stats_handles_inference_mode_buffers(self) -> None:
        device = torch.device("cpu")
        consumer = ESampTrainEngine(hidden_dim=4, lr=1e-3, enabled=True)
        with torch.inference_mode():
            consumer.state.stats = engine_module._StatsBuffers(
                loss_sum=torch.ones((1,), device=device, dtype=torch.float32),
                loss_count=torch.ones((1,), device=device, dtype=torch.int64),
            )

        stats = consumer.read_and_reset_stats(sync=False)

        self.assertEqual(stats.loss_count, 1)
        self.assertAlmostEqual(stats.loss_avg, 1.0, places=6)

    def test_read_and_reset_per_request_stats_handles_inference_mode_buffers(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=4, lr=1e-3, enabled=True, per_request_models=True)
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            with torch.inference_mode():
                consumer.ensure_resources(
                    device=torch.device("cpu"),
                    rows=4,
                    hidden_size=6,
                    hidden_dtype=torch.float32,
                )
                consumer._ensure_per_request_entry(3)

        stats = consumer.read_and_reset_per_request_stats(sync=False)

        self.assertIn(3, stats)
        self.assertEqual(stats[3].loss_count, 0)
        self.assertAlmostEqual(stats[3].loss_avg, 0.0, places=6)
        self.assertEqual(stats[3].trace_losses, ())

    def test_read_and_reset_per_request_stats_returns_trace_losses(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=4, lr=1e-3, enabled=True, per_request_models=True)
        consumer.state.trace_per_request_losses = True
        consumer.state.trace_max_points = 8
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=6,
                hidden_dtype=torch.float32,
            )
        consumer._ensure_per_request_entry(7)
        entry = consumer.state.per_request[7]
        entry.loss_sum.add_(6.0)
        entry.loss_count.add_(3)
        entry.trace_losses.extend([3.0, 2.0, 1.0])

        stats = consumer.read_and_reset_per_request_stats(sync=False)

        self.assertEqual(stats[7].loss_count, 3)
        self.assertAlmostEqual(stats[7].loss_avg, 2.0, places=6)
        self.assertEqual(stats[7].trace_losses, (3.0, 2.0, 1.0))

    def test_ensure_resources_creates_train_buffers_outside_inference_mode(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=4, lr=1e-3, enabled=True)
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            with torch.inference_mode():
                consumer.ensure_resources(
                    device=torch.device("cpu"),
                    rows=4,
                    hidden_size=8,
                    hidden_dtype=torch.bfloat16,
                )

        pipeline = consumer._require_pipeline()
        stats = consumer._require_stats()
        self.assertFalse(pipeline.src.is_inference())
        self.assertFalse(pipeline.tgt.is_inference())
        self.assertFalse(stats.loss_sum.is_inference())
        self.assertFalse(stats.loss_count.is_inference())

    def test_ensure_resources_preserves_deque_pending_train_queue(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=4, lr=1e-3, enabled=True)
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=8,
                hidden_dtype=torch.float32,
            )

        self.assertIsInstance(consumer.state.pending_train_queue, deque)

    def test_ensure_resources_rebuilds_device_bound_runtime_state_on_device_change(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=4, lr=1e-3, enabled=True)
        with mock.patch("torch.cuda.Stream", side_effect=[object(), object(), object(), object()]), mock.patch(
            "torch.cuda.Event", side_effect=lambda **_: object()
        ), mock.patch("torch.cuda.stream", return_value=contextlib.nullcontext()):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=8,
                hidden_dtype=torch.float32,
            )
            first_forward = consumer.state.forward_stream
            first_train = consumer.state.train_stream
            first_src_event = consumer.state.src_ready_events[0]

            consumer.ensure_resources(
                device=torch.device("meta"),
                rows=4,
                hidden_size=8,
                hidden_dtype=torch.float32,
            )

        self.assertIsNotNone(consumer.state.forward_stream)
        self.assertIsNotNone(consumer.state.train_stream)
        self.assertIsNot(consumer.state.forward_stream, first_forward)
        self.assertIsNot(consumer.state.train_stream, first_train)
        self.assertIsNot(consumer.state.src_ready_events[0], first_src_event)

    def test_engine_ensure_resources_creates_train_buffers_outside_inference_mode_promoted_core(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=4, lr=1e-3, enabled=True)
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            with torch.inference_mode():
                consumer.ensure_resources(
                    device=torch.device("cpu"),
                    rows=4,
                    hidden_size=8,
                    hidden_dtype=torch.bfloat16,
                )

        self.assertIsNotNone(consumer.state.pipeline)
        self.assertIsNotNone(consumer.state.stats)
        self.assertFalse(consumer.state.pipeline.src.is_inference())
        self.assertFalse(consumer.state.pipeline.tgt.is_inference())
        self.assertFalse(consumer.state.stats.loss_sum.is_inference())
        self.assertFalse(consumer.state.stats.loss_count.is_inference())

    def test_engine_ensure_shared_models_requires_initialized_hidden_metadata(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=4, lr=1e-3, enabled=True)

        with self.assertRaisesRegex(RuntimeError, "hidden metadata|ensure_resources"):
            consumer._ensure_shared_models()

    def test_engine_ensure_model_bank_resources_requires_initialized_hidden_metadata(self) -> None:
        consumer = ESampTrainEngine(
            hidden_dim=4,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=4,
        )

        with self.assertRaisesRegex(RuntimeError, "hidden metadata|ensure_resources"):
            consumer._ensure_model_bank_resources()

    def test_engine_model_bank_resource_helper_no_longer_exposes_optional_setup_wrapper(self) -> None:
        params = tuple(inspect.signature(ESampTrainEngine._ensure_model_bank_resources).parameters)

        self.assertEqual(params, ("self",))

    def test_engine_ensure_resources_initializes_shared_models_when_shared_mode_is_active(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=4, lr=1e-3, enabled=True, per_request_models=False)

        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=8,
                hidden_dtype=torch.float32,
            )

        shared = consumer._require_shared()
        self.assertIsNotNone(shared.train_model)
        self.assertIsNotNone(shared.forward_model)
        self.assertIsNotNone(shared.optimizer)

    def test_engine_ensure_resources_initializes_model_bank_when_model_bank_mode_is_active(self) -> None:
        consumer = ESampTrainEngine(
            hidden_dim=4,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=4,
        )

        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=8,
                hidden_dtype=torch.float32,
            )

        self.assertIsNotNone(consumer.state.model_bank_optimizer)
        self.assertIsNotNone(consumer.state.model_bank)
        self.assertIn("a", consumer.state.model_bank.as_dict())

    def test_launch_paths_assume_ready_buffers_after_ensure_resources(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=4, lr=1e-3, enabled=True)

        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=8,
                hidden_dtype=torch.float32,
            )

        pipeline = consumer._require_pipeline()
        self.assertIsNotNone(pipeline.src)
        self.assertIsNotNone(pipeline.tgt)
        self.assertFalse(hasattr(pipeline, "pred"))

    def test_engine_hot_path_no_longer_uses_redundant_none_guards_for_ready_buffers(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "engine.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("not s.enabled or s.forward_stream is None or s.src_buf is None or s.pred_buf is None", text)
        self.assertNotIn("not s.enabled or s.forward_stream is None or s.tgt_buf is None or s.current_step_slot is None", text)
        self.assertNotIn("if s.src_buf is None or s.tgt_buf is None:", text)

    def test_engine_model_bank_training_accumulates_nonzero_slot_loss_promoted_core(self) -> None:
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=2,
            model_bank_rank=2,
            model_bank_train_cudagraph=False,
        )
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=6,
                hidden_dtype=torch.float32,
            )

        src = torch.randn((4, 6), dtype=torch.float32)
        tgt = src + 0.25
        consumer._train_model_bank(src, tgt, 4, [0, 0, 1, 1])

        stats = consumer._require_stats()
        self.assertGreater(float(stats.loss_sum.item()), 0.0)
        self.assertEqual(int(stats.loss_count.item()), 2)

    def test_model_bank_training_uses_prefix_view_when_rows_are_already_prompt_compact(self) -> None:
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=2,
            model_bank_rank=2,
            model_bank_train_cudagraph=False,
        )
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=6,
                hidden_dtype=torch.float32,
            )

        src = torch.randn((4, 6), dtype=torch.float32)
        tgt = src + 0.25
        with mock.patch.object(consumer, "_train_model_bank_kernel") as p_kernel:
            consumer._train_model_bank(src, tgt, 2, [7, 9])

        kwargs = p_kernel.call_args.kwargs
        self.assertEqual(int(kwargs["src"].shape[0]), 2)
        self.assertEqual(int(kwargs["tgt"].shape[0]), 2)
        self.assertEqual(kwargs["src"].data_ptr(), src[:2].data_ptr())
        self.assertEqual(kwargs["tgt"].data_ptr(), tgt[:2].data_ptr())

    def test_model_bank_pipeline_slot_graph_capture_uses_external_prefix_inputs(self) -> None:
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=2,
            model_bank_rank=2,
            model_bank_train_cudagraph=False,
        )
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=6,
                hidden_dtype=torch.float32,
            )

        src = torch.randn((4, 6), dtype=torch.float32)
        tgt = src + 0.25
        with mock.patch.object(consumer, "_graph_enabled", return_value=True), mock.patch.object(
            consumer,
            "_capture_graph",
            return_value=True,
        ) as p_capture:
            consumer._train_model_bank(src, tgt, 2, [7, 9], pipeline_slot=1)

        self.assertEqual(p_capture.call_args.args, ("model_bank",))
        self.assertEqual(p_capture.call_args.kwargs["rows"], 2)
        self.assertEqual(p_capture.call_args.kwargs["src_input"].data_ptr(), src[:2].data_ptr())
        self.assertEqual(p_capture.call_args.kwargs["tgt_input"].data_ptr(), tgt[:2].data_ptr())

    def test_model_bank_training_uses_one_row_per_slot_for_expanded_samples(self) -> None:
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=2,
            model_bank_rank=2,
            model_bank_train_cudagraph=False,
        )
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=6,
                hidden_size=4,
                hidden_dtype=torch.float32,
            )

        src = torch.randn((6, 4), dtype=torch.float32)
        tgt = src + 0.25
        seen: dict[str, list[int]] = {}

        def _fake_kernel(**kwargs):
            seen["slot_ids"] = [int(x) for x in kwargs["slot_tensor"].detach().cpu().tolist()]
            seen["src_rows"] = [int(x) for x in range(int(kwargs["src"].shape[0]))]
            kwargs["stats"].loss_count.add_(len(kwargs["unique_slot_ids"]))

        with mock.patch.object(consumer, "_train_model_bank_kernel", side_effect=_fake_kernel):
            consumer._train_model_bank(src, tgt, 6, [0, 0, 0, 1, 1, 1])

        self.assertEqual(seen["slot_ids"], [0, 1])
        self.assertEqual(int(consumer._require_stats().loss_count.item()), 2)

    def test_model_bank_graph_capture_uses_active_training_rows_not_pipeline_capacity(self) -> None:
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=2,
            model_bank_rank=2,
            model_bank_train_cudagraph=False,
        )
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=6,
                hidden_size=4,
                hidden_dtype=torch.float32,
            )

        src = torch.randn((6, 4), dtype=torch.float32)
        tgt = src + 0.25
        with mock.patch.object(consumer, "_graph_enabled", return_value=True), mock.patch.object(
            consumer,
            "_capture_graph",
            return_value=True,
        ) as p_capture:
            consumer._train_model_bank(src, tgt, 6, [0, 0, 0, 1, 1, 1])

        p_capture.assert_called_once()
        self.assertEqual(p_capture.call_args.args, ("model_bank",))
        self.assertEqual(p_capture.call_args.kwargs["rows"], 2)
        self.assertIsNone(p_capture.call_args.kwargs["src_input"])
        self.assertIsNone(p_capture.call_args.kwargs["tgt_input"])

    def test_model_bank_auto_slot_capacity_respects_allocated_rows(self) -> None:
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=0,
            model_bank_rank=2,
        )
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=6,
                hidden_dtype=torch.float32,
            )

        self.assertEqual([consumer._assign_model_bank_slot(i) for i in range(4)], [0, 1, 2, 3])
        self.assertEqual(getattr(consumer.state, "per_request_trainers", {}), {})
        with self.assertRaisesRegex(RuntimeError, "slots exhausted"):
            consumer._assign_model_bank_slot(4)

    def test_per_request_training_requires_valid_prompt_metadata(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=2, lr=1e-3, enabled=True, per_request_models=True)
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=6,
                hidden_dtype=torch.float32,
            )

        src = torch.randn((2, 6), dtype=torch.float32)
        tgt = src + 0.1
        with self.assertRaisesRegex(RuntimeError, "prompt metadata"):
            consumer._train_per_request(src, tgt, 2, [-1, -1])

    def test_model_bank_training_requires_valid_prompt_metadata(self) -> None:
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=2,
            model_bank_rank=2,
        )
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=6,
                hidden_dtype=torch.float32,
            )

        src = torch.randn((2, 6), dtype=torch.float32)
        tgt = src + 0.1
        with self.assertRaisesRegex(RuntimeError, "prompt metadata"):
            consumer._train_model_bank(src, tgt, 2, [-1, -1])

    def test_model_bank_training_updates_sampling_lookup_only_for_new_prompt_slots(self) -> None:
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=2,
            model_bank_rank=2,
        )
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=6,
                hidden_dtype=torch.float32,
            )

        src = torch.randn((2, 6), dtype=torch.float32)
        tgt = src + 0.1
        with mock.patch.object(consumer, "_write_sampling_lookup_dense_entries", wraps=consumer._write_sampling_lookup_dense_entries) as p_write, mock.patch.object(
            consumer,
            "_maybe_run_model_bank_graph",
            return_value=True,
        ):
            consumer._train_model_bank(src, tgt, 2, [3, 4], pipeline_slot=0)
            consumer._train_model_bank(src, tgt, 2, [3, 4], pipeline_slot=0)

        self.assertEqual(p_write.call_count, 1)

    def test_engine_uses_configured_adaptation_pipeline_slots(self) -> None:
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=2,
            model_bank_rank=2,
            adaptation_pipeline_slots=7,
        )
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=6,
                hidden_dtype=torch.float32,
            )

        pipeline = consumer._require_pipeline()
        self.assertEqual(tuple(pipeline.src.shape), (7, 4, 6))
        self.assertEqual(len(consumer.state.src_staged_events), 7)

    def test_engine_single_adaptation_stream_mode_reuses_one_stream(self) -> None:
        stream = object()
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            adaptation_stream_mode="single",
            adaptation_stream_priority=-1,
        )
        with mock.patch("torch.cuda.Stream", return_value=stream) as make_stream, mock.patch(
            "torch.cuda.Event"
        ), mock.patch("torch.cuda.stream", return_value=contextlib.nullcontext()):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=6,
                hidden_dtype=torch.float32,
            )

        self.assertIs(consumer.state.forward_stream, stream)
        self.assertIs(consumer.state.train_stream, stream)
        make_stream.assert_called_once_with(device=torch.device("cpu"), priority=-1)

    def test_engine_serial_adaptation_stream_mode_uses_current_stream(self) -> None:
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            adaptation_stream_mode="serial",
        )
        with mock.patch("torch.cuda.Stream") as make_stream, mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=6,
                hidden_dtype=torch.float32,
            )

        self.assertIsNone(consumer.state.forward_stream)
        self.assertIsNone(consumer.state.train_stream)
        make_stream.assert_not_called()

    def test_engine_shared_training_attempts_graph_capture_when_enabled(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=2, lr=1e-3, enabled=True)
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=6,
                hidden_dtype=torch.float32,
            )

        src = torch.randn((4, 6), dtype=torch.float32)
        tgt = src + 0.1
        with mock.patch.object(consumer, "_graph_enabled", return_value=True), mock.patch.object(
            consumer, "_capture_graph", return_value=True
        ) as p_capture:
            consumer._train_shared(src, tgt, 4)

        p_capture.assert_called_once_with("shared")

    def test_shared_main_path_graph_replay_failure_disables_graph_and_falls_back(self) -> None:
        device = torch.device("cpu")
        dtype = torch.float32
        hidden = 4
        rows = 3
        consumer = ESampTrainEngine(hidden_dim=2, lr=1e-3, enabled=True)
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=device,
                rows=rows,
                hidden_size=hidden,
                hidden_dtype=dtype,
            )
        consumer.state.model_bank_train_cudagraph = True
        shared_graph = consumer.state.graphs["shared"]
        shared_graph.capture_state = "captured"
        shared_graph.replay_graph = self._ReplayFailGraph()
        shared_graph.rows = rows
        shared_graph.buffers = {
            "src": torch.zeros((rows, hidden), device=device, dtype=dtype),
            "tgt": torch.zeros((rows, hidden), device=device, dtype=dtype),
            "valid": torch.zeros((rows,), device=device, dtype=torch.float32),
            "loss": torch.zeros((1,), device=device, dtype=torch.float32),
        }
        src = torch.randn((rows, hidden), device=device, dtype=dtype)
        tgt = torch.randn((rows, hidden), device=device, dtype=dtype)

        consumer._train_shared(src, tgt, rows)

        self.assertEqual(shared_graph.capture_state, "disabled")
        self.assertGreaterEqual(int(consumer._require_stats().loss_count.item()), 1)

    def test_engine_model_bank_training_attempts_graph_capture_when_enabled(self) -> None:
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=2,
            model_bank_rank=2,
            model_bank_train_cudagraph=False,
        )
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=6,
                hidden_dtype=torch.float32,
            )

        src = torch.randn((4, 6), dtype=torch.float32)
        tgt = src + 0.25
        with mock.patch.object(consumer, "_graph_enabled", return_value=True), mock.patch.object(
            consumer, "_capture_graph", return_value=True
        ) as p_capture:
            consumer._train_model_bank(src, tgt, 4, [0, 0, 1, 1])

        p_capture.assert_called_once()
        self.assertEqual(p_capture.call_args.args, ("model_bank",))
        self.assertEqual(p_capture.call_args.kwargs["rows"], 2)
        self.assertIsNone(p_capture.call_args.kwargs["src_input"])
        self.assertIsNone(p_capture.call_args.kwargs["tgt_input"])

    def test_model_bank_main_path_graph_replay_failure_disables_graph_and_falls_back(self) -> None:
        device = torch.device("cpu")
        dtype = torch.float32
        hidden = 4
        rows = 4
        slots = 4
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=slots,
            model_bank_rank=2,
        )
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=device,
                rows=rows,
                hidden_size=hidden,
                hidden_dtype=dtype,
            )
        consumer.state.model_bank_train_cudagraph = True
        bank_graph = consumer.state.graphs["model_bank"]
        bank_graph.capture_state = "captured"
        bank_graph.replay_graph = self._ReplayFailGraph()
        bank_graph.rows = rows
        bank_graph.buffers = {
            "slot_ids": torch.zeros((rows,), device=device, dtype=torch.long),
            "src": torch.zeros((rows, hidden), device=device, dtype=dtype),
            "tgt": torch.zeros((rows, hidden), device=device, dtype=dtype),
            "valid": torch.zeros((rows,), device=device, dtype=torch.float32),
            "slot_sum": torch.zeros((slots,), device=device, dtype=torch.float32),
            "slot_cnt": torch.zeros((slots,), device=device, dtype=torch.float32),
            "loss": torch.zeros((1,), device=device, dtype=torch.float32),
        }
        src = torch.randn((rows, hidden), device=device, dtype=dtype)
        tgt = src + 0.1

        consumer._train_model_bank(src, tgt, rows, [0, 0, 1, 1])

        self.assertEqual(bank_graph.capture_state, "disabled")
        self.assertGreaterEqual(int(consumer._require_stats().loss_count.item()), 2)

    def test_promoted_engine_ensure_resources_supports_model_bank_runtime_call_path(self) -> None:
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=2,
            model_bank_rank=2,
        )
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=6,
                hidden_dtype=torch.float32,
            )

        self.assertIsNotNone(consumer.state.model_bank)
        self.assertIsNotNone(consumer.state.model_bank.a)

    def test_engine_compat_surface_rejects_none_writes_to_model_bank_params(self) -> None:
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=2,
            model_bank_rank=2,
        )
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=6,
                hidden_dtype=torch.float32,
            )

        with self.assertRaises((AttributeError, RuntimeError, TypeError)):
            consumer.model_bank_a = None

    def test_engine_compat_surface_rejects_graph_buffer_mutation_writes(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=2, lr=1e-3, enabled=True)

        with self.assertRaises((AttributeError, RuntimeError, TypeError)):
            consumer.shared_graph_src = torch.zeros((2, 4), dtype=torch.float32)

    def test_engine_compat_surface_param_reads_do_not_return_none_for_missing_resources(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=2, lr=1e-3, enabled=True)

        with self.assertRaises(AttributeError):
            _ = consumer.model_bank_a
        with self.assertRaises(AttributeError):
            _ = consumer.src_buf
        with self.assertRaises(AttributeError):
            _ = consumer.loss_sum

    def test_engine_rejects_unknown_attribute_reads_and_writes(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=2, lr=1e-3, enabled=True)

        with self.assertRaises(AttributeError):
            _ = consumer.not_a_real_attr
        with self.assertRaises(AttributeError):
            consumer.not_a_real_attr = 1

    def test_engine_runtime_surface_no_longer_exposes_stream_and_queue_internals(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=2, lr=1e-3, enabled=True)

        for name in ("forward_stream", "train_stream", "pending_train_queue", "src_ready_events", "tgt_done_events"):
            with self.assertRaises(AttributeError):
                getattr(consumer, name)
            with self.assertRaises(AttributeError):
                setattr(consumer, name, object())
        with self.assertRaises(AttributeError):
            consumer.state = object()

    def test_model_bank_forward_raises_when_active_params_are_missing(self) -> None:
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=2,
            model_bank_rank=2,
        )
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=6,
                hidden_dtype=torch.float32,
            )
        consumer.state.model_bank = None

        slot_ids = torch.zeros((2,), dtype=torch.long)
        src = torch.zeros((2, 6), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "model bank|parameter"):
            consumer._model_bank_forward_locked(slot_ids, src)

    def test_read_and_reset_stats_raises_when_enabled_engine_lost_stats_buffers(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=2, lr=1e-3, enabled=True)
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=6,
                hidden_dtype=torch.float32,
            )
        consumer.state.stats = None

        with self.assertRaisesRegex(RuntimeError, "stats buffers"):
            consumer.read_and_reset_stats(sync=False)

    def test_read_and_reset_stats_allows_pristine_enabled_engine_before_resource_init(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=2, lr=1e-3, enabled=True)

        stats = consumer.read_and_reset_stats(sync=False)

        self.assertEqual(stats.loss_avg, 0.0)
        self.assertEqual(stats.loss_count, 0)

    def test_read_and_reset_per_request_stats_raises_for_incomplete_entry(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=2, lr=1e-3, enabled=True, per_request_models=True)
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=6,
                hidden_dtype=torch.float32,
            )
        entry = consumer._ensure_per_request_entry(7)
        entry.loss_sum = None
        entry.loss_count = None

        with self.assertRaises((AttributeError, RuntimeError)):
            consumer.read_and_reset_per_request_stats(sync=False)

    def test_engine_no_longer_uses_silent_fallbacks_in_model_bank_and_stats_paths(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "engine.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("return src", text)
        self.assertNotIn(
            "if s.loss_sum is None or s.loss_count is None:\n            return ESampStats(loss_avg=0.0, loss_count=0)",
            text,
        )
        self.assertNotIn(
            "if entry.loss_sum is None or entry.loss_count is None:\n                continue",
            text,
        )
        self.assertNotIn("graph.graph is None or self.state.loss_sum is None or self.state.loss_count is None", text)
        self.assertNotIn("graph.graph is None or self.loss_sum is None or self.loss_count is None", text)

    def test_engine_uses_torch_layer_norm_for_model_bank_output_normalization(self) -> None:
        engine_text = (Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "engine.py").read_text(
            encoding="utf-8"
        )
        backend_text = (
            Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "model_bank_backend.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("def _layernorm_rows", engine_text)
        self.assertIn("F.layer_norm(", backend_text)

    def test_graph_state_uses_single_capture_state_not_disabled_plus_none(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "engine.py").read_text(
            encoding="utf-8"
        )
        start = text.index("class _GraphState:")
        end = text.index("class _ModelBankParams:")
        block = text[start:end]
        self.assertIn("capture_state:", block)
        self.assertIn("replay_graph:", block)
        self.assertNotIn("disabled: bool", block)
        self.assertNotIn("graph: torch.cuda.CUDAGraph | None", block)

    def test_engine_no_longer_exposes_graph_compat_alias_map(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "engine.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_GRAPH_RENAMES = {", text)

    def test_engine_no_longer_exposes_param_or_queue_compat_alias_maps(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "engine.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_PARAM_RENAMES = {", text)
        self.assertNotIn("_QUEUE_RENAMES = {", text)
        self.assertNotIn("_STATE_RENAMES = {", text)
        self.assertNotIn("_REJECTED_COMPAT_ATTRS = {", text)

    def test_engine_state_uses_resource_objects_for_shared_pipeline_and_stats(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "engine.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("class _SharedTrainResources:", text)
        self.assertIn("class _PipelineBuffers:", text)
        self.assertIn("class _StatsBuffers:", text)
        start = text.index("class _EngineState:")
        end = text.index("class _ESampEngineCore:")
        block = text[start:end]
        self.assertNotIn("forward_model: _ResidualModel | None = None", block)
        self.assertNotIn("train_model: _ResidualModel | None = None", block)
        self.assertIn("shared: _SharedTrainResources | None", block)
        self.assertNotIn("src_buf: torch.Tensor | None = None", block)
        self.assertNotIn("tgt_buf: torch.Tensor | None = None", block)
        self.assertNotIn("pred_buf: torch.Tensor | None = None", block)
        self.assertNotIn("loss_sum: torch.Tensor | None = None", block)
        self.assertNotIn("loss_count: torch.Tensor | None = None", block)
        self.assertNotIn("def forward_model(self)", block)
        self.assertNotIn("def train_model(self)", block)
        self.assertNotIn("def optimizer(self)", block)
        self.assertNotIn("def src_buf(self)", block)
        self.assertNotIn("def tgt_buf(self)", block)
        self.assertNotIn("def pred_buf(self)", block)
        self.assertNotIn("def loss_sum(self)", block)
        self.assertNotIn("def loss_count(self)", block)

    def test_engine_uses_require_helpers_for_concrete_resource_boundaries(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "engine.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def _require_shared(self) -> _SharedTrainResources:", text)
        self.assertIn("def _require_pipeline(self) -> _PipelineBuffers:", text)
        self.assertIn("def _require_stats(self) -> _StatsBuffers:", text)

    def test_engine_drops_proxy_attrs_duplicate_replay_and_low_value_require_wrappers(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "engine.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_STATE_ATTRS = {", text)
        self.assertNotIn("def __getattr__(self, name: str):", text)
        self.assertNotIn("def _replay_graph(", text)
        self.assertNotIn("def _require_model_bank(self) -> _ModelBankParams:", text)
        self.assertNotIn("def _require_model_bank_optimizer(self) -> torch.optim.AdamW:", text)
        self.assertNotIn("def _require_forward_stream(self) -> torch.cuda.Stream:", text)
        self.assertNotIn("def _require_train_stream(self) -> torch.cuda.Stream:", text)
        self.assertNotIn("def _require_current_step_slot(self) -> int:", text)

    def test_engine_hot_paths_no_longer_close_with_nullable_resource_asserts(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "engine.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("assert s.train_model is not None and s.forward_model is not None and s.optimizer is not None", text)
        self.assertNotIn("assert s.loss_sum is not None and s.loss_count is not None", text)
        self.assertNotIn("assert s.model_bank_optimizer is not None and s.loss_sum is not None and s.loss_count is not None", text)
        self.assertNotIn("assert s.src_buf is not None and s.tgt_buf is not None", text)
        self.assertNotIn("assert s.forward_stream is not None and s.src_buf is not None and s.pred_buf is not None", text)
        self.assertNotIn("assert s.forward_stream is not None and s.tgt_buf is not None and s.current_step_slot is not None", text)
        self.assertNotIn("assert self.state.loss_sum is not None and self.state.loss_count is not None", text)
        self.assertNotIn("assert self.loss_sum is not None and self.loss_count is not None", text)

    def test_engine_algorithm_paths_delegate_to_kernel_helpers(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "engine.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def _train_shared_kernel(", text)
        self.assertIn("def _train_model_bank_kernel(", text)
        self.assertIn("def _maybe_run_shared_graph(", text)
        self.assertIn("def _maybe_run_model_bank_graph(", text)
        self.assertIn("def _launch_forward_kernel(", text)
        self.assertIn("def _launch_target_kernel(", text)
        self.assertIn("def _require_prompt_rows(", text)
        self.assertIn("def _replay_graph_with_disable_fallback(", text)
        self.assertIn("self._train_shared_kernel(", text)
        self.assertIn("self._train_model_bank_kernel(", text)
        self.assertIn("self._maybe_run_shared_graph(", text)
        self.assertIn("self._maybe_run_model_bank_graph(", text)
        self.assertIn("self._launch_forward_kernel(", text)
        self.assertIn("self._launch_target_kernel(", text)
        self.assertIn("self._require_prompt_rows(", text)
        self.assertIn("self._replay_graph_with_disable_fallback(", text)
        self.assertNotIn("self._require_shared()", text[text.index("def _launch_forward_kernel(") : text.index("def launch_forward(")])
        self.assertNotIn("shared_forward_model", text[text.index("def _launch_forward_kernel(") : text.index("def launch_forward(")])
        self.assertNotIn("pipeline.pred", text)

    def test_engine_launch_entrypoints_derive_device_from_hidden_tensor(self) -> None:
        self.assertEqual(tuple(inspect.signature(ESampTrainEngine.launch_forward).parameters), ("self", "source_hidden"))
        self.assertEqual(tuple(inspect.signature(ESampTrainEngine.launch_target).parameters), ("self", "target_hidden"))
        self.assertEqual(
            tuple(inspect.signature(ESampTrainEngine.launch_step).parameters),
            ("self", "source_hidden", "target_hidden"),
        )
        text = (Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "consumer.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("source_device=", text)

    def test_launch_forward_requires_hidden_device_match_runtime_device(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=4, lr=1e-3, enabled=True)
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=8,
                hidden_dtype=torch.float32,
            )

        with self.assertRaisesRegex(RuntimeError, "device mismatch"):
            consumer.launch_forward(torch.zeros((1, 8), device=torch.device("meta"), dtype=torch.float32))

    def test_launch_forward_rejects_second_step_before_previous_step_is_flushed(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=4, lr=1e-3, enabled=True)
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=8,
                hidden_dtype=torch.float32,
            )

        rows = torch.zeros((1, 8), device=torch.device("cpu"), dtype=torch.float32)
        with mock.patch("torch.cuda.current_stream", return_value=mock.Mock()), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.launch_forward(rows)
            with self.assertRaisesRegex(RuntimeError, "launch order"):
                consumer.launch_forward(rows)

    def test_engine_no_longer_tracks_unused_model_bank_queue_state(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "engine.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("model_bank_queue:", text)
        self.assertNotIn("model_bank_steps_since_flush:", text)

    def test_model_bank_mode_keeps_per_request_stats_but_not_per_request_trainers(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "engine.py").read_text(
            encoding="utf-8"
        )
        stats_block = text[text.index("class _PerRequestEntry:") : text.index("class _PerRequestTrainer:")]
        self.assertNotIn("train_model:", stats_block)
        self.assertNotIn("forward_model:", stats_block)
        self.assertNotIn("optimizer:", stats_block)
        self.assertIn("class _PerRequestTrainer:", text)
        self.assertIn("per_request_trainers:", text)

    def test_engine_no_longer_materializes_unused_prediction_scratch(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "engine.py").read_text(
            encoding="utf-8"
        )
        pipeline_block = text[text.index("class _PipelineBuffers:") : text.index("class _StatsBuffers:")]
        self.assertNotIn("pred: torch.Tensor", pipeline_block)
        self.assertNotIn("pred_ready_events", text)

    def test_engine_uses_typed_model_bank_storage_instead_of_dict_gets(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "engine.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("class _ModelBankParams", text)
        self.assertNotIn("model_bank_params.get(", text)

    def test_typed_model_bank_storage_does_not_use_internal_optional_fields(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "engine.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("out_ln_weight: torch.nn.Parameter | None", text)
        self.assertNotIn("out_ln_bias: torch.nn.Parameter | None", text)

    def test_per_request_entries_are_not_partial_optional_shells(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "engine.py").read_text(
            encoding="utf-8"
        )
        start = text.index("class _PerRequestEntry:")
        end = text.index("class _GraphState:")
        block = text[start:end]
        self.assertNotIn("train_model: _ResidualModel | None", block)
        self.assertNotIn("forward_model: _ResidualModel | None", block)
        self.assertNotIn("optimizer: torch.optim.AdamW | None", block)
        self.assertNotIn("loss_sum: torch.Tensor | None", block)
        self.assertNotIn("loss_count: torch.Tensor | None", block)
        self.assertNotIn("def _ensure_per_request_stat_entry", text)


if __name__ == "__main__":
    unittest.main()
