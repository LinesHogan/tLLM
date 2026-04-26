#!/usr/bin/env python3
"""Unit tests for per-request benchmark n-compat helpers."""

from __future__ import annotations

import json
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from tllm.workflows.benchmarks import per_request_esamp_benchmark as bench
from tllm.workflows.benchmarks import profile_esamp_delivery_layers


class PerRequestBenchmarkCompatUnitTest(unittest.TestCase):
    def test_effective_batch_cap_uses_full_batch_when_non_positive(self) -> None:
        self.assertEqual(bench._resolve_effective_batch_cap(128, 0), 128)
        self.assertEqual(bench._resolve_effective_batch_cap(128, -1), 128)

    def test_effective_batch_cap_respects_positive_override(self) -> None:
        self.assertEqual(bench._resolve_effective_batch_cap(128, 64), 64)
        self.assertEqual(bench._resolve_effective_batch_cap(128, 1024), 128)

    def test_graph_scratch_rows_follow_chunk_capacity_not_total_effective_batch(self) -> None:
        self.assertEqual(bench._resolve_graph_scratch_rows(0, effective_batch_cap=64), 64)
        self.assertEqual(bench._resolve_graph_scratch_rows(0, effective_batch_cap=128), 128)
        self.assertEqual(bench._resolve_graph_scratch_rows(256, effective_batch_cap=64), 256)

    def test_model_bank_slots_follow_prompt_capacity_when_unspecified(self) -> None:
        self.assertEqual(bench._resolve_model_bank_slots(0, effective_batch_cap=64, prompt_count=8), 8)
        self.assertEqual(bench._resolve_model_bank_slots(0, effective_batch_cap=16, prompt_count=32), 32)
        self.assertEqual(bench._resolve_model_bank_slots(12, effective_batch_cap=64, prompt_count=8), 12)

    def test_distiller_benchmark_uses_isolated_llm_cases(self) -> None:
        self.assertTrue(bench._requires_isolated_llm_cases(enable_distiller_intervention=True, distiller_beta=0.4))
        self.assertFalse(bench._requires_isolated_llm_cases(enable_distiller_intervention=True, distiller_beta=0.0))
        self.assertFalse(bench._requires_isolated_llm_cases(enable_distiller_intervention=False, distiller_beta=0.4))

    def test_cli_default_effective_batch_cap_is_auto(self) -> None:
        with mock.patch.object(sys, "argv", ["per_request_esamp_benchmark"]):
            args = bench._parse_args()
        self.assertEqual(args.effective_batch_cap, 0)
        self.assertFalse(args.enable_distiller_intervention)
        self.assertEqual(args.distiller_beta, 0.0)

    def test_subprocess_cmd_includes_distiller_flags_when_enabled(self) -> None:
        args = SimpleNamespace(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            dtype="bfloat16",
            gpu_memory_utilization=0.5,
            max_model_len=512,
            graph_scratch_rows=64,
            source_layer_path="a",
            target_layer_path="b",
            distiller_hidden_dim=8,
            distiller_lr=1e-3,
            enable_distiller_intervention=True,
            distiller_beta=0.4,
            distiller_sampler_backend="pre_filter_dense",
            model_bank_slots=0,
            model_bank_flush_interval=1,
            model_bank_rank=16,
            model_bank_initializer="svd",
            model_bank_initializer_svd_method="ffn_fast_svd",
            model_bank_initializer_svd_ridge_lambda=1e-2,
            model_bank_initializer_svd_min_rows=32,
            model_bank_initializer_svd_max_wait_steps=4,
            benchmark_batch_size=8,
            benchmark_max_new_tokens=64,
            benchmark_warmup_rounds=1,
            benchmark_rounds=1,
            sampling_n=4,
            sampling_temperature=0.8,
            sampling_top_p=0.95,
            sampling_top_k=-1,
            trajectory_topk=1,
            cooldown_s=0.5,
            effective_batch_cap=64,
            sampling_seed=None,
            seed_base=None,
            prompt=[],
            prompt_file="",
            enforce_eager=False,
            run_model_bank_case=True,
            model_bank_use_output_layernorm=True,
            model_bank_train_cudagraph=True,
            model_bank_compact_capture=True,
            adaptation_stream_mode="single",
            adaptation_stream_priority=-1,
            benchmark_ignore_eos=True,
            benchmark_disable_prefix_caching=True,
        )

        args.trajectory_step_interval = 1
        args.trajectory_max_points = 16
        args.sampling_per_request_seed = False

        cmd = bench._build_isolated_case_subprocess_cmd(
            args,
            case_filter="model_bank_on",
            skip_trajectory=True,
        )

        self.assertIn("--enable-distiller-intervention", cmd)
        self.assertEqual(cmd[cmd.index("--distiller-beta") + 1], "0.4")
        self.assertEqual(cmd[cmd.index("--distiller-sampler-backend") + 1], "pre_filter_dense")
        self.assertEqual(cmd[cmd.index("--adaptation-stream-mode") + 1], "single")
        self.assertEqual(cmd[cmd.index("--adaptation-stream-priority") + 1], "-1")
        self.assertIn("--model-bank-compact-capture", cmd)

        args.model_bank_compact_capture = False
        cmd = bench._build_isolated_case_subprocess_cmd(
            args,
            case_filter="model_bank_on",
            skip_trajectory=True,
        )
        self.assertIn("--no-model-bank-compact-capture", cmd)

    def test_cli_no_longer_exposes_consumer_implementation_selector(self) -> None:
        with mock.patch.object(sys, "argv", ["per_request_esamp_benchmark"]):
            args = bench._parse_args()
        self.assertFalse(hasattr(args, "consumer_implementation"))
        self.assertFalse(hasattr(args, "compare_consumer_implementations"))

    def test_isolated_case_subprocess_disables_vllm_compile_cache(self) -> None:
        args = SimpleNamespace(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            dtype="bfloat16",
            gpu_memory_utilization=0.5,
            max_model_len=512,
            graph_scratch_rows=64,
            source_layer_path="a",
            target_layer_path="b",
            distiller_hidden_dim=8,
            distiller_lr=1e-3,
            enable_distiller_intervention=True,
            distiller_beta=0.4,
            distiller_sampler_backend="pre_filter_dense",
            model_bank_slots=0,
            model_bank_flush_interval=1,
            model_bank_rank=16,
            model_bank_initializer="svd",
            model_bank_initializer_svd_method="ffn_fast_svd",
            model_bank_initializer_svd_ridge_lambda=1e-2,
            model_bank_initializer_svd_min_rows=32,
            model_bank_initializer_svd_max_wait_steps=4,
            benchmark_batch_size=8,
            benchmark_max_new_tokens=64,
            benchmark_warmup_rounds=1,
            benchmark_rounds=1,
            sampling_n=4,
            sampling_temperature=0.8,
            sampling_top_p=0.95,
            sampling_top_k=-1,
            trajectory_topk=1,
            trajectory_step_interval=1,
            trajectory_max_points=16,
            cooldown_s=0.5,
            effective_batch_cap=64,
            sampling_seed=None,
            seed_base=None,
            prompt=[],
            prompt_file="",
            enforce_eager=False,
            run_model_bank_case=True,
            model_bank_use_output_layernorm=True,
            model_bank_train_cudagraph=True,
            model_bank_compact_capture=True,
            benchmark_ignore_eos=True,
            benchmark_disable_prefix_caching=True,
            sampling_per_request_seed=False,
        )
        completed = mock.Mock(
            returncode=0,
            stdout=f"logs\n{bench.JSON_SUMMARY_PREFIX}{json.dumps({'cases': {}, 'trajectory': {}})}\n",
            stderr="",
        )

        with mock.patch.object(bench.subprocess, "run", return_value=completed) as run_mock:
            bench._run_isolated_case_subprocess(args, case_filter="model_bank_on", skip_trajectory=True)

        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual(run_mock.call_args.kwargs["env"]["VLLM_DISABLE_COMPILE_CACHE"], "1")

    def test_delivery_layer_profile_workflow_exposes_cli_entrypoints(self) -> None:
        self.assertTrue(callable(profile_esamp_delivery_layers._parse_args))
        self.assertTrue(callable(profile_esamp_delivery_layers.main))

    def test_path_hotspot_fields_are_flattened_for_json_summary(self) -> None:
        stats = SimpleNamespace(
            cpu_ms_total={
                "execute_model.dispatch_bundles_cpu": 12.5,
                "prepare_inputs.decode_localization_cpu": 3.0,
            },
            counts={
                "execute_model.dispatch_bundles_cpu": 5,
                "prepare_inputs.decode_localization_cpu": 2,
            },
        )

        fields = bench._path_hotspot_fields(stats)

        self.assertEqual(fields["path_hotspot_execute_model_dispatch_bundles_cpu_ms_total"], 12.5)
        self.assertEqual(fields["path_hotspot_execute_model_dispatch_bundles_cpu_count"], 5.0)
        self.assertEqual(fields["path_hotspot_prepare_inputs_decode_localization_cpu_ms_avg"], 1.5)

    def test_run_one_implementation_recreates_llm_per_case_when_distiller_is_enabled(self) -> None:
        args = SimpleNamespace(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            prompt=["hello"],
            prompt_file="",
            dtype="bfloat16",
            gpu_memory_utilization=0.5,
            max_model_len=512,
            enforce_eager=False,
            graph_scratch_rows=0,
            source_layer_path="model.model.layers[0].input_layernorm",
            target_layer_path="model.model.layers[-1].input_layernorm",
            distiller_hidden_dim=256,
            distiller_lr=1e-3,
            enable_distiller_intervention=True,
            distiller_beta=0.1,
            distiller_sampler_backend="pre_filter_dense",
            model_bank_slots=0,
            model_bank_flush_interval=1,
            model_bank_rank=16,
            model_bank_use_output_layernorm=True,
            model_bank_initializer="svd",
            model_bank_initializer_svd_method="ffn_fast_svd",
            model_bank_initializer_svd_ridge_lambda=1e-2,
            model_bank_initializer_svd_min_rows=32,
            model_bank_initializer_svd_max_wait_steps=4,
            model_bank_train_cudagraph=True,
            model_bank_compact_capture=False,
            run_model_bank_case=True,
            benchmark_batch_size=2,
            benchmark_max_new_tokens=16,
            benchmark_warmup_rounds=1,
            benchmark_rounds=1,
            benchmark_ignore_eos=True,
            benchmark_disable_prefix_caching=True,
            sampling_n=2,
            sampling_temperature=0.8,
            sampling_top_p=0.95,
            sampling_top_k=-1,
            sampling_seed=None,
            sampling_per_request_seed=False,
            seed_base=None,
            trajectory_topk=1,
            trajectory_step_interval=1,
            trajectory_max_points=16,
            cooldown_s=0.5,
            effective_batch_cap=4,
            emit_json_summary=False,
            case_filter="all",
            skip_trajectory=False,
        )

        with mock.patch.object(bench, "read_prompts", return_value=["p1"]), mock.patch.object(
            bench, "build_prompt_batch", return_value=["p1", "p2"]
        ), mock.patch.object(
            bench,
            "_run_isolated_case_subprocess",
            side_effect=[
                {"cases": {"single_off": {"out_tok_per_s": 1.0}}, "trajectory": {"rounds": [], "top_models": []}},
                {"cases": {"single_on": {"out_tok_per_s": 2.0}}, "trajectory": {"rounds": [], "top_models": []}},
                {"cases": {"per_request_on": {"out_tok_per_s": 3.0}}, "trajectory": {"rounds": [], "top_models": []}},
                {"cases": {"model_bank_on": {"out_tok_per_s": 4.0}}, "trajectory": {"rounds": [], "top_models": []}},
            ],
        ) as run_subproc_mock, mock.patch.object(
            bench.core, "make_llm"
        ) as make_llm_mock, mock.patch.object(
            bench, "shutdown_llm_instance"
        ) as shutdown_mock:
            payload = bench._run_one_implementation(args, "esamp")

        self.assertEqual(run_subproc_mock.call_count, 4)
        make_llm_mock.assert_not_called()
        shutdown_mock.assert_not_called()
        self.assertIn("model_bank_on", payload["cases"])

    def test_run_one_implementation_honors_all_case_model_bank_compact_capture(self) -> None:
        args = SimpleNamespace(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            prompt=["hello"],
            prompt_file="",
            dtype="bfloat16",
            gpu_memory_utilization=0.5,
            max_model_len=512,
            enforce_eager=False,
            graph_scratch_rows=0,
            source_layer_path="model.model.layers[0].input_layernorm",
            target_layer_path="model.model.layers[-1].input_layernorm",
            distiller_hidden_dim=256,
            distiller_lr=1e-3,
            enable_distiller_intervention=False,
            distiller_beta=0.0,
            distiller_sampler_backend="post_filter_exact",
            model_bank_slots=0,
            model_bank_flush_interval=1,
            model_bank_rank=16,
            model_bank_use_output_layernorm=True,
            model_bank_initializer="svd",
            model_bank_initializer_svd_method="ffn_fast_svd",
            model_bank_initializer_svd_ridge_lambda=1e-2,
            model_bank_initializer_svd_min_rows=32,
            model_bank_initializer_svd_max_wait_steps=4,
            model_bank_train_cudagraph=True,
            model_bank_forward_backend="torch",
            model_bank_compact_capture=True,
            adaptation_pipeline_slots=4,
            adaptation_stream_mode="dual",
            adaptation_stream_priority=0,
            run_model_bank_case=True,
            benchmark_batch_size=2,
            benchmark_max_new_tokens=16,
            benchmark_warmup_rounds=1,
            benchmark_rounds=1,
            benchmark_ignore_eos=True,
            benchmark_disable_prefix_caching=True,
            sampling_n=2,
            sampling_temperature=0.8,
            sampling_top_p=0.95,
            sampling_top_k=-1,
            sampling_min_p=0.0,
            sampling_seed=None,
            sampling_per_request_seed=False,
            seed_base=None,
            trajectory_topk=1,
            trajectory_step_interval=1,
            trajectory_max_points=16,
            cooldown_s=0.5,
            effective_batch_cap=4,
            emit_json_summary=False,
            case_filter="all",
            skip_trajectory=True,
        )
        case_result = {
            "req_per_s": 1.0,
            "completion_per_s": 1.0,
            "out_tok_per_s": 1.0,
            "loss_avg": 0.5,
            "loss_count": 1.0,
        }

        with mock.patch.object(bench, "read_prompts", return_value=["p1"]), mock.patch.object(
            bench, "build_prompt_batch", return_value=["p1", "p2"]
        ), mock.patch.object(bench.esamp_workflow_support, "configure_esamp_runtime"), mock.patch.object(
            bench.core, "make_llm", return_value=object()
        ), mock.patch.object(
            bench, "_run_case", return_value=case_result
        ) as run_case_mock, mock.patch.object(
            bench, "shutdown_llm_instance"
        ):
            payload = bench._run_one_implementation(args, "esamp")

        self.assertIn("model_bank_on", payload["cases"])
        model_bank_call = next(call for call in run_case_mock.call_args_list if call.kwargs["case_id"] == "model_bank_on")
        self.assertTrue(model_bank_call.kwargs["compact_capture_lane"])
        non_model_bank_calls = [call for call in run_case_mock.call_args_list if call.kwargs["case_id"] != "model_bank_on"]
        self.assertTrue(all(not call.kwargs["per_request_model_bank"] for call in non_model_bank_calls))

    def test_trajectory_prefers_step_level_trace_losses_over_round_average(self) -> None:
        args = SimpleNamespace(
            source_layer_path="a",
            target_layer_path="b",
            distiller_hidden_dim=8,
            distiller_lr=1e-3,
            model_bank_rank=4,
            model_bank_use_output_layernorm=True,
            model_bank_initializer="svd",
            model_bank_initializer_svd_method="ffn_fast_svd",
            model_bank_initializer_svd_ridge_lambda=1e-2,
            model_bank_initializer_svd_min_rows=32,
            model_bank_initializer_svd_max_wait_steps=4,
            model_bank_train_cudagraph=True,
            adaptation_pipeline_slots=4,
            adaptation_stream_mode="single",
            adaptation_stream_priority=-2,
            benchmark_warmup_rounds=0,
            benchmark_rounds=1,
            trajectory_topk=1,
            trajectory_step_interval=1,
            trajectory_max_points=16,
        )
        per_stats = {
            3: bench.core.ESampStats(loss_avg=2.0, loss_count=3, trace_losses=(3.0, 2.0, 1.0)),
            8: bench.core.ESampStats(loss_avg=5.0, loss_count=1, trace_losses=(5.0,)),
        }

        with mock.patch.object(bench.esamp_workflow_support, "configure_esamp_runtime") as configure, mock.patch.object(
            bench.esamp_workflow_support, "run_generate_with_request_mapping"
        ), mock.patch.object(
            bench.core, "set_esamp_training_enabled"
        ), mock.patch.object(
            bench.core, "read_and_reset_esamp_per_request_stats", side_effect=[{}, per_stats]
        ):
            summary = bench._run_per_request_trajectory(
                args=args,
                llm=object(),
                prompts=["p1", "p2"],
                params=[object(), object()],
                request_prompt_indices=[0, 1],
                request_sample_indices=[0, 0],
                effective_batch_cap=2,
                rows=2,
            )

        self.assertEqual(summary["top_models"][0]["prompt_idx"], 3.0)
        self.assertEqual(summary["top_models"][0]["first"], 3.0)
        self.assertEqual(summary["top_models"][0]["last"], 1.0)
        self.assertEqual(summary["top_models"][0]["min"], 1.0)
        self.assertEqual(summary["top_models"][0]["max"], 3.0)
        self.assertEqual(configure.call_args.kwargs["adaptation_stream_mode"], "single")
        self.assertEqual(configure.call_args.kwargs["adaptation_stream_priority"], -2)


if __name__ == "__main__":
    unittest.main()
