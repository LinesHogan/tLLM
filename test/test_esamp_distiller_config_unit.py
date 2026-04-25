#!/usr/bin/env python3
"""Unit tests for ESamp distiller intervention config plumbing."""

from __future__ import annotations

import unittest
from unittest import mock

from tllm.consumers.esamp import ESampConsumerConfig
from tllm.runtime import residual_runtime
from tllm.workflows import esamp_support
from tllm.workflows.benchmarks import per_request_esamp_benchmark as bench


class ESampDistillerConfigUnitTest(unittest.TestCase):
    def test_consumer_config_exposes_distiller_intervention_fields(self) -> None:
        cfg = ESampConsumerConfig(
            enable_distiller_intervention=True,
            distiller_beta=0.75,
            distiller_sampler_backend="post_filter_exact",
        )

        self.assertTrue(cfg.enable_distiller_intervention)
        self.assertAlmostEqual(cfg.distiller_beta, 0.75)
        self.assertEqual(cfg.distiller_sampler_backend, "post_filter_exact")

    def test_configure_runtime_records_distiller_intervention_fields(self) -> None:
        residual_runtime.configure_runtime(
            graph_scratch_rows=16,
            tap_layer_paths=["a", "b"],
            source_layer_path="a",
            target_layer_path="b",
            enable_esamp_training=True,
            distiller_hidden_dim=8,
            distiller_lr=1e-3,
            enable_distiller_intervention=True,
            distiller_beta=0.6,
            distiller_sampler_backend="post_filter_exact",
        )

        self.assertTrue(residual_runtime.RUNTIME.config.enable_distiller_intervention)
        self.assertAlmostEqual(residual_runtime.RUNTIME.config.distiller_beta, 0.6)
        self.assertEqual(residual_runtime.RUNTIME.config.distiller_sampler_backend, "post_filter_exact")

    def test_configure_esamp_runtime_passes_distiller_fields_into_consumer_config(self) -> None:
        captured: dict[str, object] = {}

        class _FakeConsumer:
            def __init__(self, config):
                captured["config"] = config

        with mock.patch.object(esamp_support.runtime.RUNTIME, "consumer", None), mock.patch(
            "tllm.workflows.esamp_support.ESampConsumer",
            _FakeConsumer,
        ), mock.patch.object(esamp_support.runtime, "clear_dispatch_consumers"), mock.patch.object(
            esamp_support.runtime, "set_runtime_consumer"
        ), mock.patch.object(
            esamp_support.runtime, "configure_runtime"
        ):
            esamp_support.configure_esamp_runtime(
                graph_scratch_rows=16,
                tap_layer_paths=["a", "b"],
                source_layer_path="a",
                target_layer_path="b",
                enable_esamp_training=True,
                distiller_hidden_dim=8,
                distiller_lr=1e-3,
                enable_distiller_intervention=True,
                distiller_beta=0.9,
                distiller_sampler_backend="post_filter_exact",
            )

        cfg = captured["config"]
        self.assertTrue(cfg.enable_distiller_intervention)
        self.assertAlmostEqual(cfg.distiller_beta, 0.9)
        self.assertEqual(cfg.distiller_sampler_backend, "post_filter_exact")

    def test_benchmark_cli_exposes_distiller_intervention_flags(self) -> None:
        with mock.patch(
            "sys.argv",
            [
                "per_request_esamp_benchmark",
                "--enable-distiller-intervention",
                "--distiller-beta",
                "0.4",
                "--distiller-sampler-backend",
                "post_filter_exact",
            ],
        ):
            args = bench._parse_args()

        self.assertTrue(args.enable_distiller_intervention)
        self.assertAlmostEqual(args.distiller_beta, 0.4)
        self.assertEqual(args.distiller_sampler_backend, "post_filter_exact")

    def test_benchmark_cli_accepts_post_filter_dense_cache_backend(self) -> None:
        with mock.patch(
            "sys.argv",
            [
                "per_request_esamp_benchmark",
                "--distiller-sampler-backend",
                "post_filter_dense_cache",
            ],
        ):
            args = bench._parse_args()

        self.assertEqual(args.distiller_sampler_backend, "post_filter_dense_cache")

    def test_config_normalizes_new_backend_aliases_to_internal_canonical_names(self) -> None:
        exact = ESampConsumerConfig(distiller_sampler_backend="post_filter_exact_minp")
        reference = ESampConsumerConfig(distiller_sampler_backend="post_filter_exact_torch")

        self.assertEqual(exact.distiller_sampler_backend, "post_filter_exact")
        self.assertEqual(reference.distiller_sampler_backend, "post_filter_exact")

    def test_model_bank_forward_backend_config_is_normalized(self) -> None:
        cfg = ESampConsumerConfig(model_bank_forward_backend="triton")

        self.assertEqual(cfg.model_bank_forward_backend, "triton_grouped")

    def test_configure_runtime_records_model_bank_forward_backend(self) -> None:
        residual_runtime.configure_runtime(
            graph_scratch_rows=16,
            tap_layer_paths=["a", "b"],
            source_layer_path="a",
            target_layer_path="b",
            enable_esamp_training=True,
            distiller_hidden_dim=8,
            distiller_lr=1e-3,
            model_bank_forward_backend="triton",
        )

        self.assertEqual(residual_runtime.RUNTIME.config.model_bank_forward_backend, "triton_grouped")

    def test_benchmark_cli_exposes_model_bank_forward_backend(self) -> None:
        with mock.patch(
            "sys.argv",
            [
                "per_request_esamp_benchmark",
                "--model-bank-forward-backend",
                "triton_grouped",
            ],
        ):
            args = bench._parse_args()

        self.assertEqual(args.model_bank_forward_backend, "triton_grouped")

    def test_benchmark_cli_accepts_new_backend_aliases_for_transition_period(self) -> None:
        for backend in [
            "post_filter_exact_minp",
            "post_filter_exact_torch",
        ]:
            with self.subTest(backend=backend), mock.patch(
                "sys.argv",
                [
                    "per_request_esamp_benchmark",
                    "--distiller-sampler-backend",
                    backend,
                ],
            ):
                args = bench._parse_args()

            self.assertEqual(args.distiller_sampler_backend, backend)


if __name__ == "__main__":
    unittest.main()
