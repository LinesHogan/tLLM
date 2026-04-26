#!/usr/bin/env python3
"""Unit tests for migration finalization status."""

from __future__ import annotations

import importlib
from pathlib import Path
import unittest
from unittest import mock

from tllm.runtime.residual_runtime import SamplerPrecomputeState
from tllm.runtime.vllm_patch import port_runtime_hooks

REPO_ROOT = Path(__file__).resolve().parents[1]


class MigrationFinalizationUnitTest(unittest.TestCase):
    def test_runtime_declares_full_event_points(self) -> None:
        expected = (
            "load_model.pre",
            "load_model.post",
            "prepare_inputs.pre",
            "prepare_inputs.post",
            "layer.pre",
            "layer.post",
            "block.end",
            "stack.end",
            "execute_model.pre",
            "execute_model.post",
        )
        self.assertEqual(port_runtime_hooks.RUNTIME_EVENT_POINTS, expected)

    def test_runtime_no_longer_imports_legacy_consumer_package(self) -> None:
        root = REPO_ROOT / "tllm" / "runtime"
        self.assertFalse((root / "core.py").exists())

    def test_residual_runtime_host_no_longer_uses_runner_patch_glue_module(self) -> None:
        host_text = (REPO_ROOT / "tllm" / "runtime" / "residual_runtime.py").read_text(encoding="utf-8")
        self.assertNotIn("runner_patch_glue", host_text)
        self.assertFalse((REPO_ROOT / "tllm" / "runtime" / "runner_patch_glue.py").exists())

    def test_runtime_package_no_longer_contains_esamp_runtime_module(self) -> None:
        self.assertFalse((REPO_ROOT / "tllm" / "runtime" / "esamp_runtime.py").exists())
        self.assertTrue((REPO_ROOT / "tllm" / "runtime" / "residual_runtime.py").is_file())

    def test_runtime_package_no_longer_uses_legacy_named_modules(self) -> None:
        self.assertFalse((REPO_ROOT / "tllm" / "legacy").exists())
        self.assertFalse((REPO_ROOT / "tllm" / "runtime" / "legacy_consumer_compat.py").exists())
        self.assertFalse((REPO_ROOT / "tllm" / "runtime" / "legacy_event_bridge.py").exists())
        self.assertFalse((REPO_ROOT / "tllm" / "runtime" / "sampler_bridge" / "legacy").exists())
        self.assertTrue((REPO_ROOT / "tllm" / "runtime" / "consumer_compat.py").is_file())
        self.assertTrue((REPO_ROOT / "tllm" / "runtime" / "hidden_event_bridge.py").is_file())

    def test_public_esamp_api_no_longer_uses_legacy_training_names(self) -> None:
        support_text = (REPO_ROOT / "tllm" / "workflows" / "esamp_support.py").read_text(encoding="utf-8")
        config_text = (REPO_ROOT / "tllm" / "consumers" / "esamp" / "config.py").read_text(encoding="utf-8")
        readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("def run_esamp_throughput_case", support_text)
        self.assertNotIn("def run_" + "side" + "_train" + "_throughput_case", support_text)
        self.assertIn("enable_esamp_training", config_text)
        self.assertNotIn("enable_" + "side" + "_train", config_text)
        self.assertIn("register_consumer", readme_text)
        self.assertIn("llm.generate", readme_text)

    def test_port_runtime_hooks_no_longer_own_benchmark_request_mapping_helpers(self) -> None:
        hooks_text = (REPO_ROOT / "tllm" / "runtime" / "vllm_patch" / "port_runtime_hooks.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("def run_generate_with_request_mapping", hooks_text)
        self.assertNotIn("def run_esamp_throughput_case", hooks_text)

    def test_docs_no_longer_present_old_public_subscription_template_or_stale_vllm_version(self) -> None:
        event_catalog = (REPO_ROOT / "doc" / "reference" / "event-catalog.md").read_text(encoding="utf-8")
        vllm_compat = (REPO_ROOT / "doc" / "reference" / "vllm-compatibility.md").read_text(encoding="utf-8")
        testing_guide = (REPO_ROOT / "doc" / "development" / "testing-guide.md").read_text(encoding="utf-8")
        project_structure = (REPO_ROOT / "doc" / "reference" / "project-structure.md").read_text(encoding="utf-8")
        benchmarking = (REPO_ROOT / "doc" / "developer-guides" / "benchmarking.md").read_text(encoding="utf-8")
        esamp_guide = (REPO_ROOT / "doc" / "reference" / "esamp-usage.md").read_text(encoding="utf-8")

        self.assertNotIn("def subscriptions(", event_catalog)
        self.assertNotIn("vllm==0.7.2", vllm_compat)
        self.assertIn("tllm/verification/", testing_guide)
        self.assertNotIn("tllm/workflows/automation/automated_tests.py", testing_guide)
        self.assertIn("tllm/verification/", project_structure)
        self.assertNotIn("tllm/workflows/automation", project_structure)
        self.assertIn("ESampTrainEngine", project_structure)
        self.assertNotIn("--compare-consumer-implementations", benchmarking)
        self.assertNotIn("base_consumer / legacy", benchmarking)
        self.assertIn("ESampConsumer", esamp_guide)
        self.assertFalse((REPO_ROOT / "doc" / "dev" / "testing_guide.md").exists())

    def test_verification_harness_no_longer_lives_under_workflows(self) -> None:
        self.assertTrue((REPO_ROOT / "tllm" / "verification" / "automated_tests.py").is_file())
        self.assertFalse((REPO_ROOT / "tllm" / "workflows" / "automation").exists())

    def test_esamp_engine_files_match_planned_names(self) -> None:
        esamp_root = REPO_ROOT / "tllm" / "consumers" / "esamp"
        self.assertTrue((esamp_root / "engine.py").is_file())
        self.assertTrue((esamp_root / "consumer.py").is_file())
        self.assertFalse((esamp_root / "engine").exists())
        self.assertFalse((esamp_root / "runtime_adapter.py").exists())
        self.assertFalse((esamp_root / "runtime_support.py").exists())
        self.assertFalse((esamp_root / "workflow_support.py").exists())
        self.assertFalse((REPO_ROOT / "tllm" / "consumers" / "nn_esamp").exists())

    def test_esamp_template_helpers_no_longer_live_under_runtime_esamp_names(self) -> None:
        runtime = REPO_ROOT / "tllm" / "runtime"
        esamp = REPO_ROOT / "tllm" / "consumers" / "esamp"
        self.assertFalse((runtime / "esamp_ffn_template.py").exists())
        self.assertFalse((runtime / "esamp_templates.py").exists())
        self.assertTrue((esamp / "initializers" / "svd.py").is_file())
        self.assertFalse((esamp / "template.py").exists())
        self.assertFalse((esamp / "ffn_template.py").exists())
        self.assertFalse((esamp / "template_utils.py").exists())

    def test_runner_wrappers_are_removed(self) -> None:
        wrapper = REPO_ROOT / "tllm" / "runner" / "per_request_esamp_benchmark.py"
        self.assertFalse(wrapper.exists())

        mod = importlib.import_module("tllm.workflows.benchmarks.per_request_esamp_benchmark")
        self.assertTrue(callable(mod.main))

    def test_dummy_capture_benchmark_chain_is_removed(self) -> None:
        self.assertFalse((REPO_ROOT / "tllm" / "workflows" / "benchmarks" / "benchmark.py").exists())
        self.assertFalse((REPO_ROOT / "tllm" / "workflows" / "benchmarks" / "throughput_triplet.py").exists())
        self.assertFalse((REPO_ROOT / "tllm" / "consumers" / "dummy_tap_mlp.py").exists())

    def test_wrapped_runtime_paths_emit_pre_and_post_events(self) -> None:
        class _Runtime:
            def __init__(self) -> None:
                self.launch_consumer_from_hooks = True
                self.event_step_id = 0
                self.consumer = None
                self.dispatch_plan = type(
                    "Plan",
                    (),
                    {
                        "has_active_targets": staticmethod(lambda: True),
                        "flow_targets": staticmethod(lambda: []),
                        "required_residual_layers": staticmethod(lambda: set()),
                    },
                )()
                self.decode_count = 0
                self.decode_prompt_idxs = []
                self.source_resolved_path = ""
                self.target_resolved_path = ""
                self.tap_decode_hidden = {}
                self.sampler_precompute = SamplerPrecomputeState()

        class _Core:
            def __init__(self) -> None:
                self.RUNTIME = _Runtime()
                self._ORIG_LOAD_MODEL = lambda runner, *a, **k: "load_ok"
                self._ORIG_PREPARE_INPUTS = lambda runner, so: (None, None, None, None, None, None)
                self._ORIG_EXECUTE_MODEL = lambda runner, *a, **k: {"ok": True}

        core = _Core()
        runner = object()
        observed: list[str] = []

        def _record_event(*, event_name: str, **kwargs):
            _ = kwargs
            observed.append(event_name)
            return 0

        with mock.patch.object(port_runtime_hooks, "setup_runtime_hooks_if_active") as p_ensure, mock.patch.object(
            port_runtime_hooks, "prepare_decode_localization"
        ) as p_prepare, mock.patch.object(
            port_runtime_hooks._common_hooks, "dispatch_runtime_event", side_effect=_record_event
        ):
            p_ensure.return_value = None
            p_prepare.return_value = None
            _ = port_runtime_hooks.wrapped_load_model(core=core, runner=runner, args=tuple(), kwargs={})
            _ = port_runtime_hooks.wrapped_prepare_inputs(core=core, runner=runner, scheduler_output=object())
            _ = port_runtime_hooks.wrapped_execute_model(core=core, runner=runner, args=tuple(), kwargs={})

        self.assertEqual(
            observed,
            [
                "load_model.pre",
                "load_model.post",
                "prepare_inputs.pre",
                "prepare_inputs.post",
                "execute_model.pre",
                "execute_model.post",
            ],
        )


if __name__ == "__main__":
    unittest.main()
