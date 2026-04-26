#!/usr/bin/env python3
"""Unit tests for extracted ESamp module split."""

from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path

import torch

import tllm.consumers.esamp.engine as engine
from tllm.consumers.esamp.engine import ESampTrainEngine, ESampStats, group_row_indices_by_prompt
from tllm.consumers.esamp.model import LowRankGatedResidualModel


class ESampModuleSplitUnitTest(unittest.TestCase):
    def test_engine_uses_typed_storage_dataclass(self) -> None:
        self.assertTrue(hasattr(engine, "_EngineStorage"))
        self.assertTrue(dataclasses.is_dataclass(engine._EngineStorage))
        self.assertTrue(hasattr(engine, "ESampTrainEngine"))

    def test_engine_storage_no_longer_uses_multi_level_storage_inheritance(self) -> None:
        self.assertTrue(hasattr(engine, "_PerRequestEntry"))
        self.assertTrue(dataclasses.is_dataclass(engine._PerRequestEntry))
        for name in ("_EngineConfigStorage", "_PipelineStorage", "_PerRequestStorage", "_ModelBankStorage", "_SharedGraphStorage"):
            self.assertFalse(hasattr(engine, name), msg=name)

    def test_engine_module_no_longer_uses_post_class_method_grafting(self) -> None:
        text = Path(__file__).resolve().parents[1].joinpath("tllm", "consumers", "esamp", "engine.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("ESampTrainEngine._using_model_bank =", text)

    def test_engine_module_does_not_reintroduce_zero_like_stat_helper(self) -> None:
        text = Path(__file__).resolve().parents[1].joinpath("tllm", "consumers", "esamp", "engine.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("def _fresh_zero_like_stat(", text)

    def test_engine_module_no_longer_embeds_svd_initializer_logic(self) -> None:
        text = Path(__file__).resolve().parents[1].joinpath("tllm", "consumers", "esamp", "engine.py").read_text(
            encoding="utf-8"
        )
        for pattern in ("ridge_svd", "ffn_fast_svd", "svd_lowrank", "torch.linalg.svd"):
            self.assertNotIn(pattern, text)

    def test_engine_surface_no_longer_exposes_dead_compat_aliases(self) -> None:
        text = Path(__file__).resolve().parents[1].joinpath("tllm", "consumers", "esamp", "engine.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("model_bank_prompt_to_slot", text)

    def test_optional_svd_initializer_module_exists(self) -> None:
        root = Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp" / "initializers"
        self.assertTrue((root / "svd.py").is_file())

    def test_esamp_package_no_longer_keeps_structural_compare_file(self) -> None:
        root = Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp"
        self.assertFalse((root / "engine_structural_compare.py").exists())

    def test_engine_surface_exports_real_esamp_train_engine(self) -> None:
        self.assertEqual(ESampTrainEngine.__name__, "ESampTrainEngine")

    def test_engine_package_no_longer_exports_dummy_esamp_consumer(self) -> None:
        self.assertFalse(hasattr(engine, "DummyESampConsumer"))

    def test_esamp_engine_is_flat_module_not_package(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp"
        self.assertTrue((root / "engine.py").is_file())
        self.assertFalse((root / "engine").exists())

    def test_esamp_package_does_not_keep_invalid_plan_c_compare_file(self) -> None:
        root = Path(__file__).resolve().parents[1] / "tllm" / "consumers" / "esamp"
        self.assertFalse((root / "engine_option_c.py").exists())

    def test_grouping_matches_expected_rows(self) -> None:
        grouped = group_row_indices_by_prompt([2, 2, -1, 5, 2, 5], active_rows=6)
        self.assertEqual(grouped, {2: [0, 1, 4], 5: [3, 5]})

    def test_lowrank_model_forward_shape(self) -> None:
        m = LowRankGatedResidualModel(
            hidden_size=8,
            rank=4,
            use_output_layernorm=True,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        x = torch.randn((3, 8), dtype=torch.float32)
        y = m(x)
        self.assertEqual(tuple(y.shape), (3, 8))

    def test_stats_dataclass_fields(self) -> None:
        s = ESampStats(loss_avg=1.25, loss_count=4)
        self.assertEqual(s.loss_avg, 1.25)
        self.assertEqual(s.loss_count, 4)
        self.assertEqual(s.trace_losses, ())


if __name__ == "__main__":
    unittest.main()
