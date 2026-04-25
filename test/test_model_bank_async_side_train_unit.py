#!/usr/bin/env python3
"""Unit tests for independent model-bank async side-train experiment."""

from __future__ import annotations

import importlib
import unittest

import torch
import torch.nn.functional as F

from tllm.experiments.side_train_bank.model_bank_async_side_train import (
    ModelBankAsyncTrainer,
    PerRequestGoldTrainer,
    build_initial_slot_weights,
    generate_synthetic_sparse_steps,
)


class ModelBankAsyncSideTrainUnitTest(unittest.TestCase):
    def test_import_experiments_namespace(self) -> None:
        new_mod = importlib.import_module(
            "tllm.experiments.side_train_bank.model_bank_async_side_train"
        )
        self.assertTrue(hasattr(new_mod, "ModelBankAsyncTrainer"))
        self.assertTrue(hasattr(new_mod, "PerRequestGoldTrainer"))
        self.assertTrue(hasattr(new_mod, "build_initial_slot_weights"))

    def test_sync_bank_matches_stepwise_gold(self) -> None:
        device = torch.device("cpu")
        dtype = torch.float32
        lr = 1e-2
        w1_init, w2_init = build_initial_slot_weights(
            num_slots=5,
            hidden_size=8,
            inner_size=6,
            device=device,
            dtype=dtype,
            seed=123,
        )
        steps = generate_synthetic_sparse_steps(
            steps=12,
            num_slots=5,
            hidden_size=8,
            active_slots_per_step=3,
            rows_per_slot=4,
            device=device,
            dtype=dtype,
            seed=777,
        )

        gold = PerRequestGoldTrainer(w1_init=w1_init, w2_init=w2_init, lr=lr)
        bank = ModelBankAsyncTrainer(w1_init=w1_init, w2_init=w2_init, lr=lr, queue_flush_interval=1)

        for slot_ids, src, tgt in steps:
            loss_gold = gold.train_step(slot_ids, src, tgt)
            loss_bank = bank.observe_step(slot_ids, src, tgt)
            self.assertIsNotNone(loss_bank)
            self.assertAlmostEqual(float(loss_gold), float(loss_bank), places=6)

        self.assertAlmostEqual(float(bank.flush()), 0.0, places=8)

        gold_w1, gold_w2 = gold.export_weights()
        bank_w1, bank_w2 = bank.export_weights()
        self.assertLess(float((gold_w1 - bank_w1).abs().max().item()), 1e-6)
        self.assertLess(float((gold_w2 - bank_w2).abs().max().item()), 1e-6)

        g_loss, g_cnt, g_rows = gold.read_stats()
        b_loss, b_cnt, b_rows = bank.read_stats()
        self.assertEqual(g_cnt, b_cnt)
        self.assertEqual(g_rows, b_rows)
        self.assertAlmostEqual(g_loss, b_loss, places=6)

    def test_async_bank_matches_grouped_gold(self) -> None:
        device = torch.device("cpu")
        dtype = torch.float32
        lr = 1e-2
        flush_interval = 3
        w1_init, w2_init = build_initial_slot_weights(
            num_slots=6,
            hidden_size=10,
            inner_size=7,
            device=device,
            dtype=dtype,
            seed=55,
        )
        steps = generate_synthetic_sparse_steps(
            steps=10,
            num_slots=6,
            hidden_size=10,
            active_slots_per_step=2,
            rows_per_slot=5,
            device=device,
            dtype=dtype,
            seed=999,
        )

        gold = PerRequestGoldTrainer(w1_init=w1_init, w2_init=w2_init, lr=lr)
        bank = ModelBankAsyncTrainer(
            w1_init=w1_init,
            w2_init=w2_init,
            lr=lr,
            queue_flush_interval=flush_interval,
        )

        q_slots = []
        q_src = []
        q_tgt = []
        for slot_ids, src, tgt in steps:
            q_slots.append(slot_ids)
            q_src.append(src)
            q_tgt.append(tgt)
            loss_bank = bank.observe_step(slot_ids, src, tgt)
            if len(q_slots) < flush_interval:
                self.assertIsNone(loss_bank)
                continue
            slot_cat = torch.cat(q_slots, dim=0)
            src_cat = torch.cat(q_src, dim=0)
            tgt_cat = torch.cat(q_tgt, dim=0)
            loss_gold = gold.train_step(slot_cat, src_cat, tgt_cat)
            self.assertIsNotNone(loss_bank)
            self.assertAlmostEqual(float(loss_gold), float(loss_bank), places=6)
            q_slots = []
            q_src = []
            q_tgt = []

        if q_slots:
            slot_cat = torch.cat(q_slots, dim=0)
            src_cat = torch.cat(q_src, dim=0)
            tgt_cat = torch.cat(q_tgt, dim=0)
            loss_gold = gold.train_step(slot_cat, src_cat, tgt_cat)
            loss_bank = bank.flush()
            self.assertAlmostEqual(float(loss_gold), float(loss_bank), places=6)
        else:
            self.assertAlmostEqual(float(bank.flush()), 0.0, places=8)

        gold_w1, gold_w2 = gold.export_weights()
        bank_w1, bank_w2 = bank.export_weights()
        self.assertLess(float((gold_w1 - bank_w1).abs().max().item()), 1e-6)
        self.assertLess(float((gold_w2 - bank_w2).abs().max().item()), 1e-6)

    def test_sync_loss_decreases_on_correlated_signal(self) -> None:
        device = torch.device("cpu")
        dtype = torch.float32
        hidden_size = 16
        inner_size = 8
        rows_per_step = 16
        steps = 80
        w1_init, w2_init = build_initial_slot_weights(
            num_slots=1,
            hidden_size=hidden_size,
            inner_size=inner_size,
            device=device,
            dtype=dtype,
            seed=3,
        )
        bank = ModelBankAsyncTrainer(w1_init=w1_init, w2_init=w2_init, lr=1e-3, queue_flush_interval=1)
        g = torch.Generator(device=device)
        g.manual_seed(42)
        teacher_w1 = torch.randn((hidden_size, inner_size), generator=g, device=device, dtype=dtype) * 0.2
        teacher_w2 = torch.randn((inner_size, hidden_size), generator=g, device=device, dtype=dtype) * 0.2

        slot_ids = torch.zeros((rows_per_step,), device=device, dtype=torch.long)
        src = torch.randn((rows_per_step, hidden_size), generator=g, device=device, dtype=dtype)
        tgt = torch.matmul(F.gelu(torch.matmul(src, teacher_w1)), teacher_w2)

        def eval_loss(w1: torch.Tensor, w2: torch.Tensor) -> float:
            pred = torch.matmul(F.gelu(torch.matmul(src, w1)), w2)
            return float(F.mse_loss(pred.float(), tgt.float()).item())

        initial = eval_loss(w1_init[0], w2_init[0])
        for _ in range(steps):
            step_loss = bank.observe_step(slot_ids, src, tgt)
            self.assertIsNotNone(step_loss)

        trained_w1, trained_w2 = bank.export_weights()
        final = eval_loss(trained_w1[0], trained_w2[0])
        self.assertLess(final, initial)


if __name__ == "__main__":
    unittest.main()
