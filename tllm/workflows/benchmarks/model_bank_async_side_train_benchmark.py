#!/usr/bin/env python3
"""Independent benchmark for model-bank async side training."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from tllm.experiments.side_train_bank.model_bank_async_side_train import (
    ModelBankAsyncTrainer,
    PerRequestGoldTrainer,
    build_initial_slot_weights,
    generate_synthetic_sparse_steps,
)


@dataclass
class BenchResult:
    elapsed_s: float
    rows: int
    updates: int
    loss_avg: float

    @property
    def rows_per_s(self) -> float:
        return float(self.rows / self.elapsed_s) if self.elapsed_s > 0 else 0.0

    @property
    def updates_per_s(self) -> float:
        return float(self.updates / self.elapsed_s) if self.elapsed_s > 0 else 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num-slots", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--inner-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--active-slots-per-step", type=int, default=16)
    parser.add_argument("--rows-per-slot", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--flush-interval", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    parser.set_defaults(verify=True)
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but --device=cuda was requested")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(dtype_arg: str) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _run_noop(
    *,
    steps: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> BenchResult:
    _sync_if_needed(device)
    start = time.perf_counter()
    rows = 0
    for slot_ids, _, _ in steps:
        rows += int(slot_ids.numel())
    _sync_if_needed(device)
    elapsed = time.perf_counter() - start
    return BenchResult(elapsed_s=float(elapsed), rows=int(rows), updates=0, loss_avg=0.0)


def _run_gold_stepwise(
    *,
    w1_init: torch.Tensor,
    w2_init: torch.Tensor,
    steps: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    lr: float,
    device: torch.device,
) -> Tuple[BenchResult, torch.Tensor, torch.Tensor]:
    trainer = PerRequestGoldTrainer(w1_init=w1_init, w2_init=w2_init, lr=lr)
    _sync_if_needed(device)
    start = time.perf_counter()
    for slot_ids, src, tgt in steps:
        trainer.train_step(slot_ids, src, tgt)
    _sync_if_needed(device)
    elapsed = time.perf_counter() - start
    loss_avg, updates, rows = trainer.read_stats()
    w1, w2 = trainer.export_weights()
    return BenchResult(elapsed_s=float(elapsed), rows=rows, updates=updates, loss_avg=loss_avg), w1, w2


def _run_gold_grouped(
    *,
    w1_init: torch.Tensor,
    w2_init: torch.Tensor,
    steps: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    lr: float,
    flush_interval: int,
    device: torch.device,
) -> Tuple[BenchResult, torch.Tensor, torch.Tensor]:
    trainer = PerRequestGoldTrainer(w1_init=w1_init, w2_init=w2_init, lr=lr)
    flush_interval = max(1, int(flush_interval))
    queue_slots: List[torch.Tensor] = []
    queue_src: List[torch.Tensor] = []
    queue_tgt: List[torch.Tensor] = []
    _sync_if_needed(device)
    start = time.perf_counter()
    for slot_ids, src, tgt in steps:
        queue_slots.append(slot_ids)
        queue_src.append(src)
        queue_tgt.append(tgt)
        if len(queue_slots) < flush_interval:
            continue
        trainer.train_step(
            torch.cat(queue_slots, dim=0),
            torch.cat(queue_src, dim=0),
            torch.cat(queue_tgt, dim=0),
        )
        queue_slots = []
        queue_src = []
        queue_tgt = []
    if queue_slots:
        trainer.train_step(
            torch.cat(queue_slots, dim=0),
            torch.cat(queue_src, dim=0),
            torch.cat(queue_tgt, dim=0),
        )
    _sync_if_needed(device)
    elapsed = time.perf_counter() - start
    loss_avg, updates, rows = trainer.read_stats()
    w1, w2 = trainer.export_weights()
    return BenchResult(elapsed_s=float(elapsed), rows=rows, updates=updates, loss_avg=loss_avg), w1, w2


def _run_bank(
    *,
    w1_init: torch.Tensor,
    w2_init: torch.Tensor,
    steps: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    lr: float,
    flush_interval: int,
    device: torch.device,
) -> Tuple[BenchResult, torch.Tensor, torch.Tensor]:
    trainer = ModelBankAsyncTrainer(
        w1_init=w1_init,
        w2_init=w2_init,
        lr=lr,
        queue_flush_interval=int(flush_interval),
    )
    _sync_if_needed(device)
    start = time.perf_counter()
    for slot_ids, src, tgt in steps:
        trainer.observe_step(slot_ids, src, tgt)
    trainer.flush()
    _sync_if_needed(device)
    elapsed = time.perf_counter() - start
    loss_avg, updates, rows = trainer.read_stats()
    w1, w2 = trainer.export_weights()
    return BenchResult(elapsed_s=float(elapsed), rows=rows, updates=updates, loss_avg=loss_avg), w1, w2


def _print_result(name: str, r: BenchResult, base_rows_per_s: float) -> None:
    rel = (r.rows_per_s / base_rows_per_s) if base_rows_per_s > 0 else 0.0
    print(
        f"{name:18s} elapsed={r.elapsed_s:8.4f}s rows/s={r.rows_per_s:10.2f} "
        f"updates/s={r.updates_per_s:9.2f} loss_avg={r.loss_avg:.6f} rel_rows/s={rel:.4f}"
    )


def main() -> int:
    args = _parse_args()
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)

    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        print("[warning] float16/bfloat16 on CPU can be slow or unsupported on some platforms")

    print(
        f"[config] device={device} dtype={dtype} slots={args.num_slots} hidden={args.hidden_size} "
        f"inner={args.inner_size} steps={args.steps} active_slots/step={args.active_slots_per_step} "
        f"rows/slot={args.rows_per_slot} lr={args.lr} flush_interval={args.flush_interval}"
    )

    w1_init, w2_init = build_initial_slot_weights(
        num_slots=int(args.num_slots),
        hidden_size=int(args.hidden_size),
        inner_size=int(args.inner_size),
        device=device,
        dtype=dtype,
        seed=int(args.seed),
    )
    steps = generate_synthetic_sparse_steps(
        steps=int(args.steps),
        num_slots=int(args.num_slots),
        hidden_size=int(args.hidden_size),
        active_slots_per_step=int(args.active_slots_per_step),
        rows_per_slot=int(args.rows_per_slot),
        device=device,
        dtype=dtype,
        seed=int(args.seed) + 1,
    )

    noop = _run_noop(steps=steps, device=device)
    gold_step, gold_step_w1, gold_step_w2 = _run_gold_stepwise(
        w1_init=w1_init,
        w2_init=w2_init,
        steps=steps,
        lr=float(args.lr),
        device=device,
    )
    bank_sync, bank_sync_w1, bank_sync_w2 = _run_bank(
        w1_init=w1_init,
        w2_init=w2_init,
        steps=steps,
        lr=float(args.lr),
        flush_interval=1,
        device=device,
    )
    gold_grouped, gold_grouped_w1, gold_grouped_w2 = _run_gold_grouped(
        w1_init=w1_init,
        w2_init=w2_init,
        steps=steps,
        lr=float(args.lr),
        flush_interval=int(args.flush_interval),
        device=device,
    )
    bank_async, bank_async_w1, bank_async_w2 = _run_bank(
        w1_init=w1_init,
        w2_init=w2_init,
        steps=steps,
        lr=float(args.lr),
        flush_interval=int(args.flush_interval),
        device=device,
    )

    base = noop.rows_per_s if noop.rows_per_s > 0 else 1.0
    print("[throughput]")
    _print_result("noop", noop, base)
    _print_result("gold_stepwise", gold_step, base)
    _print_result("bank_sync", bank_sync, base)
    _print_result("gold_grouped", gold_grouped, base)
    _print_result("bank_async", bank_async, base)

    if args.verify:
        sync_w1_diff = float((gold_step_w1.float() - bank_sync_w1.float()).abs().max().item())
        sync_w2_diff = float((gold_step_w2.float() - bank_sync_w2.float()).abs().max().item())
        async_w1_diff = float((gold_grouped_w1.float() - bank_async_w1.float()).abs().max().item())
        async_w2_diff = float((gold_grouped_w2.float() - bank_async_w2.float()).abs().max().item())
        print("[verify]")
        print(f"  sync_vs_gold_max_abs: w1={sync_w1_diff:.6e} w2={sync_w2_diff:.6e}")
        print(f"  async_vs_grouped_gold_max_abs: w1={async_w1_diff:.6e} w2={async_w2_diff:.6e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
