#!/usr/bin/env python3
"""Independent experiment: async model-bank side training.

This module is intentionally decoupled from the main runtime patch path.
It provides:
- gold per-request trainer (one optimizer per request slot)
- model-bank trainer (batched updates over active slots)
- optional async queue accumulation over multiple decode steps
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class TrainReport:
    loss_avg: float
    update_count: int
    elapsed_s: float
    total_rows: int


def build_initial_slot_weights(
    *,
    num_slots: int,
    hidden_size: int,
    inner_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    # Small init keeps training stable for numeric regression tests.
    w1 = torch.randn((num_slots, hidden_size, inner_size), device=device, dtype=dtype, generator=g) * 0.02
    w2 = torch.randn((num_slots, inner_size, hidden_size), device=device, dtype=dtype, generator=g) * 0.02
    return w1, w2


class PerRequestGoldTrainer:
    """Reference trainer: one independent optimizer per request slot."""

    def __init__(
        self,
        *,
        w1_init: torch.Tensor,
        w2_init: torch.Tensor,
        lr: float,
    ) -> None:
        if w1_init.ndim != 3 or w2_init.ndim != 3:
            raise RuntimeError("w1_init/w2_init must be rank-3 tensors")
        if w1_init.shape[0] != w2_init.shape[0]:
            raise RuntimeError("slot count mismatch")
        if w1_init.shape[2] != w2_init.shape[1] or w1_init.shape[1] != w2_init.shape[2]:
            raise RuntimeError("hidden/inner shape mismatch between w1/w2")

        self.num_slots = int(w1_init.shape[0])
        self.hidden_size = int(w1_init.shape[1])
        self.inner_size = int(w1_init.shape[2])
        self.device = w1_init.device
        self.dtype = w1_init.dtype
        self.lr = float(lr)

        self.w1: Dict[int, torch.nn.Parameter] = {}
        self.w2: Dict[int, torch.nn.Parameter] = {}
        self.opt: Dict[int, torch.optim.Optimizer] = {}
        for slot in range(self.num_slots):
            p1 = torch.nn.Parameter(w1_init[slot].detach().clone())
            p2 = torch.nn.Parameter(w2_init[slot].detach().clone())
            self.w1[slot] = p1
            self.w2[slot] = p2
            self.opt[slot] = torch.optim.SGD([p1, p2], lr=self.lr)

        self.loss_sum = 0.0
        self.loss_count = 0
        self.total_rows = 0

    def train_step(self, slot_ids: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor) -> float:
        if slot_ids.numel() == 0:
            return 0.0
        if src.shape != tgt.shape:
            raise RuntimeError("src/tgt shape mismatch")
        if src.ndim != 2:
            raise RuntimeError("src/tgt must be [rows, hidden]")

        unique_slots = torch.unique(slot_ids, sorted=True)
        step_loss_sum = 0.0
        updates = 0
        self.total_rows += int(src.shape[0])

        for slot_t in unique_slots:
            slot = int(slot_t.item())
            mask = slot_ids == slot_t
            if not bool(mask.any()):
                continue
            x = src[mask]
            y = tgt[mask]

            w1 = self.w1[slot]
            w2 = self.w2[slot]
            opt = self.opt[slot]

            pred = torch.matmul(x, w1)
            pred = F.gelu(pred)
            pred = torch.matmul(pred, w2)
            loss = F.mse_loss(pred.float(), y.float())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step_loss_sum += float(loss.detach().item())
            updates += 1

        if updates > 0:
            self.loss_sum += step_loss_sum
            self.loss_count += updates
        return float(step_loss_sum / max(1, updates))

    def export_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        w1 = torch.stack([self.w1[i].detach().clone() for i in range(self.num_slots)], dim=0)
        w2 = torch.stack([self.w2[i].detach().clone() for i in range(self.num_slots)], dim=0)
        return w1, w2

    def reset_stats(self) -> None:
        self.loss_sum = 0.0
        self.loss_count = 0
        self.total_rows = 0

    def read_stats(self) -> Tuple[float, int, int]:
        return (
            float(self.loss_sum / max(1, self.loss_count)),
            int(self.loss_count),
            int(self.total_rows),
        )


class ModelBankAsyncTrainer:
    """Model-bank trainer with optional async queue accumulation."""

    def __init__(
        self,
        *,
        w1_init: torch.Tensor,
        w2_init: torch.Tensor,
        lr: float,
        queue_flush_interval: int = 1,
    ) -> None:
        if w1_init.ndim != 3 or w2_init.ndim != 3:
            raise RuntimeError("w1_init/w2_init must be rank-3 tensors")
        if w1_init.shape[0] != w2_init.shape[0]:
            raise RuntimeError("slot count mismatch")
        if w1_init.shape[2] != w2_init.shape[1] or w1_init.shape[1] != w2_init.shape[2]:
            raise RuntimeError("hidden/inner shape mismatch between w1/w2")

        self.num_slots = int(w1_init.shape[0])
        self.hidden_size = int(w1_init.shape[1])
        self.inner_size = int(w1_init.shape[2])
        self.device = w1_init.device
        self.dtype = w1_init.dtype
        self.lr = float(lr)

        self.w1 = torch.nn.Parameter(w1_init.detach().clone())
        self.w2 = torch.nn.Parameter(w2_init.detach().clone())
        # SGD is used here so model-bank can match the gold reference exactly.
        self.opt = torch.optim.SGD([self.w1, self.w2], lr=self.lr)

        self.queue_flush_interval = max(1, int(queue_flush_interval))
        self._queue_slots: List[torch.Tensor] = []
        self._queue_src: List[torch.Tensor] = []
        self._queue_tgt: List[torch.Tensor] = []
        self._steps_since_flush = 0

        self.loss_sum = 0.0
        self.loss_count = 0
        self.total_rows = 0

    def _train_batch(self, slot_ids: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor) -> float:
        if slot_ids.numel() == 0:
            return 0.0
        if src.shape != tgt.shape:
            raise RuntimeError("src/tgt shape mismatch")
        if src.ndim != 2:
            raise RuntimeError("src/tgt must be [rows, hidden]")

        # Gather per-row model weights from bank.
        w1_rows = self.w1.index_select(0, slot_ids)  # [rows, hidden, inner]
        w2_rows = self.w2.index_select(0, slot_ids)  # [rows, inner, hidden]

        hidden = torch.bmm(src.unsqueeze(1), w1_rows).squeeze(1)
        hidden = F.gelu(hidden)
        pred = torch.bmm(hidden.unsqueeze(1), w2_rows).squeeze(1)

        # Match gold update semantics: each slot contributes its own mean loss.
        row_mse = ((pred.float() - tgt.float()) ** 2).mean(dim=1)
        unique_slots = torch.unique(slot_ids, sorted=True)
        slot_losses: List[torch.Tensor] = []
        for slot_t in unique_slots:
            mask = slot_ids == slot_t
            slot_losses.append(row_mse[mask].mean())
        loss = torch.stack(slot_losses, dim=0).sum()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        step_loss = float(loss.detach().item() / max(1, len(slot_losses)))
        self.loss_sum += step_loss * float(len(slot_losses))
        self.loss_count += int(len(slot_losses))
        self.total_rows += int(src.shape[0])
        return step_loss

    def observe_step(self, slot_ids: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor) -> Optional[float]:
        self._queue_slots.append(slot_ids.detach())
        self._queue_src.append(src.detach())
        self._queue_tgt.append(tgt.detach())
        self._steps_since_flush += 1
        if self._steps_since_flush >= self.queue_flush_interval:
            return self.flush()
        return None

    def flush(self) -> float:
        if not self._queue_slots:
            self._steps_since_flush = 0
            return 0.0

        slot_ids = torch.cat(self._queue_slots, dim=0)
        src = torch.cat(self._queue_src, dim=0)
        tgt = torch.cat(self._queue_tgt, dim=0)

        self._queue_slots = []
        self._queue_src = []
        self._queue_tgt = []
        self._steps_since_flush = 0
        return self._train_batch(slot_ids, src, tgt)

    def export_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.w1.detach().clone(), self.w2.detach().clone()

    def reset_stats(self) -> None:
        self.loss_sum = 0.0
        self.loss_count = 0
        self.total_rows = 0

    def read_stats(self) -> Tuple[float, int, int]:
        return (
            float(self.loss_sum / max(1, self.loss_count)),
            int(self.loss_count),
            int(self.total_rows),
        )


def generate_synthetic_sparse_steps(
    *,
    steps: int,
    num_slots: int,
    hidden_size: int,
    active_slots_per_step: int,
    rows_per_slot: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Generate synthetic active-only traces (not all requests active each step)."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    out: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    active_slots_per_step = max(1, min(int(active_slots_per_step), int(num_slots)))
    rows_per_slot = max(1, int(rows_per_slot))

    for _ in range(int(steps)):
        perm = torch.randperm(num_slots, generator=g, device=device)
        active_slots = perm[:active_slots_per_step].to(torch.long)
        slot_ids = active_slots.repeat_interleave(rows_per_slot)

        src = torch.randn((slot_ids.numel(), hidden_size), generator=g, device=device, dtype=dtype)
        # keep target correlated to source for learnable signal
        noise = 0.05 * torch.randn((slot_ids.numel(), hidden_size), generator=g, device=device, dtype=dtype)
        tgt = 0.8 * src + noise
        out.append((slot_ids, src, tgt))
    return out


__all__ = [
    "ModelBankAsyncTrainer",
    "PerRequestGoldTrainer",
    "TrainReport",
    "build_initial_slot_weights",
    "generate_synthetic_sparse_steps",
]
