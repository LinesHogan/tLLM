#!/usr/bin/env python3
"""CPU worker for the dummy hidden demo path."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict

import torch


class DummyCpuWorker:
    def __init__(self) -> None:
        self._queue: Deque[torch.Tensor] = deque()
        self.processed_batches = 0
        self.processed_rows = 0
        self.last_mean_abs = 0.0
        self.last_noise_std = 0.0
        self.last_summary = ""

    def enqueue(self, hidden_cpu: torch.Tensor) -> None:
        self._queue.append(hidden_cpu)

    def drain(self, limit: int = 0, *, noise_std: float = 1e-3, emit_summary: bool = True) -> int:
        drained = 0
        total_rows = 0
        sum_mean_abs = 0.0
        while self._queue and (limit <= 0 or drained < limit):
            x = self._queue.popleft()
            self.processed_batches += 1
            rows = int(x.shape[0]) if x.ndim >= 1 else 0
            self.processed_rows += rows
            total_rows += rows
            mean_abs = float(x.abs().mean().item()) if x.numel() > 0 else 0.0
            sum_mean_abs += mean_abs
            self.last_mean_abs = mean_abs
            self.last_noise_std = float(noise_std)
            # Mutate the staged CPU copy so the GPU producer path stays isolated.
            if x.numel() > 0 and noise_std > 0:
                x = x.clone()
                x.add_(torch.randn_like(x) * float(noise_std))
            drained += 1
        if drained > 0:
            mean_abs = float(sum_mean_abs / drained)
            self.last_mean_abs = mean_abs
            self.last_summary = (
                f"dummy_consumer: batches={drained} rows={total_rows} "
                f"mean_abs={mean_abs:.6f} noise_std={self.last_noise_std:.6f}"
            )
            if emit_summary:
                print(self.last_summary)
        return drained

    def pending(self) -> int:
        return len(self._queue)

    def stats(self) -> Dict[str, float]:
        return {
            "processed_batches": float(self.processed_batches),
            "processed_rows": float(self.processed_rows),
            "last_mean_abs": float(self.last_mean_abs),
            "last_noise_std": float(self.last_noise_std),
            "pending": float(self.pending()),
        }
