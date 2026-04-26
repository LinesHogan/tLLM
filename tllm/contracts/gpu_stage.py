#!/usr/bin/env python3
"""Runtime-owned device tensor lease contracts for advanced consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import torch

from tllm.ports.base import BufferOwnership

LeaseLifetime = Literal["consume_call"]

@dataclass(frozen=True)
class DeviceTensorLease:
    entries: Mapping[str, torch.Tensor]
    active_rows: int
    ownership: BufferOwnership = "runtime_lease"
    lifetime: LeaseLifetime = "consume_call"

    def __post_init__(self) -> None:
        if int(self.active_rows) < 0:
            raise ValueError("DeviceTensorLease active_rows must be non-negative")
        if str(self.ownership) != "runtime_lease":
            raise ValueError("DeviceTensorLease currently requires ownership='runtime_lease'")
        if str(self.lifetime) != "consume_call":
            raise ValueError("DeviceTensorLease currently supports only lifetime='consume_call'")
        for name, tensor in self.entries.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"DeviceTensorLease entry `{name}` must be a torch.Tensor")
            if tensor.ndim < 1:
                raise ValueError(f"DeviceTensorLease entry `{name}` must have a row dimension")
            if int(self.active_rows) > int(tensor.shape[0]):
                raise ValueError(f"DeviceTensorLease active_rows exceeds entry `{name}` row capacity")
