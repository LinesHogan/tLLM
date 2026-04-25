#!/usr/bin/env python3
"""Configuration for the dummy hidden demo consumer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DummyConsumerConfig:
    consumer_id: str = "dummy"
    enable_async: bool = True
    export_to_cpu: bool = True
    max_queue_size: int = 4096
    noise_std: float = 1e-3
    feedback_interval: int = 0
    export_every_n_steps: int = 256
    export_max_rows: int = 1
    export_max_cols: int = 16
    layer_filter: Optional[str] = None
    capture_policy: str = "prefer"
