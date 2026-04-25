#!/usr/bin/env python3
"""Shared runtime context envelope for producer-consumer dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch


class CompilationConfigLike(Protocol):
    level: object
    cudagraph_mode: object | None


class RunnerLike(Protocol):
    device: object
    model: torch.nn.Module | None
    use_cuda_graph: object
    compilation_config: CompilationConfigLike | None


@dataclass(frozen=True)
class RuntimeContext:
    runner: RunnerLike | None
    model: torch.nn.Module | None
    device: torch.device
    main_stream: Optional[torch.cuda.Stream]
    is_compiling: bool
    uses_cudagraph: bool
    event_name: str
