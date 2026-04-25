#!/usr/bin/env python3
"""Optional stream runtime for dummy consumer template."""

from __future__ import annotations

from typing import Callable, Optional

import torch


class DummyStreamRuntime:
    def __init__(self, enable_async: bool) -> None:
        self._stream: Optional[torch.cuda.Stream] = None
        if enable_async and torch.cuda.is_available():
            self._stream = torch.cuda.Stream()

    def run(self, fn: Callable[[], None]) -> None:
        if self._stream is None:
            fn()
            return
        with torch.cuda.stream(self._stream):
            fn()

    def synchronize(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()
