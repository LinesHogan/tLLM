#!/usr/bin/env python3
"""Consumer registry and dispatch plan owner."""

from __future__ import annotations

from typing import Iterable, List

from tllm.consumers.base import BaseConsumer
from tllm.runtime.dispatch_plan import DispatchPlan


class ConsumerRegistry:
    def __init__(self) -> None:
        self._consumers: List[BaseConsumer] = []

    def register(self, consumer: BaseConsumer) -> None:
        self._consumers.append(consumer)

    def clear(self) -> None:
        self._consumers.clear()

    def consumers(self) -> List[BaseConsumer]:
        return list(self._consumers)

    def replace_all(self, consumers: Iterable[BaseConsumer]) -> None:
        self._consumers = list(consumers)

    def build_dispatch_plan(self) -> DispatchPlan:
        return DispatchPlan.build(self._consumers)
