#!/usr/bin/env python3
"""Base consumer interface for hidden producer-consumer framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from tllm.contracts.port_bundle import PortBundle
from tllm.contracts.runtime_context import RuntimeContext
from tllm.ports.base import ConsumerFlow


class BaseConsumer(ABC):
    @property
    @abstractmethod
    def consumer_id(self) -> str:
        raise NotImplementedError

    def flows(self) -> Sequence[ConsumerFlow]:
        """Public consumer declaration surface."""
        return ()

    def consume_bundle(self, bundle: PortBundle, ctx: RuntimeContext) -> None:
        """Public flow-based consumption surface."""
        _ = (bundle, ctx)

    def on_step_end(self, ctx: RuntimeContext) -> None:
        """Optional hook called after the model step is complete."""
        _ = ctx

    def synchronize(self) -> None:
        """Optional flush hook for async/background consumers."""
        return None
