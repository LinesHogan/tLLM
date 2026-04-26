"""Public tLLM API."""

from __future__ import annotations

from tllm.consumers.base import BaseConsumer
from tllm.runtime.residual_runtime import make_llm


def register_consumer(consumer: BaseConsumer) -> None:
    from tllm.runtime import residual_runtime

    residual_runtime.register_dispatch_consumer(consumer)


def clear_consumers() -> None:
    from tllm.runtime import residual_runtime

    residual_runtime.clear_dispatch_consumers()


__all__ = ["clear_consumers", "make_llm", "register_consumer"]
