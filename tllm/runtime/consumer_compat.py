#!/usr/bin/env python3
"""Runtime-internal compatibility helpers for event-style consumers."""

from __future__ import annotations

from typing import Protocol, Sequence

from tllm.contracts.hidden_batch import HiddenBatch
from tllm.contracts.runtime_context import RuntimeContext
from tllm.contracts.subscription import ConsumerSubscription


class SupportsSubscriptions(Protocol):
    def subscriptions(self) -> Sequence[ConsumerSubscription]:
        ...


class SupportsConsume(Protocol):
    def consume(self, batch: HiddenBatch, ctx: RuntimeContext) -> None:
        ...


class SupportsTick(Protocol):
    def on_tick(self, event_name: str, ctx: RuntimeContext) -> None:
        ...


class SupportsStepEnd(Protocol):
    def on_step_end(self, ctx: RuntimeContext) -> None:
        ...


class SupportsSynchronize(Protocol):
    def synchronize(self) -> None:
        ...


def consumer_subscriptions(consumer: SupportsSubscriptions | object) -> Sequence[ConsumerSubscription]:
    fn = getattr(consumer, "subscriptions", None)
    if not callable(fn):
        return ()
    subs = fn()
    return tuple(subs) if subs is not None else ()


def on_step_end(consumer: SupportsStepEnd | object, ctx: RuntimeContext) -> None:
    fn = getattr(consumer, "on_step_end", None)
    if callable(fn):
        fn(ctx)


def dispatch_consumer_event(
    *,
    consumer: object,
    payload: HiddenBatch | None,
    event_name: str,
    ctx: RuntimeContext,
) -> None:
    consume_fn = getattr(consumer, "consume", None)
    if payload is not None and callable(consume_fn):
        consume_fn(payload, ctx)

    tick_fn = getattr(consumer, "on_tick", None)
    if callable(tick_fn):
        tick_fn(event_name, ctx)

    if event_name == "execute_model.post":
        on_step_end(consumer, ctx)


def synchronize(consumer: SupportsSynchronize | object) -> None:
    fn = getattr(consumer, "synchronize", None)
    if callable(fn):
        fn()
