#!/usr/bin/env python3
"""Runtime-internal compatibility helpers for pre-port consumers."""

from __future__ import annotations

from typing import Protocol, Sequence

from tllm.contracts.hidden_batch import HiddenBatch
from tllm.contracts.runtime_context import RuntimeContext
from tllm.contracts.subscription import ConsumerSubscription


class SupportsLegacySubscriptions(Protocol):
    def subscriptions(self) -> Sequence[ConsumerSubscription]:
        ...


class SupportsLegacyConsume(Protocol):
    def consume(self, batch: HiddenBatch, ctx: RuntimeContext) -> None:
        ...


class SupportsLegacyTick(Protocol):
    def on_tick(self, event_name: str, ctx: RuntimeContext) -> None:
        ...


class SupportsFeedback(Protocol):
    def apply_feedback(self, ctx: RuntimeContext) -> None:
        ...


class SupportsSynchronize(Protocol):
    def synchronize(self) -> None:
        ...


def legacy_subscriptions(consumer: SupportsLegacySubscriptions | object) -> Sequence[ConsumerSubscription]:
    fn = getattr(consumer, "subscriptions", None)
    if not callable(fn):
        return ()
    subs = fn()
    return tuple(subs) if subs is not None else ()


def apply_feedback(consumer: SupportsFeedback | object, ctx: RuntimeContext) -> None:
    fn = getattr(consumer, "apply_feedback", None)
    if callable(fn):
        fn(ctx)


def dispatch_legacy_event(
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
        apply_feedback(consumer, ctx)
def synchronize(consumer: SupportsSynchronize | object) -> None:
    fn = getattr(consumer, "synchronize", None)
    if callable(fn):
        fn()
