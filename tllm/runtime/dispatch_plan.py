#!/usr/bin/env python3
"""Static dispatch plan for runtime event -> consumer routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from tllm.consumers.base import BaseConsumer
from tllm.ports.base import ConsumerFlow
from tllm.ports.base import PortKind
from tllm.contracts.subscription import ConsumerSubscription
from tllm.runtime.consumer_compat import consumer_subscriptions


@dataclass(frozen=True)
class DispatchTarget:
    consumer: BaseConsumer
    subscription: ConsumerSubscription


@dataclass(frozen=True)
class FlowDispatchTarget:
    consumer: BaseConsumer
    flow: ConsumerFlow


class DispatchPlan:
    def __init__(self, by_event: Dict[str, List[DispatchTarget]], flow_targets: List[FlowDispatchTarget]) -> None:
        self._by_event = by_event
        self._flow_targets = flow_targets

    @classmethod
    def build(cls, consumers: Iterable[BaseConsumer]) -> "DispatchPlan":
        by_event: Dict[str, List[DispatchTarget]] = {}
        flow_targets: List[FlowDispatchTarget] = []
        for consumer in consumers:
            flows = list(consumer.flows())
            for flow in flows:
                flow_targets.append(FlowDispatchTarget(consumer=consumer, flow=flow))
            if flows:
                continue
            for sub in consumer_subscriptions(consumer):
                by_event.setdefault(sub.event_name, []).append(DispatchTarget(consumer=consumer, subscription=sub))
        return cls(by_event=by_event, flow_targets=flow_targets)

    def select(
        self,
        *,
        event_name: str,
        phase: Optional[str],
        layer_path: Optional[str],
        capture_enabled: bool,
    ) -> List[DispatchTarget]:
        targets = self._by_event.get(event_name, [])
        out: List[DispatchTarget] = []
        for target in targets:
            sub = target.subscription
            if sub.phase_filter is not None and phase is not None and sub.phase_filter != phase:
                continue
            if sub.layer_filter is not None and layer_path is not None and sub.layer_filter != layer_path:
                continue
            if sub.capture_policy == "required" and (not capture_enabled):
                continue
            if sub.capture_policy == "never" and capture_enabled:
                continue
            out.append(target)
        return out

    def flow_targets(self) -> List[FlowDispatchTarget]:
        return list(self._flow_targets)

    def has_active_targets(self) -> bool:
        return not self.is_empty()

    def is_empty(self) -> bool:
        return (not self._flow_targets) and all((not targets) for targets in self._by_event.values())

    def required_residual_layers(self) -> set[tuple[int, str, str]]:
        required: set[tuple[int, str, str]] = set()
        for target in self._flow_targets:
            for read in target.flow.reads:
                if read.kind != PortKind.RESIDUAL_STREAM or read.locator is None:
                    continue
                layer = int(getattr(read.locator, "layer"))
                site = str(getattr(read.locator, "site"))
                phase = str(getattr(read.locator, "phase"))
                required.add((layer, site, phase))
        return required

    def max_residual_bundle_rows(self) -> int:
        caps: list[int] = []
        has_uncapped_residual_flow = False
        for target in self._flow_targets:
            has_residual_read = any(read.kind == PortKind.RESIDUAL_STREAM for read in target.flow.reads)
            if not has_residual_read:
                continue
            cap = int(getattr(target.flow, "max_bundle_rows", 0) or 0)
            if cap <= 0:
                has_uncapped_residual_flow = True
            else:
                caps.append(cap)
        if has_uncapped_residual_flow or not caps:
            return 0
        return max(1, max(caps))
