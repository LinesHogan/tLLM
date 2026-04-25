#!/usr/bin/env python3
"""Shared helpers for deciding whether dispatch-plan targets are active."""

from __future__ import annotations

from typing import Protocol


class ActivityPlan(Protocol):
    def has_active_targets(self) -> bool:
        ...


class RuntimeWithTargets(Protocol):
    dispatch_plan: ActivityPlan | None


def runtime_has_active_targets(runtime: RuntimeWithTargets) -> bool:
    plan = runtime.dispatch_plan
    return bool(plan is not None and plan.has_active_targets())
