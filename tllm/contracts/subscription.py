#!/usr/bin/env python3
"""Consumer subscription contract for runtime event dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

CapturePolicy = Literal["required", "prefer", "never"]
DispatchMode = Literal["inline", "consumer_async"]
Phase = Literal["decode", "prefill"]


@dataclass(frozen=True)
class ConsumerSubscription:
    consumer_id: str
    event_name: str
    phase_filter: Optional[Phase] = None
    layer_filter: Optional[str] = None
    capture_policy: CapturePolicy = "prefer"
    dispatch_mode: DispatchMode = "inline"

    def __post_init__(self) -> None:
        if not self.consumer_id.strip():
            raise ValueError("consumer_id must be non-empty")
        if not self.event_name.strip():
            raise ValueError("event_name must be non-empty")
        if self.capture_policy not in {"required", "prefer", "never"}:
            raise ValueError(f"invalid capture_policy: {self.capture_policy}")
        if self.dispatch_mode not in {"inline", "consumer_async"}:
            raise ValueError(f"invalid dispatch_mode: {self.dispatch_mode}")
