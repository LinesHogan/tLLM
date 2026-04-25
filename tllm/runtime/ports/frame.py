#!/usr/bin/env python3
"""Internal runtime port frame records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from tllm.contracts.port_bundle import BundleKey
from tllm.ports.base import Locator, PortKind, Window


class Ownership(str, Enum):
    BORROWED = "borrowed"
    STAGED = "staged"


@dataclass(frozen=True)
class PortFrame:
    key: BundleKey
    kind: PortKind
    locator: Locator | None
    payload: Any
    ownership: Ownership
    ready_window: Window
