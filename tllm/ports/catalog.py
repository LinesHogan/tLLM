#!/usr/bin/env python3
"""Closed public port catalog."""

from __future__ import annotations

from tllm.ports.base import PortKind

PUBLIC_PORT_KINDS = tuple(PortKind)

__all__ = ["PUBLIC_PORT_KINDS", "PortKind"]
