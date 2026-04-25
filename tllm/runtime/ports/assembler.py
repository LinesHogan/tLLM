#!/usr/bin/env python3
"""Assemble internal port frames into stable consumer bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from tllm.contracts.port_bundle import BundleKey
from tllm.contracts.port_bundle import PortBundle
from tllm.ports.base import ConsumerFlow, PortRead
from tllm.runtime.ports.frame import PortFrame


@dataclass(frozen=True)
class _Binding:
    name: str
    spec: PortRead


class BundleAssembler:
    def __init__(self, flow: ConsumerFlow) -> None:
        self._flow = flow
        self._bindings = tuple(self._build_bindings(flow))
        self._pending: Dict[object, Dict[str, object]] = {}
        self._pending_keys: Dict[object, BundleKey] = {}

    def _build_bindings(self, flow: ConsumerFlow) -> List[_Binding]:
        bindings: List[_Binding] = []
        for read in flow.reads:
            name = str(read.role).strip() or str(read.kind.value)
            bindings.append(_Binding(name=name, spec=read))
        return bindings

    def _match_binding(self, frame: PortFrame) -> _Binding | None:
        for binding in self._bindings:
            spec = binding.spec
            if spec.kind != frame.kind:
                continue
            if spec.locator != frame.locator:
                continue
            return binding
        return None

    def _project_group_key(self, key: BundleKey) -> object:
        if not self._flow.bundle_key:
            return key
        return tuple(getattr(key, field_name) for field_name in self._flow.bundle_key)

    def _is_aggregating(self) -> bool:
        return bool(self._flow.bundle_key)

    def _merge_entry(self, existing: object | None, payload: object) -> object:
        if not self._is_aggregating():
            return payload
        if existing is None:
            return [payload]
        assert isinstance(existing, list)
        existing.append(payload)
        return existing

    def _finalize_entry(self, value: object) -> object:
        if not self._is_aggregating():
            return value
        assert isinstance(value, list)
        if value and all(isinstance(item, torch.Tensor) for item in value):
            return torch.stack(value, dim=0)
        return list(value)

    def _build_bundle(self, group_key: object, entries: Dict[str, object]) -> PortBundle:
        key = self._pending_keys[group_key]
        finalized = {name: self._finalize_entry(value) for name, value in entries.items()}
        if self._is_aggregating():
            projected_values = tuple(getattr(key, field_name) for field_name in self._flow.bundle_key)
            normalized = {
                "engine_step_id": key.engine_step_id,
                "phase": key.phase,
                "request_id": key.request_id,
                "sample_idx": key.sample_idx,
            }
            for field_name, field_value in zip(self._flow.bundle_key, projected_values):
                normalized[field_name] = field_value
            key = BundleKey(
                engine_step_id=int(normalized["engine_step_id"]),
                phase=str(normalized["phase"]),
                request_id=str(normalized["request_id"]),
                sample_idx=int(normalized["sample_idx"]),
            )
        return PortBundle(key=key, entries=finalized)

    def push(self, frame: PortFrame) -> List[PortBundle]:
        binding = self._match_binding(frame)
        if binding is None:
            return []

        group_key = self._project_group_key(frame.key)
        entries = self._pending.setdefault(group_key, {})
        self._pending_keys.setdefault(group_key, frame.key)
        entries[binding.name] = self._merge_entry(entries.get(binding.name), frame.payload)
        if len(entries) < len(self._bindings):
            return []
        if self._is_aggregating():
            return []

        bundle = self._build_bundle(group_key, entries)
        self._pending.pop(group_key, None)
        self._pending_keys.pop(group_key, None)
        return [bundle]

    def pending_bundle_count(self) -> int:
        return len(self._pending)

    def finalize_pending(self) -> List[PortBundle]:
        bundles: List[PortBundle] = []
        for group_key, entries in list(self._pending.items()):
            if len(entries) < len(self._bindings):
                continue
            bundles.append(self._build_bundle(group_key, entries))
            self._pending.pop(group_key, None)
            self._pending_keys.pop(group_key, None)
        return bundles
