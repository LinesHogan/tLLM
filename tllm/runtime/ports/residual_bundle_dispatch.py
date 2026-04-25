#!/usr/bin/env python3
"""Generic residual bundle dispatch helpers."""

from __future__ import annotations

from typing import List, Protocol

import torch

from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.ports.base import ConsumerFlow, PortKind
from tllm.ports.request_meta import RequestMeta
from tllm.ports.residual_stream import ResidualLocator, ResidualStream
from tllm.ports.base import PortRead
from tllm.runtime.dispatch_plan import DispatchPlan, FlowDispatchTarget
from tllm.runtime.ports.frame import Ownership, PortFrame
from tllm.runtime.ports.assembler import BundleAssembler
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.decode_runtime_metadata import active_request_prompt_sample_metadata
from tllm.runtime.ports import residual_bindings as _residual_bindings
from tllm.runtime.vllm_patch import common_hooks as _common_hooks
from tllm.runtime.legacy_consumer_compat import apply_feedback as apply_consumer_feedback
from tllm.contracts.runtime_context import RunnerLike


class PortDispatchRuntimeLike(Protocol):
    decode_count: int
    decode_prompt_idxs: list[int]
    decode_sample_idxs: list[int]
    decode_request_ids: list[str]
    tap_decode_hidden: dict[str, torch.Tensor]
    residual_bindings: dict[str, ResidualPathBinding]
    event_step_id: int
    dispatch_plan: DispatchPlan | None


class PortDispatchCoreLike(Protocol):
    RUNTIME: PortDispatchRuntimeLike


def _request_meta_payload(core: PortDispatchCoreLike, active: int) -> list[dict[str, object]]:
    request_ids, prompt_idxs, sample_idxs = active_request_prompt_sample_metadata(core.RUNTIME, active)
    return [
        {
            "request_id": request_ids[row_i],
            "prompt_idx": prompt_idxs[row_i],
            "sample_idx": sample_idxs[row_i],
            "phase": "decode",
            "engine_step_id": int(core.RUNTIME.event_step_id),
        }
        for row_i in range(active)
    ]


def _entry_for_flow_read(
    *,
    core: PortDispatchCoreLike,
    read: PortRead,
    request_meta_payload: list[dict[str, object]],
    active: int,
) -> tuple[str, object] | None:
    name = str(read.role).strip() or str(read.kind.value)
    if read.kind is PortKind.REQUEST_META:
        return name, request_meta_payload
    if read.kind is not PortKind.RESIDUAL_STREAM:
        return None
    locator = read.locator
    if not isinstance(locator, ResidualLocator):
        return None
    resolved_path = _residual_bindings.resolved_path_for_locator(core.RUNTIME.residual_bindings, locator)
    if resolved_path is None:
        return None
    hidden = core.RUNTIME.tap_decode_hidden.get(resolved_path)
    if hidden is None or not isinstance(hidden, torch.Tensor):
        return None
    return name, hidden[:active]


def build_decode_port_frames(*, core: PortDispatchCoreLike, layer_path: str) -> List[PortFrame]:
    decode_buf = core.RUNTIME.tap_decode_hidden.get(str(layer_path))
    if not isinstance(decode_buf, torch.Tensor):
        return []

    active = int(core.RUNTIME.decode_count)
    if active <= 0:
        return []

    binding = core.RUNTIME.residual_bindings.get(str(layer_path))
    if binding is None:
        return []
    locator = binding.locator
    request_ids, prompt_idxs, sample_idxs = active_request_prompt_sample_metadata(core.RUNTIME, active)
    frames: List[PortFrame] = []
    request_meta_locator = RequestMeta.read().locator
    include_request_meta = bool(binding.include_request_meta)

    for row_i in range(active):
        prompt_idx = prompt_idxs[row_i]
        sample_idx = sample_idxs[row_i]
        request_id = request_ids[row_i]
        key = BundleKey(
            engine_step_id=int(core.RUNTIME.event_step_id),
            phase="decode",
            request_id=request_id,
            sample_idx=sample_idx,
        )
        frames.append(
            PortFrame(
                key=key,
                kind=ResidualStream.KIND,
                locator=locator,
                payload=decode_buf[row_i],
                ownership=Ownership.BORROWED,
                ready_window="same_step",
            )
        )
        if include_request_meta:
            frames.append(
                PortFrame(
                    key=key,
                    kind=RequestMeta.KIND,
                    locator=request_meta_locator,
                    payload={
                        "request_id": request_id,
                        "prompt_idx": prompt_idx,
                        "sample_idx": sample_idx,
                        "phase": "decode",
                        "engine_step_id": int(core.RUNTIME.event_step_id),
                    },
                    ownership=Ownership.STAGED,
                    ready_window="same_step",
                )
            )
    return frames


def build_step_scope_port_bundle(*, core: PortDispatchCoreLike, flow: ConsumerFlow) -> PortBundle | None:
    if tuple(flow.bundle_key) != ("engine_step_id", "phase"):
        return None
    active = int(core.RUNTIME.decode_count)
    row_cap = int(getattr(flow, "max_bundle_rows", 0) or 0)
    if row_cap > 0:
        active = min(active, row_cap)
    if active <= 0:
        return None

    entries: dict[str, object] = {}
    request_meta_payload = _request_meta_payload(core, active)

    for read in flow.reads:
        entry = _entry_for_flow_read(
            core=core,
            read=read,
            request_meta_payload=request_meta_payload,
            active=active,
        )
        if entry is None:
            name = str(read.role).strip() or str(read.kind.value)
            raise RuntimeError(f"active flow bundle missing required entry `{name}`")
        name, payload = entry
        entries[name] = payload

    request_id = request_meta_payload[0]["request_id"] if request_meta_payload else ""
    sample_idx = int(request_meta_payload[0]["sample_idx"]) if request_meta_payload else 0
    return PortBundle(
        key=BundleKey(
            engine_step_id=int(core.RUNTIME.event_step_id),
            phase="decode",
            request_id=str(request_id),
            sample_idx=sample_idx,
        ),
        entries=entries,
    )


def _flow_due_for_step(*, flow: ConsumerFlow, step_id: int) -> bool:
    stride = max(1, int(getattr(flow, "dispatch_every_n_steps", 1)))
    return (int(step_id) % stride) == 0


def dispatch_decode_port_bundles(*, core: PortDispatchCoreLike, runner: RunnerLike) -> int:
    plan = getattr(core.RUNTIME, "dispatch_plan", None)
    if plan is None:
        return 0

    step_id = int(core.RUNTIME.event_step_id)
    targets = [target for target in plan.flow_targets() if _flow_due_for_step(flow=target.flow, step_id=step_id)]
    if not targets:
        return 0

    dispatched = 0
    ctx = _common_hooks.build_runtime_context(runner=runner, event_name="flow:decode")
    frame_targets: List[FlowDispatchTarget] = []
    for target in targets:
        direct_bundle = build_step_scope_port_bundle(core=core, flow=target.flow)
        if direct_bundle is not None:
            target.consumer.consume_bundle(direct_bundle, ctx)
            dispatched += 1
            if str(target.flow.window) != "background":
                apply_consumer_feedback(target.consumer, ctx)
        else:
            frame_targets.append(target)

    if not frame_targets:
        return dispatched

    frames: List[PortFrame] = []
    for layer_path in _residual_bindings.tap_paths(core.RUNTIME.residual_bindings):
        layer_key = str(layer_path).strip()
        if not layer_key:
            continue
        frames.extend(build_decode_port_frames(core=core, layer_path=layer_key))
    if not frames:
        return dispatched

    for target in frame_targets:
        assembler = BundleAssembler(target.flow)
        for frame in frames:
            bundles = assembler.push(frame)
            for bundle in bundles:
                target.consumer.consume_bundle(bundle, ctx)
                dispatched += 1
        for bundle in assembler.finalize_pending():
            target.consumer.consume_bundle(bundle, ctx)
            dispatched += 1
        if str(target.flow.window) != "background":
            apply_consumer_feedback(target.consumer, ctx)
    return dispatched
