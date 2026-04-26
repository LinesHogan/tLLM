#!/usr/bin/env python3
"""Generic residual bundle dispatch helpers."""

from __future__ import annotations

import time
from typing import List, Protocol

import torch

from tllm.contracts.gpu_stage import DeviceTensorLease
from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.request_meta_view import RowBatchMeta
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
from tllm.runtime.consumer_compat import on_step_end as call_consumer_step_end
from tllm.contracts.runtime_context import RunnerLike


class PortDispatchRuntimeLike(Protocol):
    decode_count: int
    decode_prompt_idxs: list[int]
    decode_sample_idxs: list[int]
    decode_request_ids: list[str]
    tap_decode_hidden: dict[str, torch.Tensor]
    tap_decode_hidden_compact: dict[str, torch.Tensor]
    decode_compact_count: int
    decode_compact_row_ids: tuple[int, ...]
    residual_bindings: dict[str, ResidualPathBinding]
    event_step_id: int
    dispatch_plan: DispatchPlan | None


class PortDispatchCoreLike(Protocol):
    RUNTIME: PortDispatchRuntimeLike


def _path_hotspots_enabled(core: object) -> bool:
    runtime = getattr(core, "RUNTIME", None)
    return bool(getattr(runtime, "path_hotspot_enabled", False)) and callable(getattr(core, "record_path_hotspot_cpu", None))


def _record_cpu_hotspot(core: object, name: str, start_s: float, end_s: float) -> None:
    if not _path_hotspots_enabled(core):
        return
    getattr(core, "record_path_hotspot_cpu")(name, float((end_s - start_s) * 1000.0))


def _compact_residual_hidden_for_read(
    *,
    core: PortDispatchCoreLike,
    read: PortRead,
    active: int,
    row_ids: tuple[int, ...] | None,
) -> torch.Tensor | None:
    if row_ids is None:
        return None
    if int(getattr(core.RUNTIME, "decode_compact_count", -1)) != int(active):
        return None
    if tuple(getattr(core.RUNTIME, "decode_compact_row_ids", ()) or ()) != tuple(row_ids):
        return None
    locator = read.locator
    if not isinstance(locator, ResidualLocator):
        return None
    resolved_path = _residual_bindings.resolved_path_for_locator(core.RUNTIME.residual_bindings, locator)
    if resolved_path is None:
        return None
    compact_map = getattr(core.RUNTIME, "tap_decode_hidden_compact", None)
    if not isinstance(compact_map, dict):
        return None
    hidden = compact_map.get(resolved_path)
    if not isinstance(hidden, torch.Tensor) or int(hidden.shape[0]) < int(active):
        return None
    return hidden[:active]


def _regular_row_view(hidden: torch.Tensor, row_ids: tuple[int, ...]) -> torch.Tensor | None:
    if not row_ids:
        return hidden[:0]
    if hidden.ndim != 2:
        return None
    if len(row_ids) == 1:
        start = int(row_ids[0])
        if start < 0 or start >= int(hidden.shape[0]):
            return None
        return hidden[start : start + 1]
    step = int(row_ids[1]) - int(row_ids[0])
    if step <= 0:
        return None
    start = int(row_ids[0])
    last = int(row_ids[-1])
    if start < 0 or last >= int(hidden.shape[0]):
        return None
    for left, right in zip(row_ids, row_ids[1:]):
        if int(right) - int(left) != step:
            return None
    return hidden.as_strided(
        (len(row_ids), int(hidden.shape[1])),
        (int(hidden.stride(0)) * step, int(hidden.stride(1))),
        storage_offset=int(hidden.storage_offset()) + start * int(hidden.stride(0)),
    )


def _request_meta_payload(core: PortDispatchCoreLike, active: int) -> list[dict[str, object]]:
    return _request_meta_view(core, active).as_legacy_dicts()


def _request_meta_view(core: PortDispatchCoreLike, active: int) -> RowBatchMeta:
    request_ids, prompt_idxs, sample_idxs = active_request_prompt_sample_metadata(core.RUNTIME, active)
    return RowBatchMeta(
        request_ids=tuple(str(request_ids[row_i]) for row_i in range(active)),
        prompt_idxs=tuple(int(prompt_idxs[row_i]) for row_i in range(active)),
        sample_idxs=tuple(int(sample_idxs[row_i]) for row_i in range(active)),
        phase="decode",
        engine_step_id=int(core.RUNTIME.event_step_id),
        row_ids=tuple(range(active)),
    )


def _compact_first_per_prompt(view: RowBatchMeta) -> tuple[RowBatchMeta, tuple[int, ...]]:
    seen: set[int] = set()
    keep: list[int] = []
    for row_i, prompt_idx in enumerate(view.prompt_idxs):
        prompt_key = int(prompt_idx)
        if prompt_key < 0 or prompt_key in seen:
            continue
        seen.add(prompt_key)
        keep.append(int(row_i))
    row_ids = tuple(keep)
    compact = RowBatchMeta(
        request_ids=tuple(view.request_ids[row_i] for row_i in row_ids),
        prompt_idxs=tuple(view.prompt_idxs[row_i] for row_i in row_ids),
        sample_idxs=tuple(view.sample_idxs[row_i] for row_i in row_ids),
        phase=view.phase,
        engine_step_id=int(view.engine_step_id),
        row_compaction="first_per_prompt",
        row_ids=tuple(view.row_ids[row_i] for row_i in row_ids) if view.row_ids else row_ids,
    )
    return compact, row_ids


def _compact_first_per_prompt_from_runtime(core: PortDispatchCoreLike, active: int) -> tuple[RowBatchMeta, tuple[int, ...]]:
    runtime = core.RUNTIME
    request_ids = runtime.decode_request_ids
    prompt_idxs = runtime.decode_prompt_idxs
    sample_idxs = runtime.decode_sample_idxs
    if len(request_ids) < active or len(prompt_idxs) < active or len(sample_idxs) < active:
        raise RuntimeError(
            "decode runtime metadata is inconsistent: "
            f"active={active} request_ids={len(request_ids)} prompt_idxs={len(prompt_idxs)} sample_idxs={len(sample_idxs)}"
        )
    seen: set[int] = set()
    keep: list[int] = []
    compact_request_ids: list[str] = []
    compact_prompt_idxs: list[int] = []
    compact_sample_idxs: list[int] = []
    for row_i in range(active):
        prompt_key = int(prompt_idxs[row_i])
        if prompt_key < 0 or prompt_key in seen:
            continue
        seen.add(prompt_key)
        keep.append(int(row_i))
        compact_request_ids.append(str(request_ids[row_i]))
        compact_prompt_idxs.append(prompt_key)
        compact_sample_idxs.append(int(sample_idxs[row_i]))
    row_ids = tuple(keep)
    compact = RowBatchMeta(
        request_ids=tuple(compact_request_ids),
        prompt_idxs=tuple(compact_prompt_idxs),
        sample_idxs=tuple(compact_sample_idxs),
        phase="decode",
        engine_step_id=int(runtime.event_step_id),
        row_compaction="first_per_prompt",
        row_ids=row_ids,
    )
    return compact, row_ids


def _request_meta_payload_from_view(view: RowBatchMeta) -> list[dict[str, object]]:
    return [
        {
            "request_id": view.request_ids[row_i],
            "prompt_idx": view.prompt_idxs[row_i],
            "sample_idx": view.sample_idxs[row_i],
            "phase": view.phase,
            "engine_step_id": int(view.engine_step_id),
        }
        for row_i in range(len(view.request_ids))
    ]


def _entry_for_flow_read(
    *,
    core: PortDispatchCoreLike,
    read: PortRead,
    request_meta_payload: list[dict[str, object]],
    active: int,
    row_ids: tuple[int, ...] | None = None,
) -> tuple[str, object] | None:
    name = str(read.role).strip() or str(read.kind.value)
    if read.kind is PortKind.REQUEST_META:
        return name, request_meta_payload
    if read.kind is not PortKind.RESIDUAL_STREAM:
        return None
    compact_hidden = _compact_residual_hidden_for_read(core=core, read=read, active=active, row_ids=row_ids)
    if compact_hidden is not None:
        return name, compact_hidden
    locator = read.locator
    if not isinstance(locator, ResidualLocator):
        return None
    resolved_path = _residual_bindings.resolved_path_for_locator(core.RUNTIME.residual_bindings, locator)
    if resolved_path is None:
        return None
    hidden = core.RUNTIME.tap_decode_hidden.get(resolved_path)
    if hidden is None or not isinstance(hidden, torch.Tensor):
        return None
    if row_ids is None:
        return name, hidden[:active]
    if not row_ids:
        return name, hidden[:0]
    if row_ids == tuple(range(len(row_ids))):
        return name, hidden[: len(row_ids)]
    view = _regular_row_view(hidden, row_ids)
    if view is not None:
        return name, view
    row_tensor = torch.as_tensor(row_ids, device=hidden.device, dtype=torch.long)
    return name, hidden.index_select(0, row_tensor)


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
    if active <= 0:
        return None

    entries: dict[str, object] = {}
    row_ids: tuple[int, ...] | None = None
    row_compaction = str(getattr(flow, "row_compaction", "none"))
    if row_compaction == "first_per_prompt":
        request_meta_view, row_ids = _compact_first_per_prompt_from_runtime(core, active)
        active = len(row_ids)
        if active <= 0:
            return None
    elif row_compaction != "none":
        raise RuntimeError(f"unsupported row_compaction mode `{row_compaction}`")
    else:
        request_meta_view = _request_meta_view(core, active)

    row_cap = int(getattr(flow, "max_bundle_rows", 0) or 0)
    if row_cap > 0 and active > row_cap:
        raise RuntimeError(
            f"flow row cap exceeded: active_rows={active} max_bundle_rows={row_cap}. "
            "Increase the flow capacity or reduce the workload; tLLM does not silently drop residual rows."
        )

    delivery = str(getattr(flow, "delivery", "bundle"))
    if delivery == "device_lease":
        lease_entries: dict[str, torch.Tensor] = {}
        for read in flow.reads:
            entry = _entry_for_flow_read(
                core=core,
                read=read,
                request_meta_payload=[],
                active=active,
                row_ids=row_ids,
            )
            if entry is None:
                name = str(read.role).strip() or str(read.kind.value)
                raise RuntimeError(f"active flow bundle missing required entry `{name}`")
            name, payload = entry
            if read.kind is PortKind.REQUEST_META:
                entries[name] = request_meta_view
                continue
            if read.kind is not PortKind.RESIDUAL_STREAM or not isinstance(payload, torch.Tensor):
                raise RuntimeError("device_lease delivery currently supports residual_stream and request_meta reads")
            lease_entries[name] = payload
        lease = DeviceTensorLease(
            entries=lease_entries,
            active_rows=int(active),
            ownership=str(getattr(flow, "ownership", "runtime_lease")),
        )
        entries["device_lease"] = lease
        request_id = request_meta_view.request_ids[0] if request_meta_view.request_ids else ""
        sample_idx = int(request_meta_view.sample_idxs[0]) if request_meta_view.sample_idxs else 0
    else:
        request_meta_payload = _request_meta_payload_from_view(request_meta_view)
        for read in flow.reads:
            entry = _entry_for_flow_read(
                core=core,
                read=read,
                request_meta_payload=request_meta_payload,
                active=active,
                row_ids=row_ids,
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
    trace_hotspots = _path_hotspots_enabled(core)
    t_ctx0 = time.perf_counter() if trace_hotspots else 0.0
    ctx = _common_hooks.build_runtime_context(runner=runner, event_name="flow:decode")
    t_ctx1 = time.perf_counter() if trace_hotspots else 0.0
    _record_cpu_hotspot(core, "dispatch_bundles.context_cpu", t_ctx0, t_ctx1)
    frame_targets: List[FlowDispatchTarget] = []
    for target in targets:
        t_build0 = time.perf_counter() if trace_hotspots else 0.0
        direct_bundle = build_step_scope_port_bundle(core=core, flow=target.flow)
        t_build1 = time.perf_counter() if trace_hotspots else 0.0
        _record_cpu_hotspot(core, "dispatch_bundles.step_bundle_cpu", t_build0, t_build1)
        if direct_bundle is not None:
            t_consume0 = time.perf_counter() if trace_hotspots else 0.0
            target.consumer.consume_bundle(direct_bundle, ctx)
            t_consume1 = time.perf_counter() if trace_hotspots else 0.0
            _record_cpu_hotspot(core, "dispatch_bundles.consume_bundle_cpu", t_consume0, t_consume1)
            dispatched += 1
            if str(target.flow.window) != "background":
                t_feedback0 = time.perf_counter() if trace_hotspots else 0.0
                call_consumer_step_end(target.consumer, ctx)
                t_feedback1 = time.perf_counter() if trace_hotspots else 0.0
                _record_cpu_hotspot(core, "dispatch_bundles.feedback_cpu", t_feedback0, t_feedback1)
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
            call_consumer_step_end(target.consumer, ctx)
    return dispatched
