#!/usr/bin/env python3
"""Common runtime hook helpers for consumer dispatch."""

from __future__ import annotations

from typing import Callable, Optional, Protocol

import torch

from tllm.contracts.port_bundle import PortBundle
from tllm.contracts.hidden_batch import HiddenBatch
from tllm.contracts.runtime_context import RunnerLike, RuntimeContext
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.consumer_compat import dispatch_consumer_event


class RuntimeWithDispatchPlan(Protocol):
    dispatch_plan: DispatchPlan | None


def _is_torch_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None:
        fn = getattr(compiler, "is_compiling", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                pass

    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None:
        fn = getattr(dynamo, "is_compiling", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                pass
    return False


def _runner_uses_cudagraph(runner: RunnerLike) -> bool:
    use_cuda_graph = getattr(runner, "use_cuda_graph", None)
    if use_cuda_graph is True:
        return True

    cfg = getattr(runner, "compilation_config", None)
    if cfg is None:
        return False
    try:
        level = int(getattr(cfg, "level", 0))
    except Exception:
        level = 0
    mode_raw = getattr(cfg, "cudagraph_mode", 0)
    if mode_raw is None:
        mode = 0
    else:
        mode = 0
        value = getattr(mode_raw, "value", mode_raw)
        try:
            mode = int(value)
        except Exception:
            if isinstance(value, tuple):
                try:
                    mode = 1 if any(int(x) != 0 for x in value) else 0
                except Exception:
                    mode = 1
            else:
                name = str(getattr(mode_raw, "name", value)).strip().upper()
                mode = 0 if name in {"", "0", "NONE"} else 1
    return (level > 0) or (mode > 0)


def _resolve_device(runner: RunnerLike) -> torch.device:
    device = getattr(runner, "device", None)
    if isinstance(device, torch.device):
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_runtime_context(*, runner: RunnerLike, event_name: str) -> RuntimeContext:
    device = _resolve_device(runner)
    main_stream = None
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            main_stream = torch.cuda.current_stream(device=device)
        except Exception:
            main_stream = None
    return RuntimeContext(
        runner=runner,
        model=getattr(runner, "model", None),
        device=device,
        main_stream=main_stream,
        is_compiling=_is_torch_compiling(),
        uses_cudagraph=_runner_uses_cudagraph(runner),
        event_name=event_name,
    )


def dispatch_runtime_event(
    *,
    runtime: RuntimeWithDispatchPlan,
    runner: RunnerLike,
    event_name: str,
    phase: Optional[str] = None,
    layer_path: Optional[str] = None,
    capture_enabled: Optional[bool] = None,
    batch: Optional[HiddenBatch] = None,
    batch_factory: Optional[Callable[[], Optional[HiddenBatch]]] = None,
) -> int:
    """Dispatch one runtime event to registered consumers, if any."""
    plan = getattr(runtime, "dispatch_plan", None)
    if plan is None:
        return 0

    if capture_enabled is None:
        capture_enabled = not _runner_uses_cudagraph(runner)

    targets = plan.select(
        event_name=event_name,
        phase=phase,
        layer_path=layer_path,
        capture_enabled=bool(capture_enabled),
    )
    if not targets:
        return 0

    ctx = build_runtime_context(runner=runner, event_name=event_name)
    payload = batch
    if payload is None and batch_factory is not None:
        payload = batch_factory()

    for target in targets:
        dispatch_consumer_event(
            consumer=target.consumer,
            payload=payload,
            event_name=event_name,
            ctx=ctx,
        )
    return len(targets)


def dispatch_port_bundle(
    *,
    runtime: RuntimeWithDispatchPlan,
    runner: RunnerLike,
    bundle: PortBundle,
    window: str,
) -> int:
    """Dispatch one assembled bundle to flow-based consumers."""
    plan = getattr(runtime, "dispatch_plan", None)
    if plan is None:
        return 0

    targets = [target for target in plan.flow_targets() if str(target.flow.window) == str(window)]
    if not targets:
        return 0

    ctx = build_runtime_context(runner=runner, event_name=f"flow:{window}")
    for target in targets:
        target.consumer.consume_bundle(bundle, ctx)
    return len(targets)
