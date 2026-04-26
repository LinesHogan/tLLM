#!/usr/bin/env python3
"""Async CPU-export demo consumer for the extension template."""

from __future__ import annotations

from typing import Dict, Sequence

from tllm.consumers.base import BaseConsumer
from tllm.consumers.dummy.config import DummyConsumerConfig
from tllm.consumers.dummy.stream_runtime import DummyStreamRuntime
from tllm.consumers.dummy.transfer import clone_hidden_to_cpu
from tllm.consumers.dummy.worker import DummyCpuWorker
from tllm.contracts.port_bundle import PortBundle
from tllm.contracts.runtime_context import RuntimeContext
from tllm.ports.base import ConsumerFlow
from tllm.ports.cpu_export import CpuExport
from tllm.ports.request_meta import RequestMeta
from tllm.ports.residual_stream import ResidualStream


class DummyConsumer(BaseConsumer):
    """Minimal consumer that stages hidden rows to CPU without hot-path drain."""

    def __init__(self, config: DummyConsumerConfig) -> None:
        self.config = config
        self._stream_runtime = DummyStreamRuntime(enable_async=bool(config.enable_async))
        self._worker = DummyCpuWorker()
        self._consumed_batches = 0
        self._consumed_rows = 0
        self._dropped_batches = 0
        self._feedback_calls = 0

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

    def flows(self) -> Sequence[ConsumerFlow]:
        writes = (CpuExport.write(channel="dummy_hidden", format="row_batch"),) if self.config.export_to_cpu else ()
        return [
            ConsumerFlow(
                reads=(
                    ResidualStream.read(layer=0, site="block_output", phase="decode", role="hidden"),
                    RequestMeta.read(),
                ),
                writes=writes,
                window="background",
                bundle_key=("engine_step_id", "phase"),
                dispatch_every_n_steps=max(1, int(self.config.export_every_n_steps)),
                max_bundle_rows=max(0, int(self.config.export_max_rows)),
            )
        ]

    def _handle_hidden_rows(self, rows_hidden) -> None:
        self._consumed_batches += 1
        self._consumed_rows += int(rows_hidden.shape[0]) if rows_hidden.ndim >= 1 else 0
        if not self.config.export_to_cpu:
            return
        export_rows = rows_hidden
        max_rows = int(self.config.export_max_rows)
        max_cols = int(self.config.export_max_cols)
        if max_rows > 0 and export_rows.ndim >= 1:
            export_rows = export_rows[:max_rows]
        if max_cols > 0 and export_rows.ndim >= 2:
            export_rows = export_rows[:, :max_cols]

        if self.config.enable_async:
            if self._worker.pending() >= int(self.config.max_queue_size):
                self._dropped_batches += 1
                return

            def _submit() -> None:
                hidden_cpu = clone_hidden_to_cpu(export_rows)
                self._worker.enqueue(hidden_cpu)

            self._stream_runtime.run(_submit)
            return

        hidden_cpu = clone_hidden_to_cpu(export_rows)
        self._worker.enqueue(hidden_cpu)
        self._worker.drain(limit=1, noise_std=self.config.noise_std)

    def consume_bundle(self, bundle: PortBundle, ctx: RuntimeContext) -> None:
        _ = ctx
        rows_hidden = bundle.entries.get("hidden")
        if rows_hidden is None:
            return
        stride = max(1, int(self.config.export_every_n_steps))
        if (int(bundle.key.engine_step_id) % stride) != 0:
            return
        self._handle_hidden_rows(rows_hidden)

    def on_step_end(self, ctx: RuntimeContext) -> None:
        _ = ctx
        self._feedback_calls += 1
        if self.config.enable_async:
            interval = max(1, int(self.config.feedback_interval))
            if int(self.config.feedback_interval) <= 0:
                return
            if (self._feedback_calls % interval) != 0:
                return
            self._stream_runtime.synchronize()
            self._worker.drain(limit=0, noise_std=self.config.noise_std)

    def synchronize(self) -> None:
        if self.config.enable_async:
            self._stream_runtime.synchronize()
            self._worker.drain(limit=0, noise_std=self.config.noise_std)

    def read_stats(self) -> Dict[str, float]:
        out = {
            "consumed_batches": float(self._consumed_batches),
            "consumed_rows": float(self._consumed_rows),
            "dropped_batches": float(self._dropped_batches),
            "feedback_calls": float(self._feedback_calls),
        }
        out.update(self._worker.stats())
        return out
