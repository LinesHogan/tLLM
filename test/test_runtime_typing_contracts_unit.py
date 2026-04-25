#!/usr/bin/env python3
"""Unit tests for runtime typing contracts on retained framework code."""

from __future__ import annotations

import inspect
import typing
import unittest

from tllm.contracts.runtime_context import RuntimeContext
from tllm.consumers.esamp.consumer import ESampConsumer
from tllm.runtime import consumer_compat
from tllm.runtime import residual_runtime as esamp_runtime
from tllm.runtime.ports.provider_registry import ProviderRegistry
from tllm.runtime.vllm_patch import port_runtime_hooks
from tllm.runtime.vllm_patch import common_hooks


class RuntimeTypingContractsUnitTest(unittest.TestCase):
    def test_runtime_context_runner_and_model_are_not_any(self) -> None:
        hints = typing.get_type_hints(RuntimeContext)

        self.assertIsNot(hints["runner"], typing.Any)
        self.assertIsNot(hints["model"], typing.Any)

    def test_common_hooks_build_runtime_context_runner_is_not_any(self) -> None:
        hints = typing.get_type_hints(common_hooks.build_runtime_context)

        self.assertIsNot(hints["runner"], typing.Any)

    def test_consumer_compat_does_not_expose_any_consumer_parameters(self) -> None:
        for fn in (
            consumer_compat.consumer_subscriptions,
            consumer_compat.apply_feedback,
            consumer_compat.dispatch_consumer_event,
            consumer_compat.synchronize,
        ):
            hints = typing.get_type_hints(fn)
            for name, hint in hints.items():
                if name == "return":
                    continue
                if name == "ctx":
                    continue
                self.assertIsNot(
                    hint,
                    typing.Any,
                    msg=f"{fn.__module__}.{fn.__name__} parameter `{name}` should not be Any",
                )

    def test_provider_registry_is_generic_over_provider_type(self) -> None:
        register_hints = typing.get_type_hints(ProviderRegistry.register)
        get_hints = typing.get_type_hints(ProviderRegistry.get)

        self.assertIsNot(register_hints["provider"], typing.Any)
        self.assertIsNot(get_hints["return"], typing.Any)

    def test_runtime_context_exposes_runner_protocol_alias(self) -> None:
        self.assertTrue(hasattr(inspect.getmodule(RuntimeContext), "RunnerLike"))

    def test_esamp_consumer_model_annotations_are_not_any(self) -> None:
        resolve_hints = typing.get_type_hints(ESampConsumer._resolve_layer_with_fallback)
        template_hints = typing.get_type_hints(ESampConsumer._maybe_prepare_initializer)

        self.assertIsNot(resolve_hints["model"], typing.Any)
        self.assertIsNot(template_hints["model"], typing.Any)

    def test_esamp_runtime_public_runtime_helpers_are_not_any(self) -> None:
        runner_hints = typing.get_type_hints(esamp_runtime.runner_uses_compilation_or_cudagraph)
        resolve_hints = typing.get_type_hints(esamp_runtime.resolve_module_by_path_with_fallback)
        runtime_hints = typing.get_type_hints(esamp_runtime.ResidualRuntimeState)

        self.assertIsNot(runner_hints["runner"], typing.Any)
        self.assertIsNot(resolve_hints["model"], typing.Any)
        self.assertIsNot(runtime_hints["consumer"], typing.Any)

    def test_esamp_runtime_exposes_runtime_consumer_alias(self) -> None:
        self.assertTrue(hasattr(esamp_runtime, "RuntimeConsumer"))

    def test_port_runtime_hooks_decode_entrypoints_are_not_any(self) -> None:
        prepare_hints = typing.get_type_hints(port_runtime_hooks.prepare_decode_localization)
        tap_hints = typing.get_type_hints(port_runtime_hooks.build_tap_path_list)
        dispatch_hints = typing.get_type_hints(port_runtime_hooks.dispatch_decode_port_bundles)

        self.assertIsNot(prepare_hints["core"], typing.Any)
        self.assertIsNot(prepare_hints["runner"], typing.Any)
        self.assertIsNot(tap_hints["core"], typing.Any)
        self.assertIsNot(dispatch_hints["core"], typing.Any)
        self.assertIsNot(dispatch_hints["runner"], typing.Any)


if __name__ == "__main__":
    unittest.main()
