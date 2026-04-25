#!/usr/bin/env python3
"""Unit tests for residual locator/path bindings in the runtime layer."""

from __future__ import annotations

import unittest

from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime.ports import residual_bindings


class RuntimeResidualBindingsUnitTest(unittest.TestCase):
    def _cfg(self):
        return type(
            "Config",
            (),
            {
                "source_layer_path": "model.model.layers[0].input_layernorm",
                "target_layer_path": "model.model.layers[-1].input_layernorm",
            },
        )()

    def _raw_paths(self):
        return residual_bindings.default_raw_paths_from_config(self._cfg())

    def test_build_raw_tap_paths_uses_required_logical_locators(self) -> None:
        paths = residual_bindings.build_raw_tap_paths(
            raw_paths_by_locator=self._raw_paths(),
            required={(0, "block_output", "decode")},
        )

        self.assertEqual(paths, ["model.model.layers[0].input_layernorm"])

    def test_build_resolved_bindings_maps_logical_locators_to_runtime_paths(self) -> None:
        bindings = residual_bindings.build_resolved_bindings(
            raw_paths_by_locator=self._raw_paths(),
            resolve_alias={
                "model.model.layers[0].input_layernorm": "layers.0",
                "model.model.layers[-1].input_layernorm": "layers.1",
            },
            required=set(),
        )

        self.assertEqual(list(bindings.keys()), ["layers.0", "layers.1"])
        self.assertEqual(bindings["layers.0"].locator, ResidualLocator(layer=0, site="block_output", phase="decode"))
        self.assertTrue(bindings["layers.0"].include_request_meta)
        self.assertEqual(bindings["layers.1"].locator, ResidualLocator(layer=-1, site="block_output", phase="decode"))
        self.assertFalse(bindings["layers.1"].include_request_meta)

    def test_resolved_path_for_locator_reads_from_binding_table(self) -> None:
        bindings = residual_bindings.build_resolved_bindings(
            raw_paths_by_locator=self._raw_paths(),
            resolve_alias={
                "model.model.layers[0].input_layernorm": "layers.0",
                "model.model.layers[-1].input_layernorm": "layers.1",
            },
            required=set(),
        )

        path = residual_bindings.resolved_path_for_locator(
            bindings,
            ResidualLocator(layer=-1, site="block_output", phase="decode"),
        )

        self.assertEqual(path, "layers.1")

    def test_default_raw_paths_from_config_is_only_the_transitional_cfg_adapter(self) -> None:
        raw_paths = residual_bindings.default_raw_paths_from_config(self._cfg())

        self.assertEqual(
            raw_paths,
            {
                ResidualLocator(layer=0, site="block_output", phase="decode"): "model.model.layers[0].input_layernorm",
                ResidualLocator(layer=-1, site="block_output", phase="decode"): "model.model.layers[-1].input_layernorm",
            },
        )


if __name__ == "__main__":
    unittest.main()
