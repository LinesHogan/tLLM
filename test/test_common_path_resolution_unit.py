#!/usr/bin/env python3
"""Unit tests for extracted common path resolution helpers."""

from __future__ import annotations

import unittest

from tllm.common.path_resolution import (
    candidate_capture_paths,
    parse_component,
    resolve_object_by_path,
)


class _Node:
    def __init__(self) -> None:
        self.model = self
        self.layers = ["L0", "L1"]


class CommonPathResolutionUnitTest(unittest.TestCase):
    def test_parse_component_attr_and_indices(self) -> None:
        attr, idx = parse_component("layers[2][-1]")
        self.assertEqual(attr, "layers")
        self.assertEqual(idx, [2, -1])

    def test_resolve_object_by_path_dotted_with_indices(self) -> None:
        root = _Node()
        obj = resolve_object_by_path(root, "model.layers[1]")
        self.assertEqual(obj, "L1")

    def test_candidate_capture_paths_progressively_strips_prefix(self) -> None:
        self.assertEqual(
            candidate_capture_paths("model.model.layers[0]"),
            ["model.model.layers[0]", "model.layers[0]", "layers[0]"],
        )


if __name__ == "__main__":
    unittest.main()
