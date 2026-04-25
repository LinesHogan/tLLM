#!/usr/bin/env python3
"""Unit tests for runtime active-target detection."""

from __future__ import annotations

import unittest

from tllm.runtime import active_targets


class RuntimeActiveTargetsUnitTest(unittest.TestCase):
    def test_returns_false_without_plan(self) -> None:
        runtime = type("Runtime", (), {"dispatch_plan": None})()

        self.assertFalse(active_targets.runtime_has_active_targets(runtime))

    def test_returns_true_when_flow_targets_exist(self) -> None:
        plan = type(
            "Plan",
            (),
            {"has_active_targets": staticmethod(lambda: True)},
        )()
        runtime = type("Runtime", (), {"dispatch_plan": plan})()

        self.assertTrue(active_targets.runtime_has_active_targets(runtime))

    def test_returns_true_when_required_residual_layers_exist(self) -> None:
        plan = type(
            "Plan",
            (),
            {"has_active_targets": staticmethod(lambda: True)},
        )()
        runtime = type("Runtime", (), {"dispatch_plan": plan})()

        self.assertTrue(active_targets.runtime_has_active_targets(runtime))


if __name__ == "__main__":
    unittest.main()
