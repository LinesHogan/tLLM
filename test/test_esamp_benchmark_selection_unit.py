#!/usr/bin/env python3
"""Unit tests for ESamp benchmark implementation selection helpers."""

from __future__ import annotations

import sys
import unittest
from unittest import mock

from tllm.workflows.benchmarks import esamp_benchmark as bench


class ESampBenchmarkSelectionUnitTest(unittest.TestCase):
    def test_cli_no_longer_exposes_consumer_implementation_selector(self) -> None:
        with mock.patch.object(sys, "argv", ["esamp_benchmark"]):
            args = bench._parse_args()
        self.assertFalse(hasattr(args, "consumer_implementation"))
        self.assertFalse(hasattr(args, "compare_consumer_implementations"))


if __name__ == "__main__":
    unittest.main()
