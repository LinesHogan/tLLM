#!/usr/bin/env python3
"""Unit tests for documentation link integrity in tools docs."""

from __future__ import annotations

import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_README = _REPO_ROOT / "doc" / "README.md"
_TUTORIAL = _REPO_ROOT / "doc" / "getting-started" / "write-your-first-consumer.md"
_VALIDATION = _REPO_ROOT / "doc" / "developer-guides" / "validation.md"


class DocsLinksUnitTest(unittest.TestCase):
    def test_readme_links_consumer_tutorial(self) -> None:
        text = _README.read_text(encoding="utf-8")
        self.assertIn("getting-started/write-your-first-consumer.md", text)
        self.assertIn("developer-guides/validation.md", text)

    def test_tutorial_mentions_dummy_consumer_paths(self) -> None:
        text = _TUTORIAL.read_text(encoding="utf-8")
        self.assertIn("tllm/consumers/dummy/", text)
        self.assertIn("consumer.py", text)
        self.assertIn("DummyConsumer", text)

    def test_validation_doc_exists(self) -> None:
        text = _VALIDATION.read_text(encoding="utf-8")
        self.assertIn("Decode MSE", text)
        self.assertIn("ESamp Correctness", text)

    def test_dummy_tutorial_mentions_read_write_demo(self) -> None:
        text = _TUTORIAL.read_text(encoding="utf-8")
        self.assertIn("Reads `residual_stream`", text)
        self.assertIn("Export to CPU Asynchronously", text)
        self.assertIn("queue", text)


if __name__ == "__main__":
    unittest.main()
