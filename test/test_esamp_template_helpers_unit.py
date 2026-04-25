#!/usr/bin/env python3
"""Unit tests for extracted ESamp template helper functions."""

from __future__ import annotations

import unittest

import torch

from tllm.consumers.esamp.initializers.svd import (
    compose_hidden_linear_map,
    model_bank_uses_ffn_fast_svd,
    resolve_linear_weight,
)


class _Cfg:
    def __init__(self, method: str) -> None:
        self.model_bank_init_method = method


class _Mod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = torch.nn.Linear(4, 6, bias=False)


class ESampTemplateHelpersUnitTest(unittest.TestCase):
    def test_model_bank_method_aliases(self) -> None:
        self.assertTrue(model_bank_uses_ffn_fast_svd(_Cfg("ffn_fast_svd")))
        self.assertTrue(model_bank_uses_ffn_fast_svd(_Cfg("ffn-fast-svd")))
        self.assertFalse(model_bank_uses_ffn_fast_svd(_Cfg("ridge_svd")))

    def test_resolve_linear_weight(self) -> None:
        m = _Mod()
        w = resolve_linear_weight(m, ["down_proj", "gate_proj"])
        self.assertIsNotNone(w)
        assert w is not None
        self.assertEqual(tuple(w.shape), tuple(m.down_proj.weight.shape))

    def test_compose_hidden_linear_map(self) -> None:
        down_w = torch.randn((4, 8), dtype=torch.float32)
        up_w = torch.randn((8, 4), dtype=torch.float32)
        mat = compose_hidden_linear_map(down_w=down_w, up_w=up_w, hidden_size=4)
        self.assertIsNotNone(mat)
        assert mat is not None
        self.assertEqual(tuple(mat.shape), (4, 4))


if __name__ == "__main__":
    unittest.main()
