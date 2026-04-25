#!/usr/bin/env python3
"""Unit tests for ESamp model-bank forward backends."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import pytest
import torch
import torch.nn.functional as F

from tllm.consumers.esamp import engine as engine_module
from tllm.consumers.esamp.engine import ESampTrainEngine
from tllm.consumers.esamp.model_bank_backend import (
    TorchModelBankForwardBackend,
    TritonGroupedModelBankForwardBackend,
    normalize_model_bank_forward_backend,
    select_model_bank_forward_backend,
)


def _params(*, slots: int = 3, hidden: int = 8, rank: int = 4, device: str = "cpu", dtype: torch.dtype = torch.float32):
    torch.manual_seed(1234)
    return SimpleNamespace(
        a=torch.randn((slots, hidden, rank), device=device, dtype=dtype) * 0.1,
        g=torch.randn((slots, hidden, rank), device=device, dtype=dtype) * 0.1,
        b=torch.randn((slots, rank, hidden), device=device, dtype=dtype) * 0.1,
        gate_bias=torch.randn((slots, rank), device=device, dtype=dtype) * 0.1,
        out_ln_weight=torch.randn((slots, hidden), device=device, dtype=dtype) * 0.1 + 1.0,
        out_ln_bias=torch.randn((slots, hidden), device=device, dtype=dtype) * 0.1,
    )


def _manual_forward(params, slot_ids: torch.Tensor, src: torch.Tensor, *, use_output_layernorm: bool) -> torch.Tensor:
    a_rows = params.a.index_select(0, slot_ids)
    g_rows = params.g.index_select(0, slot_ids)
    b_rows = params.b.index_select(0, slot_ids)
    gb_rows = params.gate_bias.index_select(0, slot_ids)
    up = torch.bmm(src.unsqueeze(1), a_rows).squeeze(1)
    gate = F.silu(torch.bmm(src.unsqueeze(1), g_rows).squeeze(1) + gb_rows)
    out = torch.bmm((up * gate).unsqueeze(1), b_rows).squeeze(1) + src
    if use_output_layernorm:
        out = F.layer_norm(out, (int(out.shape[-1]),), weight=None, bias=None)
        out = out * params.out_ln_weight.index_select(0, slot_ids) + params.out_ln_bias.index_select(0, slot_ids)
    return out


class ESampModelBankBackendUnitTest(unittest.TestCase):
    def test_torch_backend_matches_reference_with_reused_slots_and_layernorm(self) -> None:
        params = _params()
        slot_ids = torch.tensor([2, 0, 2, 1], dtype=torch.long)
        src = torch.randn((4, 8), dtype=torch.float32)

        out = TorchModelBankForwardBackend().forward(
            slot_ids=slot_ids,
            src=src,
            params=params,
            use_output_layernorm=True,
        )

        self.assertTrue(torch.allclose(out, _manual_forward(params, slot_ids, src, use_output_layernorm=True)))

    def test_backend_name_normalization(self) -> None:
        self.assertEqual(normalize_model_bank_forward_backend(None), "torch")
        self.assertEqual(normalize_model_bank_forward_backend("reference"), "torch")
        self.assertEqual(normalize_model_bank_forward_backend("triton"), "triton_grouped")

    def test_selector_keeps_training_path_on_torch(self) -> None:
        backend = select_model_bank_forward_backend("triton_grouped", require_grad=True, device=torch.device("cuda"))

        self.assertIsInstance(backend, TorchModelBankForwardBackend)

    def test_engine_routes_model_bank_forward_through_configured_backend(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=2, lr=1e-3, enabled=True)
        params = _params(slots=3, hidden=8, rank=4)
        consumer.state.model_bank = engine_module._ModelBankParams(
            a=torch.nn.Parameter(params.a),
            g=torch.nn.Parameter(params.g),
            b=torch.nn.Parameter(params.b),
            gate_bias=torch.nn.Parameter(params.gate_bias),
            out_ln_weight=torch.nn.Parameter(params.out_ln_weight),
            out_ln_bias=torch.nn.Parameter(params.out_ln_bias),
        )
        consumer.state.model_bank_use_output_layernorm = False
        consumer.state.model_bank_forward_backend = "triton_grouped"
        slot_ids = torch.tensor([2, 0, 2, 1], dtype=torch.long)
        src = torch.randn((4, 8), dtype=torch.float32)

        with unittest.mock.patch.object(
            engine_module,
            "select_model_bank_forward_backend",
            return_value=TorchModelBankForwardBackend(),
        ) as select_backend:
            out = consumer._model_bank_forward_locked(slot_ids, src, require_grad=False)

        select_backend.assert_called_once_with("triton_grouped", require_grad=False, device=src.device)
        self.assertTrue(torch.allclose(out, _manual_forward(consumer.state.model_bank, slot_ids, src, use_output_layernorm=False)))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton model-bank backend tests")
    def test_triton_grouped_backend_matches_torch_without_layernorm_on_cuda(self) -> None:
        params = _params(slots=4, hidden=32, rank=8, device="cuda", dtype=torch.float32)
        slot_ids = torch.tensor([2, 0, 2, 1, 3], device="cuda", dtype=torch.long)
        src = torch.randn((5, 32), device="cuda", dtype=torch.float32)

        expected = TorchModelBankForwardBackend().forward(
            slot_ids=slot_ids,
            src=src,
            params=params,
            use_output_layernorm=False,
        )
        out = TritonGroupedModelBankForwardBackend().forward(
            slot_ids=slot_ids,
            src=src,
            params=params,
            use_output_layernorm=False,
        )

        self.assertTrue(torch.allclose(out, expected, atol=1e-4, rtol=1e-4))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton model-bank backend tests")
    def test_triton_grouped_backend_matches_torch_with_layernorm_on_cuda(self) -> None:
        params = _params(slots=4, hidden=32, rank=8, device="cuda", dtype=torch.float32)
        slot_ids = torch.tensor([2, 0, 2, 1, 3], device="cuda", dtype=torch.long)
        src = torch.randn((5, 32), device="cuda", dtype=torch.float32)

        expected = TorchModelBankForwardBackend().forward(
            slot_ids=slot_ids,
            src=src,
            params=params,
            use_output_layernorm=True,
        )
        out = TritonGroupedModelBankForwardBackend().forward(
            slot_ids=slot_ids,
            src=src,
            params=params,
            use_output_layernorm=True,
        )

        self.assertTrue(torch.allclose(out, expected, atol=1e-4, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
