#!/usr/bin/env python3
"""Optional SVD-based model-bank initializer for ESamp."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any, Literal, Sequence

import torch

from tllm.common.state import resolve_object_by_path

if TYPE_CHECKING:
    from tllm.consumers.esamp.engine import ESampTrainEngine


SVDInitMethod = Literal["ffn_fast_svd", "ridge_svd"]
_SILU_ONE = 1.0 / (1.0 + math.exp(-1.0))


@dataclass(slots=True, frozen=True)
class SVDModelBankInitializerConfig:
    method: SVDInitMethod = "ffn_fast_svd"
    ridge_lambda: float = 1e-2
    min_rows: int = 32
    max_wait_steps: int = 4


@dataclass(slots=True)
class _SVDRuntimeState:
    template_a: torch.Tensor | None = None
    template_b: torch.Tensor | None = None
    template_key: str = ""
    slot_init_done: dict[int, bool] = field(default_factory=dict)
    slot_seen_steps: dict[int, int] = field(default_factory=dict)
    slot_src_cache: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    slot_tgt_cache: dict[int, list[torch.Tensor]] = field(default_factory=dict)


def model_bank_uses_ffn_fast_svd(cfg: Any) -> bool:
    method = str(getattr(cfg, "method", getattr(cfg, "model_bank_init_method", ""))).strip().lower()
    if method in {"ffn-fast-svd", "ffn+svd"}:
        method = "ffn_fast_svd"
    return method == "ffn_fast_svd"


def resolve_linear_weight(module: Any, names: Sequence[str]) -> torch.Tensor | None:
    for name in names:
        obj = getattr(module, name, None)
        if isinstance(obj, torch.nn.Linear) and isinstance(getattr(obj, "weight", None), torch.Tensor):
            weight = obj.weight
        elif isinstance(obj, torch.Tensor):
            weight = obj
        else:
            continue
        if weight.ndim == 2:
            return weight
    return None


def compose_hidden_linear_map(
    *,
    down_w: torch.Tensor,
    up_w: torch.Tensor,
    hidden_size: int,
) -> torch.Tensor | None:
    candidates = [
        (down_w, up_w),
        (down_w, up_w.transpose(0, 1)),
        (down_w.transpose(0, 1), up_w),
        (down_w.transpose(0, 1), up_w.transpose(0, 1)),
    ]
    for left, right in candidates:
        if left.ndim != 2 or right.ndim != 2:
            continue
        if int(left.shape[1]) != int(right.shape[0]):
            continue
        if int(left.shape[0]) != int(hidden_size) or int(right.shape[1]) != int(hidden_size):
            continue
        return torch.matmul(left, right)
    return None


def _detached_cpu_float(weight: torch.Tensor) -> torch.Tensor:
    return weight.detach().to(device="cpu", dtype=torch.float32)


def extract_ffn_fast_svd_template(
    *,
    target_layer: torch.nn.Module,
    hidden_size: int,
    rank: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    rank_eff = min(max(1, int(rank)), int(hidden_size))
    named_params = {str(n): p for n, p in target_layer.named_parameters(recurse=True) if isinstance(p, torch.Tensor)}

    def _find_param_suffix(suffixes: Sequence[str]) -> torch.Tensor | None:
        for name, param in named_params.items():
            for suffix in suffixes:
                if (name == suffix or name.endswith("." + suffix)) and param.ndim == 2:
                    return param
        return None

    up_w = _find_param_suffix(["up_proj.weight", "fc1.weight", "w1.weight", "dense_h_to_4h.weight"])
    gate_w = _find_param_suffix(["gate_proj.weight", "w3.weight"])
    down_w = _find_param_suffix(["down_proj.weight", "fc2.weight", "w2.weight", "dense_4h_to_h.weight"])
    if up_w is None and gate_w is None:
        gate_up_w = _find_param_suffix(["gate_up_proj.weight"])
        if gate_up_w is not None and int(gate_up_w.shape[0]) % 2 == 0:
            half = int(gate_up_w.shape[0]) // 2
            gate_w = gate_up_w[:half, :]
            up_w = gate_up_w[half:, :]

    linear = None
    if up_w is not None and down_w is not None:
        up_f = _detached_cpu_float(up_w)
        down_f = _detached_cpu_float(down_w)
        if gate_w is not None and tuple(gate_w.shape) == tuple(up_w.shape):
            up_f = 0.5 * (up_f + _detached_cpu_float(gate_w))
        linear = compose_hidden_linear_map(down_w=down_f, up_w=up_f, hidden_size=hidden_size)

    if linear is None:
        mlp_candidates: list[Any] = [getattr(target_layer, "mlp", None), getattr(target_layer, "feed_forward", None), target_layer]
        for block in mlp_candidates:
            if block is None:
                continue
            up_w = resolve_linear_weight(block, ["up_proj", "fc1", "w1", "dense_h_to_4h"])
            gate_w = resolve_linear_weight(block, ["gate_proj", "w3"])
            down_w = resolve_linear_weight(block, ["down_proj", "fc2", "w2", "dense_4h_to_h"])
            if up_w is None or down_w is None:
                continue
            up_f = _detached_cpu_float(up_w)
            down_f = _detached_cpu_float(down_w)
            if gate_w is not None and tuple(gate_w.shape) == tuple(up_w.shape):
                up_f = 0.5 * (up_f + _detached_cpu_float(gate_w))
            linear = compose_hidden_linear_map(down_w=down_f, up_w=up_f, hidden_size=hidden_size)
            if linear is not None:
                break

    if linear is None:
        return None, None

    m, n = int(linear.shape[0]), int(linear.shape[1])
    r = min(rank_eff, m, n)
    if r <= 0:
        return None, None
    q = min(max(r + 8, r), min(m, n))
    try:
        u, s, v = torch.svd_lowrank(linear, q=q, niter=2)
        u_r = u[:, :r]
        s_r = s[:r]
        v_r = v[:, :r]
    except Exception:
        try:
            u_full, s_full, vh_full = torch.linalg.svd(linear, full_matrices=False)
        except Exception:
            return None, None
        u_r = u_full[:, :r]
        s_r = s_full[:r]
        v_r = vh_full[:r, :].transpose(0, 1)
    s_r = torch.clamp(s_r, min=1e-8)
    sqrt_s = torch.sqrt(s_r)
    a = u_r * sqrt_s.unsqueeze(0)
    b = sqrt_s.unsqueeze(1) * v_r.transpose(0, 1)
    return a.cpu(), (b / max(1e-6, _SILU_ONE)).cpu()


def pick_template_layer_for_target(
    *,
    model: Any,
    target_resolved: str,
    target_layer: torch.nn.Module,
) -> tuple[torch.nn.Module, str]:
    suffix = ".input_layernorm"
    if not str(target_resolved).endswith(suffix):
        return target_layer, target_resolved
    parent_path = str(target_resolved)[: -len(suffix)]
    if not parent_path:
        return target_layer, target_resolved
    try:
        parent = resolve_object_by_path(model, parent_path)
    except Exception:
        return target_layer, target_resolved
    return (parent, parent_path) if isinstance(parent, torch.nn.Module) else (target_layer, target_resolved)


class SVDModelBankInitializer:
    """Optional SVD-based model-bank initializer."""

    def __init__(self, config: SVDModelBankInitializerConfig) -> None:
        self.config = SVDModelBankInitializerConfig(
            method=str(config.method).strip().lower(),  # type: ignore[arg-type]
            ridge_lambda=float(config.ridge_lambda),
            min_rows=max(1, int(config.min_rows)),
            max_wait_steps=max(1, int(config.max_wait_steps)),
        )
        self.state = _SVDRuntimeState()

    def reset_runtime_state(self) -> None:
        self.state.slot_init_done = {}
        self.state.slot_seen_steps = {}
        self.state.slot_src_cache = {}
        self.state.slot_tgt_cache = {}

    def reset_all(self) -> None:
        self.state = _SVDRuntimeState()

    def set_template(self, *, a: torch.Tensor | None, b: torch.Tensor | None, key: str = "") -> None:
        if a is None or b is None or a.ndim != 2 or b.ndim != 2:
            self.state.template_a = None
            self.state.template_b = None
            self.state.template_key = ""
            return
        self.state.template_a = a.detach().to(device="cpu")
        self.state.template_b = b.detach().to(device="cpu")
        self.state.template_key = str(key)

    def prepare_from_model(
        self,
        *,
        engine: Any,
        model: Any,
        target_layer: torch.nn.Module,
        target_resolved: str,
        hidden_size: int,
    ) -> None:
        if self.config.method != "ffn_fast_svd":
            return
        template_layer, template_key_path = pick_template_layer_for_target(
            model=model,
            target_resolved=target_resolved,
            target_layer=target_layer,
        )
        rank = int(engine.state.model_bank_rank)
        key = f"{template_key_path}|hidden={int(hidden_size)}|rank={rank}"
        if self.state.template_key == key and self.state.template_a is not None and self.state.template_b is not None:
            return
        a, b = extract_ffn_fast_svd_template(
            target_layer=template_layer,
            hidden_size=int(hidden_size),
            rank=rank,
        )
        self.set_template(a=a, b=b, key=key if a is not None and b is not None else "")

    def on_slot_assigned(self, engine: Any, slot: int) -> None:
        if self.config.method == "ffn_fast_svd":
            ok = self._init_slot_ffn_fast_svd(engine, slot) if self.state.template_a is not None and self.state.template_b is not None else False
            self.state.slot_init_done[int(slot)] = bool(ok)
        else:
            self.state.slot_init_done[int(slot)] = False
            self.state.slot_seen_steps[int(slot)] = 0
            self.state.slot_src_cache[int(slot)] = []
            self.state.slot_tgt_cache[int(slot)] = []

    def ensure_existing_slot(self, engine: Any, slot: int) -> None:
        if self.config.method != "ffn_fast_svd" or self.state.slot_init_done.get(int(slot), False):
            return
        ok = self._init_slot_ffn_fast_svd(engine, int(slot)) if self.state.template_a is not None and self.state.template_b is not None else True
        self.state.slot_init_done[int(slot)] = bool(ok)

    def maybe_prepare_slots(self, engine: Any, slot_ids: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor) -> None:
        if self.config.method != "ridge_svd" or slot_ids.numel() == 0:
            return
        cache_cap = max(self.config.min_rows, self.config.min_rows * 2)
        for slot, is_done in list(self.state.slot_init_done.items()):
            if bool(is_done):
                continue
            slot = int(slot)
            mask = slot_ids == slot
            slot_src = src[mask].detach()
            slot_tgt = tgt[mask].detach()
            if slot_src.numel() <= 0:
                continue
            src_cache = self.state.slot_src_cache.setdefault(slot, [])
            tgt_cache = self.state.slot_tgt_cache.setdefault(slot, [])
            src_cache.append(slot_src)
            tgt_cache.append(slot_tgt)
            self.state.slot_seen_steps[slot] = int(self.state.slot_seen_steps.get(slot, 0)) + 1
            cache_rows = int(sum(int(x.shape[0]) for x in src_cache))
            while cache_rows > cache_cap and len(src_cache) > 1:
                src_cache.pop(0)
                tgt_cache.pop(0)
                cache_rows = int(sum(int(x.shape[0]) for x in src_cache))
            if cache_rows < self.config.min_rows and int(self.state.slot_seen_steps[slot]) < self.config.max_wait_steps:
                continue
            ok = self._init_slot_ridge_svd(engine, slot, torch.cat(src_cache, dim=0), torch.cat(tgt_cache, dim=0))
            self.state.slot_init_done[slot] = bool(ok)
            self.state.slot_src_cache.pop(slot, None)
            self.state.slot_tgt_cache.pop(slot, None)

    def _model_bank_params(self, engine: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        params = engine.state.model_bank
        if params is None:
            raise RuntimeError("model bank parameters are not initialized")
        return params.a, params.g, params.b, params.gate_bias

    def _init_slot_ffn_fast_svd(self, engine: Any, slot: int) -> bool:
        if self.state.template_a is None or self.state.template_b is None:
            return False
        a_param, g_param, b_param, gate_bias = self._model_bank_params(engine)
        rank = min(
            int(engine.state.model_bank_rank_effective),
            int(a_param.shape[2]),
            int(self.state.template_a.shape[1]),
            int(self.state.template_b.shape[0]),
        )
        if rank <= 0:
            return False
        with torch.no_grad():
            a_param[slot].zero_()
            g_param[slot].zero_()
            b_param[slot].zero_()
            a_param[slot, :, :rank].copy_(self.state.template_a[:, :rank].to(device=a_param.device, dtype=a_param.dtype))
            b_param[slot, :rank, :].copy_(self.state.template_b[:rank, :].to(device=b_param.device, dtype=b_param.dtype))
            gate_bias[slot].fill_(1.0)
            params = engine.state.model_bank
            if params is not None:
                params.out_ln_weight[slot].fill_(1.0)
                params.out_ln_bias[slot].zero_()
        return True

    def _init_slot_ridge_svd(self, engine: Any, slot: int, src: torch.Tensor, tgt: torch.Tensor) -> bool:
        a_param, g_param, b_param, gate_bias = self._model_bank_params(engine)
        if src.numel() == 0 or tgt.numel() == 0 or src.shape != tgt.shape or src.ndim != 2:
            return False
        rows = int(src.shape[0])
        hidden = int(src.shape[1])
        rank = min(int(engine.state.model_bank_rank_effective), hidden, rows)
        if rank <= 0:
            return False
        x = src.float()
        y = (tgt - src).float()
        ridge = max(0.0, float(self.config.ridge_lambda))
        xtx = x.transpose(0, 1) @ x
        if ridge > 0:
            xtx = xtx + (ridge * rows) * torch.eye(hidden, device=xtx.device, dtype=xtx.dtype)
        xty = x.transpose(0, 1) @ y
        try:
            w = torch.linalg.solve(xtx, xty)
        except RuntimeError:
            w = torch.linalg.lstsq(xtx, xty).solution
        try:
            u, s, vh = torch.linalg.svd(w, full_matrices=False)
        except RuntimeError:
            return False
        if s.numel() == 0:
            return False
        rank = min(rank, int(s.numel()))
        if rank <= 0:
            return False
        s_clamped = torch.clamp(s[:rank], min=1e-8)
        sqrt_s = torch.sqrt(s_clamped)
        a_fit = u[:, :rank] * sqrt_s.unsqueeze(0)
        b_fit = sqrt_s.unsqueeze(1) * vh[:rank, :]
        b_fit = b_fit / max(1e-6, _SILU_ONE)
        with torch.no_grad():
            a_param[slot].zero_()
            g_param[slot].zero_()
            b_param[slot].zero_()
            a_param[slot, :, :rank].copy_(a_fit.to(device=a_param.device, dtype=a_param.dtype))
            b_param[slot, :rank, :].copy_(b_fit.to(device=b_param.device, dtype=b_param.dtype))
            gate_bias[slot].fill_(1.0)
            params = engine.state.model_bank
            if params is not None:
                params.out_ln_weight[slot].fill_(1.0)
                params.out_ln_bias[slot].zero_()
        return True


def build_model_bank_initializer(
    config: SVDModelBankInitializerConfig | None,
) -> SVDModelBankInitializer | None:
    return None if config is None else SVDModelBankInitializer(config)
