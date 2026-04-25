#!/usr/bin/env python3
"""Typed internal contracts for same-step sampler intervention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch

CanonicalSamplerBackend = Literal["post_filter_exact", "pre_filter_dense", "post_filter_dense_cache"]
SamplerBackend = Literal[
    "post_filter_exact",
    "post_filter_exact_minp",
    "post_filter_exact_torch",
    "pre_filter_dense",
    "post_filter_dense_cache",
]
SAMPLER_BACKEND_CHOICES: tuple[str, ...] = (
    "post_filter_exact",
    "post_filter_exact_minp",
    "post_filter_exact_torch",
    "pre_filter_dense",
    "post_filter_dense_cache",
)
_SAMPLER_BACKEND_ALIASES: dict[str, CanonicalSamplerBackend] = {
    "post_filter_exact": "post_filter_exact",
    "post_filter_exact_minp": "post_filter_exact",
    "post_filter_exact_torch": "post_filter_exact",
    "pre_filter_dense": "pre_filter_dense",
    "post_filter_dense_cache": "post_filter_dense_cache",
}
TemperatureMode = Literal["all_random", "all_greedy", "mixed"]
CandidateSamplingMode = Literal["random", "greedy", "mixed"]


def normalize_sampler_backend(backend: str | None) -> CanonicalSamplerBackend:
    key = str(backend or "post_filter_exact").strip() or "post_filter_exact"
    try:
        return _SAMPLER_BACKEND_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported distiller sampler backend: {key}") from exc


def _metadata_temperature_mode(sampling_metadata: Any) -> TemperatureMode:
    if bool(getattr(sampling_metadata, "all_greedy", False)):
        return "all_greedy"
    if bool(getattr(sampling_metadata, "all_random", False)):
        return "all_random"
    return "mixed"


def _normalize_top_k(top_k: torch.Tensor | None, *, vocab_size: int | None) -> torch.Tensor | None:
    if top_k is None:
        return None
    if top_k.numel() <= 0:
        return None
    if top_k.device.type != "cpu":
        return top_k
    if bool(torch.all(top_k <= 0)):
        return None
    if vocab_size is not None:
        vocab = int(vocab_size)
        if bool(torch.all((top_k <= 0) | (top_k >= vocab))):
            return None
    return top_k


def _normalize_top_p(top_p: torch.Tensor | None) -> torch.Tensor | None:
    if top_p is None:
        return None
    if top_p.numel() <= 0:
        return None
    if top_p.device.type != "cpu":
        return top_p
    if bool(torch.all(top_p >= 1.0)):
        return None
    return top_p


def _normalize_min_p(min_p: torch.Tensor | None) -> torch.Tensor | None:
    if min_p is None:
        return None
    if min_p.numel() <= 0:
        return None
    if min_p.device.type != "cpu":
        return min_p
    if bool(torch.all(min_p <= 0.0)):
        return None
    return min_p


@dataclass(frozen=True)
class SamplerStepView:
    engine_step_id: int
    phase: str
    logits: torch.Tensor
    sampling_metadata: Any
    decode_count: int
    request_ids: tuple[str, ...]
    prompt_idxs: tuple[int, ...]
    sample_idxs: tuple[int, ...]
    prompt_idx_tensor: torch.Tensor | None
    sample_idx_tensor: torch.Tensor | None
    source_hidden: torch.Tensor
    device: torch.device
    model: Any
    runner: Any


@dataclass(frozen=True)
class SamplerFilterSpec:
    top_k: torch.Tensor | None
    top_p: torch.Tensor | None
    min_p: torch.Tensor | None
    temperature_mode: TemperatureMode
    has_generators: bool = False

    @property
    def has_top_k(self) -> bool:
        return self.top_k is not None

    @property
    def has_top_p(self) -> bool:
        return self.top_p is not None

    @property
    def has_min_p(self) -> bool:
        return self.min_p is not None


def filter_spec_from_sampling_metadata(
    sampling_metadata: Any,
    *,
    vocab_size: int | None = None,
) -> SamplerFilterSpec:
    generators = getattr(sampling_metadata, "generators", {}) or {}
    return SamplerFilterSpec(
        top_k=_normalize_top_k(getattr(sampling_metadata, "top_k", None), vocab_size=vocab_size),
        top_p=_normalize_top_p(getattr(sampling_metadata, "top_p", None)),
        min_p=_normalize_min_p(getattr(sampling_metadata, "min_p", None)),
        temperature_mode=_metadata_temperature_mode(sampling_metadata),
        has_generators=bool(generators),
    )


@dataclass(frozen=True)
class CandidateModifierRequest:
    logits: torch.Tensor
    affected_row_ids: torch.Tensor
    filter_spec: SamplerFilterSpec
    sampling_mode: CandidateSamplingMode


@dataclass(frozen=True)
class CandidateModifierState:
    beta: float
    backend: SamplerBackend
    affected_row_ids: torch.Tensor
    pred_hidden: torch.Tensor
    lm_head_weight: torch.Tensor
    lm_head_bias: torch.Tensor | None
    precomputed_dense_logits: torch.Tensor | None = None
    pred_hidden_row_map: torch.Tensor | None = None


@dataclass(frozen=True)
class CandidateSampleResult:
    sampled_token_ids: torch.Tensor
    debug_stats: Mapping[str, int | float | str] | None = None


SamplerModifierState = CandidateModifierState
