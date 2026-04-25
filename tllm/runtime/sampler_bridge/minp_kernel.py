#!/usr/bin/env python3
"""Min-p candidate sampling kernel interface."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.runtime.sampler_bridge.backends.dense_cache import gather_dense_candidate_logits
from tllm.runtime.sampler_bridge.min_p import min_p_keep_mask
from tllm.runtime.sampler_bridge.truth import project_candidate_logits
from tllm.runtime.sampler_bridge.types import CandidateModifierState, CandidateSampleResult


@dataclass(frozen=True)
class MinPCandidateKernelRequest:
    logits: torch.Tensor
    min_p: torch.Tensor
    state: CandidateModifierState
    greedy: bool = False


class TorchMinPCandidateKernel:
    name = "torch"

    def keep_mask(self, logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
        return min_p_keep_mask(logits, min_p)

    def sample(self, request: MinPCandidateKernelRequest) -> CandidateSampleResult:
        mask = self.keep_mask(request.logits, request.min_p)
        candidate_row_ids, token_ids = mask.nonzero(as_tuple=True)
        if candidate_row_ids.numel() <= 0:
            return CandidateSampleResult(
                sampled_token_ids=request.logits.argmax(dim=-1).to(dtype=torch.long),
                debug_stats={"kernel": self.name, "candidate_count": 0, "row_count": int(request.logits.shape[0])},
            )
        llm_candidate_logits = request.logits[candidate_row_ids, token_ids]
        state = request.state
        if state.backend == "post_filter_dense_cache" and state.precomputed_dense_logits is not None:
            distiller_candidate_logits = gather_dense_candidate_logits(
                state=state,
                row_ids=candidate_row_ids,
                token_ids=token_ids,
                dtype=request.logits.dtype,
                device=request.logits.device,
            )
        else:
            distiller_candidate_logits = project_candidate_logits(
                pred_hidden=state.pred_hidden,
                row_ids=candidate_row_ids,
                token_ids=token_ids,
                lm_head_weight=state.lm_head_weight,
                lm_head_bias=state.lm_head_bias,
                pred_hidden_row_map=state.pred_hidden_row_map,
            )
        candidate_logits = ((1.0 + float(state.beta)) * llm_candidate_logits) - (
            float(state.beta) * distiller_candidate_logits.to(device=request.logits.device, dtype=request.logits.dtype)
        )
        if request.greedy:
            scores = candidate_logits.to(dtype=torch.float32)
        else:
            noise = torch.empty_like(candidate_logits, dtype=torch.float32)
            noise.exponential_()
            scores = candidate_logits.to(dtype=torch.float32) - noise.log_()
        row_count = int(request.logits.shape[0])
        row_scores = torch.full((row_count,), -float("inf"), device=request.logits.device, dtype=torch.float32)
        row_scores.scatter_reduce_(0, candidate_row_ids, scores, reduce="amax", include_self=True)
        winner_mask = scores == row_scores.index_select(0, candidate_row_ids)
        sampled = torch.empty((row_count,), device=request.logits.device, dtype=torch.long)
        sampled.scatter_(0, candidate_row_ids[winner_mask], token_ids[winner_mask].to(device=sampled.device, dtype=torch.long))
        return CandidateSampleResult(
            sampled_token_ids=sampled,
            debug_stats={
                "kernel": self.name,
                "candidate_count": int(candidate_row_ids.numel()),
                "row_count": row_count,
            },
        )


class TritonMinPCandidateKernel(TorchMinPCandidateKernel):
    name = "triton_minp_keep_mask"

    def keep_mask(self, logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
        from tllm.runtime.sampler_bridge.kernels import minp_candidate_triton

        return minp_candidate_triton.keep_mask(logits, min_p)

    def sample(self, request: MinPCandidateKernelRequest) -> CandidateSampleResult:
        try:
            return super().sample(request)
        except NotImplementedError as exc:
            fallback = TorchMinPCandidateKernel().sample(request)
            stats = dict(fallback.debug_stats or {})
            stats["fallback_reason"] = str(exc)
            return CandidateSampleResult(sampled_token_ids=fallback.sampled_token_ids, debug_stats=stats)


def select_minp_candidate_kernel(
    *,
    prefer_triton: bool,
    logits_device: torch.device,
    logits: torch.Tensor | None = None,
) -> TorchMinPCandidateKernel:
    if prefer_triton and logits_device.type == "cuda":
        if logits is not None:
            from tllm.runtime.sampler_bridge.kernels import minp_candidate_triton

            supported, _reason = minp_candidate_triton.supports_keep_mask(
                logits,
                torch.empty((int(logits.shape[0]),), device=logits.device, dtype=torch.float32),
            )
            if not supported:
                return TorchMinPCandidateKernel()
        return TritonMinPCandidateKernel()
    return TorchMinPCandidateKernel()
