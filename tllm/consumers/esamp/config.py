#!/usr/bin/env python3
"""Configuration for the BaseConsumer ESamp wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from tllm.consumers.esamp.initializers.svd import SVDModelBankInitializerConfig
from tllm.consumers.esamp.model_bank_backend import ModelBankForwardBackendName, normalize_model_bank_forward_backend
from tllm.runtime.sampler_bridge.types import SamplerBackend, normalize_sampler_backend


@dataclass
class ESampConsumerConfig:
    consumer_id: str = "esamp"
    graph_scratch_rows: int = 0
    source_layer_path: str = "model.model.layers[0].input_layernorm"
    target_layer_path: str = "model.model.layers[-1].input_layernorm"
    enable_side_train: bool = True
    side_hidden_dim: int = 128
    side_lr: float = 1e-3
    per_request_models: bool = False
    per_request_model_bank: bool = False
    model_bank_slots: int = 0
    model_bank_flush_interval: int = 1
    model_bank_rank: int = 64
    model_bank_use_output_layernorm: bool = True
    model_bank_initializer: SVDModelBankInitializerConfig | None = None
    model_bank_train_cudagraph: bool = False
    model_bank_forward_backend: ModelBankForwardBackendName = "torch"
    trace_per_request_losses: bool = False
    trace_interval: int = 1
    trace_max_points: int = 0
    enable_distiller_intervention: bool = False
    distiller_beta: float = 0.0
    distiller_sampler_backend: SamplerBackend = "post_filter_exact"

    def __post_init__(self) -> None:
        self.distiller_sampler_backend = normalize_sampler_backend(self.distiller_sampler_backend)
        self.model_bank_forward_backend = normalize_model_bank_forward_backend(self.model_bank_forward_backend)
