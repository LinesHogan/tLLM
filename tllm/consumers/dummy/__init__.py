#!/usr/bin/env python3
"""Dummy consumer package used as extension template."""

from tllm.consumers.dummy.config import DummyConsumerConfig
from tllm.consumers.dummy.consumer import DummyConsumer

__all__ = ["DummyConsumer", "DummyConsumerConfig"]
