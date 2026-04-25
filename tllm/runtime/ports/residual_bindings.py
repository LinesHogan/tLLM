#!/usr/bin/env python3
"""Runtime bindings between logical residual locators and concrete layer paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from tllm.ports.residual_stream import ResidualLocator


@dataclass(frozen=True)
class ResidualPathBinding:
    locator: ResidualLocator
    resolved_path: str
    include_request_meta: bool = False


def _supported_logical_locators(required: set[tuple[int, str, str]]) -> list[ResidualLocator]:
    if not required:
        return [
            ResidualLocator(layer=0, site="block_output", phase="decode"),
            ResidualLocator(layer=-1, site="block_output", phase="decode"),
        ]

    locators: list[ResidualLocator] = []
    for layer, site, phase in sorted(required, key=lambda item: (item[0], item[1], item[2])):
        if str(site) != "block_output" or str(phase) != "decode":
            continue
        locators.append(ResidualLocator(layer=int(layer), site="block_output", phase="decode"))
    return locators


def default_raw_paths_from_config(cfg: object) -> dict[ResidualLocator, str]:
    return {
        ResidualLocator(layer=0, site="block_output", phase="decode"): str(getattr(cfg, "source_layer_path")),
        ResidualLocator(layer=-1, site="block_output", phase="decode"): str(getattr(cfg, "target_layer_path")),
    }


def raw_paths_from_runtime(runtime: object) -> dict[ResidualLocator, str]:
    raw_paths = getattr(runtime, "residual_raw_paths", None)
    if isinstance(raw_paths, Mapping) and raw_paths:
        return {
            locator: str(path)
            for locator, path in raw_paths.items()
            if isinstance(locator, ResidualLocator) and str(path).strip()
        }
    cfg = getattr(runtime, "config", None)
    if cfg is None:
        return {}
    return default_raw_paths_from_config(cfg)


def default_source_locator() -> ResidualLocator:
    return ResidualLocator(layer=0, site="block_output", phase="decode")


def default_target_locator() -> ResidualLocator:
    return ResidualLocator(layer=-1, site="block_output", phase="decode")


def raw_path_for_locator(
    raw_paths_by_locator: Mapping[ResidualLocator, str],
    locator: ResidualLocator,
) -> str:
    raw_path = str(raw_paths_by_locator.get(locator, "")).strip()
    if raw_path:
        return raw_path
    raise RuntimeError(
        f"unsupported logical residual locator for current runtime binding config: {locator}"
    )


def build_raw_tap_paths(
    *,
    raw_paths_by_locator: Mapping[ResidualLocator, str],
    required: set[tuple[int, str, str]],
) -> list[str]:
    paths: list[str] = []
    for locator in _supported_logical_locators(required):
        raw_path = raw_path_for_locator(raw_paths_by_locator, locator)
        if raw_path and raw_path not in paths:
            paths.append(raw_path)
    return paths


def build_resolved_bindings(
    *,
    raw_paths_by_locator: Mapping[ResidualLocator, str],
    resolve_alias: Mapping[str, str],
    required: set[tuple[int, str, str]],
) -> dict[str, ResidualPathBinding]:
    bindings: dict[str, ResidualPathBinding] = {}
    locators = _supported_logical_locators(required)
    for index, locator in enumerate(locators):
        raw_path = raw_path_for_locator(raw_paths_by_locator, locator)
        resolved_path = str(resolve_alias.get(raw_path, "")).strip()
        if not resolved_path:
            raise RuntimeError(f"required residual path was not resolved for locator {locator}: {raw_path}")
        bindings[resolved_path] = ResidualPathBinding(
            locator=locator,
            resolved_path=resolved_path,
            include_request_meta=(index == 0),
        )
    return bindings


def tap_paths(bindings: Mapping[str, ResidualPathBinding]) -> list[str]:
    return [binding.resolved_path for binding in bindings.values()]


def resolved_path_for_locator(
    bindings: Mapping[str, ResidualPathBinding],
    locator: ResidualLocator,
) -> str | None:
    for binding in bindings.values():
        if binding.locator == locator:
            return binding.resolved_path
    return None


def default_resolved_paths(runtime: object) -> tuple[str, str]:
    bindings = getattr(runtime, "residual_bindings", None)
    if isinstance(bindings, Mapping) and bindings:
        source = resolved_path_for_locator(bindings, default_source_locator()) or ""
        target = resolved_path_for_locator(bindings, default_target_locator()) or ""
        return source, target
    return (
        str(getattr(runtime, "source_resolved_path", "") or ""),
        str(getattr(runtime, "target_resolved_path", "") or ""),
    )
