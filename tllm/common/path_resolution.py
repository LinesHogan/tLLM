#!/usr/bin/env python3
"""Path and layer-container resolution helpers for model introspection."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple


def parse_component(component: str) -> tuple[str, List[int]]:
    if not component:
        raise RuntimeError("Empty path component is not allowed.")

    attr = component
    bracket_pos = component.find("[")
    if bracket_pos >= 0:
        attr = component[:bracket_pos]

    if not attr:
        raise RuntimeError(f"Invalid path component `{component}`: missing attribute name.")

    indices: List[int] = []
    rest = component[len(attr) :]
    while rest:
        if not rest.startswith("["):
            raise RuntimeError(f"Invalid path component `{component}` near `{rest}`.")
        close = rest.find("]")
        if close < 0:
            raise RuntimeError(f"Invalid path component `{component}`: missing `]`.")
        idx_text = rest[1:close].strip()
        if not idx_text or not idx_text.lstrip("-").isdigit():
            raise RuntimeError(f"Invalid index `{idx_text}` in path component `{component}`.")
        indices.append(int(idx_text))
        rest = rest[close + 1 :]
    return attr, indices


def resolve_object_by_path(root: Any, path: str) -> Any:
    """Resolve object from a dotted path with optional list indices."""
    stripped = str(path).strip()
    if not stripped:
        raise RuntimeError("capture layer path must be non-empty.")

    obj = root
    for component in stripped.split("."):
        attr, indices = parse_component(component)
        if not hasattr(obj, attr):
            raise RuntimeError(
                f"Capture layer path `{stripped}` is invalid: `{attr}` not found on `{type(obj).__name__}`."
            )
        obj = getattr(obj, attr)
        for idx in indices:
            try:
                obj = obj[idx]
            except Exception as e:
                raise RuntimeError(
                    f"Capture layer path `{stripped}` is invalid: cannot index `{attr}[{idx}]`."
                ) from e
    return obj


def candidate_capture_paths(path: str) -> List[str]:
    """Build fallback candidates by stripping leading `model.` prefixes."""
    stripped = str(path).strip()
    if not stripped:
        return []
    candidates = [stripped]
    cur = stripped
    while cur.startswith("model."):
        cur = cur[len("model.") :]
        if cur and cur not in candidates:
            candidates.append(cur)
    return candidates


def resolve_layers_container(model: Any) -> Tuple[Sequence[Any], str]:
    """Resolve a layer list-like container across common model wrappers."""
    cur = model
    prefix_parts: List[str] = []
    for _ in range(8):
        direct_layers = getattr(cur, "layers", None)
        if direct_layers is not None and len(direct_layers) > 0:
            prefix = ".".join(prefix_parts + ["layers"]) if prefix_parts else "layers"
            return direct_layers, prefix

        decoder = getattr(cur, "decoder", None)
        decoder_layers = getattr(decoder, "layers", None) if decoder is not None else None
        if decoder_layers is not None and len(decoder_layers) > 0:
            prefix = ".".join(prefix_parts + ["decoder", "layers"]) if prefix_parts else "decoder.layers"
            return decoder_layers, prefix

        nxt = getattr(cur, "model", None)
        if nxt is None:
            break
        prefix_parts.append("model")
        cur = nxt

    raise RuntimeError(
        "Cannot resolve layer container from model root. Tried nested "
        "`*.layers` and `*.decoder.layers` along `.model` chain."
    )
