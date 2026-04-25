#!/usr/bin/env python3
"""DEPRECATED compatibility entrypoint for legacy decode-row verification."""

from __future__ import annotations

import sys

from tllm.legacy.verify_v1_decode_rows_main import main


def emit_deprecation_notice(*, old_module: str, new_module: str) -> None:
    print(
        f"[DEPRECATED] `{old_module}` has moved to `{new_module}`; "
        "please update scripts/imports.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    emit_deprecation_notice(
        old_module="tllm.legacy.verify_v1_decode_rows",
        new_module="verify_v1_decode_rows_minimal",
    )
    raise SystemExit(main())
