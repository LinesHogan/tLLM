#!/usr/bin/env python3
"""ESamp parity guardrail plus single-path aligned workload verifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from tllm.runtime import residual_runtime as _runtime


JSON_SUMMARY_PREFIX = "ESAMP_LOSS_PARITY_JSON:"
REPO_ROOT = Path(__file__).resolve().parents[3]
PARITY_GUARDRAIL_DOC = REPO_ROOT / "doc" / "guides" / "validation.md"
REQUIRED_TRAINING_CASES = ("single_on", "per_request_on")
OPTIONAL_TRAINING_CASES = ("model_bank_on",)


def _build_result_summary(
    *,
    payload: Dict[str, object],
    guardrail_doc: str,
) -> Dict[str, object]:
    cases = payload.get("cases", {})
    if not isinstance(cases, dict):
        cases = {}
    doc_path = (REPO_ROOT / guardrail_doc).resolve() if not Path(guardrail_doc).is_absolute() else Path(guardrail_doc)

    def _has_updates(case_name: str) -> bool:
        case = cases.get(case_name, {})
        if not isinstance(case, dict):
            return False
        return float(case.get("loss_count", 0.0) or 0.0) > 0.0

    training_active_passed = all(_has_updates(name) for name in REQUIRED_TRAINING_CASES) and all(
        _has_updates(name) for name in OPTIONAL_TRAINING_CASES if name in cases
    )

    return {
        "parity_guardrail_doc": str(doc_path),
        "parity_guardrail_doc_available": bool(doc_path.is_file()),
        "training_active_passed": bool(training_active_passed),
        "cases": cases,
        "trajectory": payload.get("trajectory", {}),
    }


def _parse_args():
    from tllm.workflows.benchmarks import per_request_esamp_benchmark as benchmark

    return benchmark._parse_args()


def _run_aligned_single_path(args) -> Dict[str, object]:
    from tllm.workflows.benchmarks import per_request_esamp_benchmark as benchmark

    _ = _runtime.RUNTIME
    if getattr(args, "sampling_seed", None) is None:
        args.sampling_seed = 1234
    payload = benchmark._run_one_implementation(args, "esamp")
    summary = _build_result_summary(
        payload=payload,
        guardrail_doc=str(PARITY_GUARDRAIL_DOC.relative_to(REPO_ROOT)),
    )
    return summary


def main() -> int:
    args = _parse_args()
    summary = _run_aligned_single_path(args)
    print(JSON_SUMMARY_PREFIX + json.dumps(summary, sort_keys=True))
    return 0 if bool(summary["parity_guardrail_doc_available"]) and bool(summary["training_active_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
