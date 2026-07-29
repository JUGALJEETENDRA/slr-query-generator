from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .comparison import compare_results
from .contracts import BenchmarkVerdict
from .errors import BenchmarkError
from .evaluator import evaluate_run, validate_metric_definitions
from .loader import load_gold, load_run, load_spec
from .registry import check_registry, publish_completed_evaluation
from .report import (
    publish_staged_report,
    stage_comparison_report,
    stage_result_report,
)


def _common(command: argparse.ArgumentParser, *, output: bool = False) -> None:
    command.add_argument("--spec", required=True)
    command.add_argument("--artifacts-root", default="outputs")
    command.add_argument("--registry-dir", default="outputs/benchmarks/_registry")
    command.add_argument("--json", action="store_true", dest="as_json")
    if output:
        command.add_argument("--output-dir", required=True)
        command.add_argument("--enforce-gate", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LitSync offline screening benchmark and release gate"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-spec")
    _common(validate)
    evaluate = commands.add_parser("evaluate")
    _common(evaluate, output=True)
    evaluate.add_argument("--job-id", required=True)
    compare = commands.add_parser("compare")
    _common(compare, output=True)
    compare.add_argument("--job-id", action="append", required=True)
    return parser


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


def _validate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    loaded = load_spec(args.spec)
    validate_metric_definitions(loaded.spec.metric_definitions)
    gold = load_gold(loaded)
    check_registry(loaded.spec, args.registry_dir)
    return {
        "status": "valid",
        "benchmark_id": loaded.spec.benchmark_id,
        "benchmark_version": loaded.spec.benchmark_version,
        "benchmark_spec_fingerprint": loaded.spec.benchmark_spec_fingerprint,
        "gold_file_fingerprint": gold.file_fingerprint,
        "run_population": len(loaded.spec.run_selected_source_row_ids),
        "gold_population": len(loaded.spec.gold_selected_source_row_ids),
        "registry_mutated": False,
    }, 0


def _evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    loaded = load_spec(args.spec)
    gold = load_gold(loaded)
    check_registry(loaded.spec, args.registry_dir, allow_pending_retry=True)
    run = load_run(loaded, gold, args.job_id, args.artifacts_root)
    result = evaluate_run(loaded, gold, run)
    staged = stage_result_report(result, args.output_dir)
    if result.gate.verdict == BenchmarkVerdict.INVALID:
        paths = publish_staged_report(
            staged,
            args.output_dir,
            publication_kind="evaluation",
            benchmark_id=result.benchmark_id,
            benchmark_version=result.benchmark_version,
            benchmark_spec_fingerprint=result.benchmark_spec_fingerprint,
            job_ids=[result.job_id],
            verdict=result.gate.verdict.value,
        )
    else:
        paths = publish_completed_evaluation(
            loaded.spec,
            result.gate.verdict,
            staged,
            args.output_dir,
            args.registry_dir,
            result_key="result",
            publication_kind="evaluation",
            job_ids=[result.job_id],
        )
    payload = {
        "status": "complete",
        "job_id": result.job_id,
        "verdict": result.gate.verdict.value,
        "provenance": result.provenance.classification.value,
        "result": str(paths["result"]),
        "report": str(paths["html"]),
        "errors": str(paths["errors"]),
    }
    if result.gate.verdict == BenchmarkVerdict.INVALID:
        return payload, 2
    if args.enforce_gate and result.gate.verdict != BenchmarkVerdict.PASS:
        return payload, 3
    return payload, 0


def _compare(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if len(args.job_id) < 2:
        raise BenchmarkError("compare requires at least two --job-id values")
    loaded = load_spec(args.spec)
    gold = load_gold(loaded)
    check_registry(loaded.spec, args.registry_dir, allow_pending_retry=True)
    results = [
        evaluate_run(
            loaded,
            gold,
            load_run(loaded, gold, job_id, args.artifacts_root),
        )
        for job_id in args.job_id
    ]
    comparison = compare_results(results)
    staged = stage_comparison_report(comparison, args.output_dir)
    valid_results = all(
        result.gate.verdict != BenchmarkVerdict.INVALID for result in results
    )
    if comparison.valid and valid_results:
        paths = publish_completed_evaluation(
            loaded.spec,
            "COMPARISON_VALID",
            staged,
            args.output_dir,
            args.registry_dir,
            result_key="comparison",
            publication_kind="comparison",
            job_ids=comparison.job_ids,
        )
    else:
        paths = publish_staged_report(
            staged,
            args.output_dir,
            publication_kind="comparison",
            benchmark_id=comparison.benchmark_id,
            benchmark_version=comparison.benchmark_version,
            benchmark_spec_fingerprint=comparison.benchmark_spec_fingerprint,
            job_ids=comparison.job_ids,
            verdict="INVALID",
        )
    payload = {
        "status": "complete" if comparison.valid else "invalid",
        "valid": comparison.valid,
        "job_ids": comparison.job_ids,
        "comparison": str(paths["comparison"]),
        "report": str(paths["html"]),
        "errors": str(paths["errors"]),
    }
    if not comparison.valid or not valid_results:
        return payload, 2
    if args.enforce_gate and any(
        result.gate.verdict != BenchmarkVerdict.PASS for result in results
    ):
        return payload, 3
    return payload, 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-spec":
            payload, code = _validate(args)
        elif args.command == "evaluate":
            payload, code = _evaluate(args)
        else:
            payload, code = _compare(args)
    except (BenchmarkError, ValueError) as exc:
        _emit({"status": "invalid", "error": str(exc)}, getattr(args, "as_json", False))
        return 2
    _emit(payload, args.as_json)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
