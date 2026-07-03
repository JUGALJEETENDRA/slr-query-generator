from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from .models import BenchmarkRunResult


def load_previous_results(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    result_path = Path(path)
    if not result_path.exists():
        return []
    return json.loads(result_path.read_text(encoding="utf-8"))


def compare_against_previous(
    current: list[BenchmarkRunResult],
    previous: list[dict[str, Any]],
) -> dict[str, Any]:
    if not previous:
        return {"available": False, "regressions": [], "improvements": []}

    previous_by_strategy = _summarize_previous(previous)
    current_by_strategy = _summarize_current(current)
    regressions = []
    improvements = []

    for strategy_id, current_summary in current_by_strategy.items():
        old_summary = previous_by_strategy.get(strategy_id)
        if not old_summary:
            continue
        runtime_delta = current_summary["avg_runtime_ms"] - old_summary["avg_runtime_ms"]
        failure_delta = current_summary["failed_checks"] - old_summary["failed_checks"]
        record = {
            "strategy_id": strategy_id,
            "runtime_delta_ms": runtime_delta,
            "failure_delta": failure_delta,
            "current": current_summary,
            "previous": old_summary,
        }
        if runtime_delta > max(1000.0, old_summary["avg_runtime_ms"] * 0.2) or failure_delta > 0:
            regressions.append(record)
        elif runtime_delta < 0 or failure_delta < 0:
            improvements.append(record)

    return {
        "available": True,
        "regressions": regressions,
        "improvements": improvements,
    }


def _summarize_current(results: list[BenchmarkRunResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[BenchmarkRunResult]] = {}
    for result in results:
        grouped.setdefault(result.strategy_id, []).append(result)
    return {
        strategy_id: {
            "avg_runtime_ms": mean(item.runtime_ms for item in items),
            "failed_checks": sum(
                sum(1 for evaluation in item.evaluations if evaluation.passed is False)
                + sum(1 for regression in item.regressions if not regression.passed)
                + (1 if item.error else 0)
                for item in items
            ),
        }
        for strategy_id, items in grouped.items()
    }


def _summarize_previous(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["strategy_id"], []).append(result)
    return {
        strategy_id: {
            "avg_runtime_ms": mean(item.get("runtime_ms", 0.0) for item in items),
            "failed_checks": sum(
                sum(1 for evaluation in item.get("evaluations", []) if evaluation.get("passed") is False)
                + sum(1 for regression in item.get("regressions", []) if regression.get("passed") is False)
                + (1 if item.get("error") else 0)
                for item in items
            ),
        }
        for strategy_id, items in grouped.items()
    }
