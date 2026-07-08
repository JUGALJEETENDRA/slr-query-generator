from __future__ import annotations

from collections import Counter

from .models import EvaluationResult, FailureAnalysis, RegressionResult


def analyze_failures(
    evaluations: list[EvaluationResult],
    regressions: list[RegressionResult],
    error: str | None = None,
) -> FailureAnalysis:
    categories: Counter[str] = Counter()
    warnings: list[str] = []
    details: list[dict] = []

    if error:
        categories["execution_error"] += 1
        warnings.append(error)
        details.append({"category": "execution_error", "message": error})

    for item in evaluations:
        if item.passed is False:
            categories[item.category] += 1
            warnings.append(f"{item.metric_id} failed")
            details.append({
                "source": "evaluation",
                "category": item.category,
                "id": item.metric_id,
                "severity": item.severity,
                "details": item.details,
            })

    for item in regressions:
        if not item.passed:
            categories[item.category] += 1
            warnings.append(f"{item.test_id} failed")
            details.append({
                "source": "regression",
                "category": item.category,
                "id": item.test_id,
                "severity": item.severity,
                "details": item.details,
            })

    return FailureAnalysis(
        categories=dict(categories),
        warnings=warnings,
        details=details,
    )
