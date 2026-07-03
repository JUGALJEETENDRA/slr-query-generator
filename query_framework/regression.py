from __future__ import annotations

import json
from pathlib import Path

from .models import BenchmarkCase, QueryGenerationResult, RegressionResult, RegressionTest


DEFAULT_REGRESSION_CATEGORIES = (
    "missing_application_domains",
    "comparator_loss",
    "missing_outcomes",
    "semantic_generalization",
    "hallucinated_concepts",
    "ontology_contamination",
    "cross_domain_contamination",
    "acronym_failures",
    "duplicate_expansions",
    "boolean_syntax_failures",
    "incorrect_operator_usage",
    "generic_parent_replacement",
    "excessive_synonym_expansion",
    "under_expansion",
    "empty_facet_generation",
    "runtime_regression",
)


class RegressionSuite:
    def __init__(self, tests: list[RegressionTest] | None = None):
        self.tests = tests or default_regression_tests()

    def evaluate(
        self,
        case: BenchmarkCase,
        result: QueryGenerationResult,
        runtime_ms: float,
    ) -> list[RegressionResult]:
        return [_evaluate_test(test, case, result, runtime_ms) for test in self.tests]


def default_regression_tests() -> list[RegressionTest]:
    return [
        RegressionTest(
            id=f"{category}_sentinel",
            category=category,
            description=f"Permanent sentinel for {category.replace('_', ' ')}.",
            metadata={"sentinel": True},
        )
        for category in DEFAULT_REGRESSION_CATEGORIES
    ]


def discover_regression_tests(directory: str | Path = "benchmark/query_generator/regressions") -> list[RegressionTest]:
    root = Path(directory)
    if not root.exists():
        return default_regression_tests()

    tests = default_regression_tests()
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("tests", [])
        for item in items:
            tests.append(
                RegressionTest(
                    id=item["id"],
                    category=item["category"],
                    description=item.get("description", ""),
                    query_contains_all=tuple(item.get("query_contains_all", ())),
                    query_contains_none=tuple(item.get("query_contains_none", ())),
                    max_runtime_ms=item.get("max_runtime_ms"),
                    metadata={key: value for key, value in item.items() if key not in {
                        "id",
                        "category",
                        "description",
                        "query_contains_all",
                        "query_contains_none",
                        "max_runtime_ms",
                    }},
                )
            )
    return tests


def _evaluate_test(
    test: RegressionTest,
    case: BenchmarkCase,
    result: QueryGenerationResult,
    runtime_ms: float,
) -> RegressionResult:
    query = result.google_scholar.lower()
    missing = [term for term in test.query_contains_all if term.lower() not in query]
    forbidden = [term for term in test.query_contains_none if term.lower() in query]
    runtime_failed = test.max_runtime_ms is not None and runtime_ms > test.max_runtime_ms
    passed = not missing and not forbidden and not runtime_failed
    return RegressionResult(
        test_id=test.id,
        category=test.category,
        passed=passed,
        severity="error" if not passed else "info",
        details={
            "case_id": case.id,
            "missing_required_terms": missing,
            "found_forbidden_terms": forbidden,
            "runtime_ms": runtime_ms,
            "max_runtime_ms": test.max_runtime_ms,
        },
    )
