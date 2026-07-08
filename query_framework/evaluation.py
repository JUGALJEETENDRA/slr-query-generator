from __future__ import annotations

import re
from importlib import import_module
from pkgutil import iter_modules
from typing import Protocol

from .models import BenchmarkCase, EvaluationResult, QueryGenerationResult


class EvaluationMetric(Protocol):
    id: str

    def evaluate(self, case: BenchmarkCase, result: QueryGenerationResult) -> EvaluationResult:
        ...


class BooleanBalanceMetric:
    id = "boolean_balance"

    def evaluate(self, case: BenchmarkCase, result: QueryGenerationResult) -> EvaluationResult:
        query = result.google_scholar
        balanced = query.count("(") == query.count(")") and query.count('"') % 2 == 0
        return EvaluationResult(
            metric_id=self.id,
            score=1.0 if balanced else 0.0,
            passed=balanced,
            category="boolean_validity",
            severity="error" if not balanced else "info",
            details={
                "open_parentheses": query.count("("),
                "close_parentheses": query.count(")"),
                "quote_count": query.count('"'),
            },
        )


class QueryComplexityMetric:
    id = "query_complexity"

    def evaluate(self, case: BenchmarkCase, result: QueryGenerationResult) -> EvaluationResult:
        query = result.google_scholar
        terms = re.findall(r'"([^"]+)"', query)
        and_count = len(re.findall(r"\bAND\b", query))
        or_count = len(re.findall(r"\bOR\b", query))
        return EvaluationResult(
            metric_id=self.id,
            score=float(len(terms)),
            passed=None,
            category="query_complexity",
            details={
                "quoted_terms": len(terms),
                "and_count": and_count,
                "or_count": or_count,
                "character_count": len(query),
            },
        )


class ExpectedConceptCoverageMetric:
    id = "expected_concept_coverage"

    def evaluate(self, case: BenchmarkCase, result: QueryGenerationResult) -> EvaluationResult:
        expected = [concept.lower() for concept in case.expected_concepts]
        query = result.google_scholar.lower()
        missing = [concept for concept in expected if concept not in query]
        total = len(expected)
        score = None if total == 0 else (total - len(missing)) / total
        return EvaluationResult(
            metric_id=self.id,
            score=score,
            passed=None if total == 0 else not missing,
            category="concept_preservation",
            severity="error" if missing else "info",
            details={
                "expected": expected,
                "missing": missing,
            },
        )


class ComparatorPreservationMetric:
    id = "comparator_preservation"

    def evaluate(self, case: BenchmarkCase, result: QueryGenerationResult) -> EvaluationResult:
        expected = _metadata_terms(case, "expected_comparators")
        return _coverage_result(self.id, "comparator_preservation", expected, result.google_scholar)


class ApplicationDomainPreservationMetric:
    id = "application_domain_preservation"

    def evaluate(self, case: BenchmarkCase, result: QueryGenerationResult) -> EvaluationResult:
        expected = _metadata_terms(case, "expected_domains")
        return _coverage_result(self.id, "application_domain_preservation", expected, result.google_scholar)


class OutcomePreservationMetric:
    id = "outcome_preservation"

    def evaluate(self, case: BenchmarkCase, result: QueryGenerationResult) -> EvaluationResult:
        expected = _metadata_terms(case, "expected_outcomes")
        return _coverage_result(self.id, "outcome_preservation", expected, result.google_scholar)


class HallucinatedConceptMetric:
    id = "hallucinated_concepts"

    def evaluate(self, case: BenchmarkCase, result: QueryGenerationResult) -> EvaluationResult:
        forbidden = _metadata_terms(case, "forbidden_concepts")
        query = result.google_scholar.lower()
        found = [term for term in forbidden if term.lower() in query]
        return EvaluationResult(
            metric_id=self.id,
            score=0.0 if found else 1.0,
            passed=not found,
            category="hallucination",
            severity="error" if found else "info",
            details={"forbidden": forbidden, "found": found},
        )


class DuplicateTermMetric:
    id = "duplicate_terms"

    def evaluate(self, case: BenchmarkCase, result: QueryGenerationResult) -> EvaluationResult:
        terms = [_normalize_term(term) for term in re.findall(r'"([^"]+)"', result.google_scholar)]
        duplicates = sorted({term for term in terms if terms.count(term) > 1})
        return EvaluationResult(
            metric_id=self.id,
            score=float(len(duplicates)),
            passed=not duplicates,
            category="redundancy",
            severity="warning" if duplicates else "info",
            details={"duplicates": duplicates},
        )


class RedundancyMetric:
    id = "redundancy_detection"

    def evaluate(self, case: BenchmarkCase, result: QueryGenerationResult) -> EvaluationResult:
        terms = [_normalize_term(term) for term in re.findall(r'"([^"]+)"', result.google_scholar)]
        redundant = []
        for left in terms:
            for right in terms:
                if left != right and left in right:
                    redundant.append({"shorter": left, "longer": right})
        return EvaluationResult(
            metric_id=self.id,
            score=float(len(redundant)),
            passed=not redundant,
            category="redundancy",
            severity="warning" if redundant else "info",
            details={"redundant_pairs": redundant[:20]},
        )


class RuntimeMetric:
    id = "runtime_ms"

    def evaluate(self, case: BenchmarkCase, result: QueryGenerationResult) -> EvaluationResult:
        runtime = result.telemetry.get("runtime_ms")
        return EvaluationResult(
            metric_id=self.id,
            score=float(runtime) if runtime is not None else None,
            passed=None,
            category="runtime",
            details={"runtime_ms": runtime},
        )


def _metadata_terms(case: BenchmarkCase, key: str) -> list[str]:
    value = case.metadata.get(key, ())
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _normalize_term(term: str) -> str:
    return re.sub(r"\s+", " ", term.lower().strip())


def _coverage_result(metric_id: str, category: str, expected: list[str], query: str) -> EvaluationResult:
    normalized_query = query.lower()
    missing = [term for term in expected if term.lower() not in normalized_query]
    total = len(expected)
    score = None if total == 0 else (total - len(missing)) / total
    return EvaluationResult(
        metric_id=metric_id,
        score=score,
        passed=None if total == 0 else not missing,
        category=category,
        severity="error" if missing else "info",
        details={"expected": expected, "missing": missing},
    )


def discover_evaluation_metrics(package_name: str = "query_framework.evaluation_modules") -> list[EvaluationMetric]:
    try:
        package = import_module(package_name)
    except ModuleNotFoundError:
        return []

    discovered: list[EvaluationMetric] = []
    for module_info in iter_modules(package.__path__):
        module = import_module(f"{package_name}.{module_info.name}")
        for value in vars(module).values():
            if isinstance(value, type) and hasattr(value, "evaluate") and hasattr(value, "id"):
                discovered.append(value())
    return discovered


class EvaluationSuite:
    def __init__(self, metrics: list[EvaluationMetric] | None = None):
        self.metrics = metrics or [
            BooleanBalanceMetric(),
            QueryComplexityMetric(),
            ExpectedConceptCoverageMetric(),
            ComparatorPreservationMetric(),
            ApplicationDomainPreservationMetric(),
            OutcomePreservationMetric(),
            HallucinatedConceptMetric(),
            DuplicateTermMetric(),
            RedundancyMetric(),
            RuntimeMetric(),
            *discover_evaluation_metrics(),
        ]

    def evaluate(self, case: BenchmarkCase, result: QueryGenerationResult) -> list[EvaluationResult]:
        return [metric.evaluate(case, result) for metric in self.metrics]
