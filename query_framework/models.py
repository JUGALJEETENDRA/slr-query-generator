from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class StrategyMetadata:
    id: str
    label: str
    description: str = ""
    aliases: tuple[str, ...] = ()
    experimental: bool = False


@dataclass
class QueryGenerationResult:
    question: str
    strategy_id: str
    strategy_label: str
    google_scholar: str
    scopus: str
    web_of_science: str
    ieee_xplore: str
    pubmed: str
    concepts: dict[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)

    def to_api_response(self) -> dict[str, Any]:
        return {
            "status": "success",
            "google_scholar": self.google_scholar,
            "scopus": self.scopus,
            "web_of_science": self.web_of_science,
            "ieee_xplore": self.ieee_xplore,
            "pubmed": self.pubmed,
            "concepts": self.concepts,
        }


@dataclass
class StrategyExecutionResult:
    result: QueryGenerationResult | None
    runtime_ms: float
    error: str | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)


class QueryGenerationStrategy(Protocol):
    metadata: StrategyMetadata

    def generate(self, question: str) -> QueryGenerationResult:
        ...


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    question: str
    suite: str = "default"
    expected_concepts: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    metric_id: str
    score: float | None
    passed: bool | None = None
    category: str = "quality"
    severity: str = "info"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegressionTest:
    id: str
    category: str
    description: str
    query_contains_all: tuple[str, ...] = ()
    query_contains_none: tuple[str, ...] = ()
    max_runtime_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionResult:
    test_id: str
    category: str
    passed: bool
    severity: str = "warning"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureAnalysis:
    categories: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BenchmarkRunResult:
    benchmark_id: str
    case: BenchmarkCase
    strategy_id: str
    strategy_label: str
    query: str | None
    runtime_ms: float
    evaluations: list[EvaluationResult] = field(default_factory=list)
    regressions: list[RegressionResult] = field(default_factory=list)
    failure_analysis: FailureAnalysis = field(default_factory=FailureAnalysis)
    telemetry: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "case": {
                "id": self.case.id,
                "question": self.case.question,
                "suite": self.case.suite,
                "expected_concepts": list(self.case.expected_concepts),
                "metadata": self.case.metadata,
            },
            "strategy_id": self.strategy_id,
            "strategy_label": self.strategy_label,
            "query": self.query,
            "runtime_ms": self.runtime_ms,
            "evaluations": [
                {
                    "metric_id": item.metric_id,
                    "score": item.score,
                    "passed": item.passed,
                    "category": item.category,
                    "severity": item.severity,
                    "details": item.details,
                }
                for item in self.evaluations
            ],
            "regressions": [
                {
                    "test_id": item.test_id,
                    "category": item.category,
                    "passed": item.passed,
                    "severity": item.severity,
                    "details": item.details,
                }
                for item in self.regressions
            ],
            "failure_analysis": {
                "categories": self.failure_analysis.categories,
                "warnings": self.failure_analysis.warnings,
                "details": self.failure_analysis.details,
            },
            "telemetry": self.telemetry,
            "error": self.error,
        }
