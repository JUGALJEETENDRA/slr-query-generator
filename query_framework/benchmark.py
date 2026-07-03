from __future__ import annotations

from time import perf_counter

from .evaluation import EvaluationSuite
from .failure_analysis import analyze_failures
from .models import BenchmarkCase, BenchmarkRunResult
from .regression import RegressionSuite
from .registry import StrategyRegistry


class BenchmarkRunner:
    def __init__(
        self,
        registry: StrategyRegistry,
        evaluator: EvaluationSuite | None = None,
        regression_suite: RegressionSuite | None = None,
        benchmark_id: str = "query_generation",
    ):
        self.registry = registry
        self.evaluator = evaluator or EvaluationSuite()
        self.regression_suite = regression_suite or RegressionSuite()
        self.benchmark_id = benchmark_id

    def run(
        self,
        cases: list[BenchmarkCase],
        strategy_ids: list[str] | None = None,
    ) -> list[BenchmarkRunResult]:
        selected = strategy_ids or [metadata.id for metadata in self.registry.list_metadata()]
        results: list[BenchmarkRunResult] = []

        for case in cases:
            for strategy_id in selected:
                strategy = self.registry.get(strategy_id)
                start = perf_counter()
                try:
                    generation = strategy.generate(case.question)
                    runtime_ms = (perf_counter() - start) * 1000
                    generation.telemetry["runtime_ms"] = runtime_ms
                    evaluations = self.evaluator.evaluate(case, generation)
                    regressions = self.regression_suite.evaluate(case, generation, runtime_ms)
                    failure_analysis = analyze_failures(evaluations, regressions)
                    results.append(
                        BenchmarkRunResult(
                            benchmark_id=self.benchmark_id,
                            case=case,
                            strategy_id=generation.strategy_id,
                            strategy_label=generation.strategy_label,
                            query=generation.google_scholar,
                            runtime_ms=runtime_ms,
                            evaluations=evaluations,
                            regressions=regressions,
                            failure_analysis=failure_analysis,
                            telemetry=generation.telemetry,
                        )
                    )
                except Exception as exc:
                    runtime_ms = (perf_counter() - start) * 1000
                    failure_analysis = analyze_failures([], [], error=str(exc))
                    results.append(
                        BenchmarkRunResult(
                            benchmark_id=self.benchmark_id,
                            case=case,
                            strategy_id=strategy.metadata.id,
                            strategy_label=strategy.metadata.label,
                            query=None,
                            runtime_ms=runtime_ms,
                            failure_analysis=failure_analysis,
                            error=str(exc),
                        )
                    )

        return results
