import argparse
import json
import sys
from pathlib import Path

import instructor
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_MODEL
from query_framework import (
    BenchmarkCase,
    BenchmarkReporter,
    BenchmarkRunner,
    EvaluationSuite,
    RegressionSuite,
    compare_against_previous,
    create_default_strategy_registry,
    discover_regression_tests,
    load_previous_results,
)


BASE_DIR = Path(__file__).parent
QUESTION_FILE = BASE_DIR / "capability_benchmark_questions.json"
RESULT_DIR = BASE_DIR / "results"


def load_cases(path: Path, suite: str | None = None, limit: int | None = None) -> list[BenchmarkCase]:
    with open(path, "r", encoding="utf-8") as file:
        raw_cases = json.load(file)

    cases = []
    for index, item in enumerate(raw_cases, start=1):
        if suite and item.get("suite") != suite:
            continue
        cases.append(
            BenchmarkCase(
                id=str(item.get("id") or f"case-{index}"),
                question=item["question"],
                suite=item.get("suite", "default"),
                expected_concepts=tuple(item.get("expected_concepts", ())),
                metadata={key: value for key, value in item.items() if key not in {"id", "question", "suite", "expected_concepts"}},
            )
        )
        if limit and len(cases) >= limit:
            break

    return cases


def build_registry():
    client = instructor.from_openai(
        OpenAI(base_url="http://localhost:11434/v1", api_key="ollama-local"),
        mode=instructor.Mode.MD_JSON,
    )
    return create_default_strategy_registry(client, DEFAULT_MODEL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run registered LitSync query-generation strategies.")
    parser.add_argument("--suite", default="Prediction", help="Benchmark suite to run. Use an empty string for all suites.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of benchmark cases.")
    parser.add_argument("--strategy", action="append", dest="strategies", help="Strategy id to run. Repeat for multiple strategies.")
    parser.add_argument("--questions", type=Path, default=QUESTION_FILE, help="Path to benchmark question JSON.")
    parser.add_argument("--previous-results", type=Path, default=None, help="Optional previous JSON result file for regression comparison.")
    args = parser.parse_args()

    suite = args.suite or None
    cases = load_cases(args.questions, suite=suite, limit=args.limit)
    registry = build_registry()
    runner = BenchmarkRunner(
        registry=registry,
        evaluator=EvaluationSuite(),
        regression_suite=RegressionSuite(discover_regression_tests()),
        benchmark_id="query_generation_phase1",
    )

    results = runner.run(cases, strategy_ids=args.strategies)
    comparison = compare_against_previous(results, load_previous_results(args.previous_results))
    reporter = BenchmarkReporter()
    json_path = reporter.write_json(results, RESULT_DIR / "benchmark_results.json")
    markdown_path = reporter.write_markdown(results, RESULT_DIR / "benchmark_summary.md", comparison=comparison)

    print(f"Cases: {len(cases)}")
    print(f"Executions: {len(results)}")
    print(f"JSON: {json_path}")
    print(f"Summary: {markdown_path}")


if __name__ == "__main__":
    main()
