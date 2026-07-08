from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .models import BenchmarkRunResult


class BenchmarkReporter:
    def write_json(self, results: list[BenchmarkRunResult], path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps([result.to_dict() for result in results], indent=2),
            encoding="utf-8",
        )
        return output_path

    def write_markdown(
        self,
        results: list[BenchmarkRunResult],
        path: str | Path,
        comparison: dict | None = None,
    ) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        by_strategy: dict[str, list[BenchmarkRunResult]] = defaultdict(list)
        for result in results:
            by_strategy[result.strategy_label].append(result)

        lines = ["# LitSync Benchmark Report", ""]
        lines.append(f"Total executions: {len(results)}")
        lines.append("")
        lines.append("| Strategy | Runs | Errors | Failed Checks | Avg Runtime ms |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for strategy_label, items in by_strategy.items():
            errors = sum(1 for item in items if item.error)
            failed_checks = sum(
                sum(1 for evaluation in item.evaluations if evaluation.passed is False)
                + sum(1 for regression in item.regressions if not regression.passed)
                for item in items
            )
            avg_runtime = sum(item.runtime_ms for item in items) / len(items)
            lines.append(f"| {strategy_label} | {len(items)} | {errors} | {failed_checks} | {avg_runtime:.0f} |")

        if comparison:
            lines.append("")
            lines.append("## Regression Comparison")
            lines.append("")
            if not comparison.get("available"):
                lines.append("No previous result file was available for comparison.")
            else:
                lines.append(f"- Regressions detected: {len(comparison.get('regressions', []))}")
                lines.append(f"- Improvements detected: {len(comparison.get('improvements', []))}")

        lines.append("")
        lines.append("## Executions")
        lines.append("")
        for result in results:
            status = "ERROR" if result.error else "OK"
            lines.append(f"### {result.case.id} - {result.strategy_label} - {status}")
            lines.append("")
            lines.append(f"- Suite: {result.case.suite}")
            lines.append(f"- Runtime: {result.runtime_ms:.0f} ms")
            if result.failure_analysis.categories:
                categories = ", ".join(
                    f"{category}: {count}"
                    for category, count in sorted(result.failure_analysis.categories.items())
                )
                lines.append(f"- Failure categories: {categories}")
            if result.error:
                lines.append(f"- Error: `{result.error}`")
            elif result.query:
                lines.append("")
                lines.append("```text")
                lines.append(result.query)
                lines.append("```")
                failed_evaluations = [item for item in result.evaluations if item.passed is False]
                failed_regressions = [item for item in result.regressions if not item.passed]
                if failed_evaluations or failed_regressions:
                    lines.append("")
                    lines.append("Failed checks:")
                    for item in failed_evaluations:
                        lines.append(f"- {item.metric_id} ({item.category})")
                    for item in failed_regressions:
                        lines.append(f"- {item.test_id} ({item.category})")
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path
