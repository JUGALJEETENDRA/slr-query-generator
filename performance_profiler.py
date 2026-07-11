from __future__ import annotations

import json
import os
import statistics
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any


class PerformanceProfiler:
    def __init__(self, enabled: bool = True, output_dir: str = "outputs"):
        self.enabled = enabled
        self.output_dir = output_dir
        self.started_at = time.perf_counter()
        self.components: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.row_seconds: list[float] = []
        self.extra: dict[str, Any] = {}

    @contextmanager
    def measure(self, component: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add_time(component, time.perf_counter() - started)

    def add_time(self, component: str, seconds: float) -> None:
        if not self.enabled:
            return
        self.components[component] = self.components.get(component, 0.0) + float(seconds or 0.0)

    def increment(self, name: str, amount: int = 1) -> None:
        if not self.enabled:
            return
        self.counts[name] = self.counts.get(name, 0) + int(amount)

    def add_row_seconds(self, seconds: float) -> None:
        if self.enabled:
            self.row_seconds.append(float(seconds or 0.0))

    def update_extra(self, **kwargs) -> None:
        if self.enabled:
            self.extra.update(kwargs)

    def summary(self) -> dict[str, Any]:
        total_runtime = time.perf_counter() - self.started_at
        rows = self.row_seconds
        return {
            **self.extra,
            "total_runtime_seconds": round(total_runtime, 4),
            "total_rows": len(rows),
            "average_per_row_seconds": round(sum(rows) / len(rows), 4) if rows else 0.0,
            "p50_per_row_seconds": _quantile(rows, 0.50),
            "p90_per_row_seconds": _quantile(rows, 0.90),
            "p95_per_row_seconds": _quantile(rows, 0.95),
            "total_time_by_component": {
                key: round(value, 4) for key, value in sorted(self.components.items())
            },
            "counts": dict(sorted(self.counts.items())),
        }

    def write(self) -> str:
        if not self.enabled:
            return ""
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"performance_profile_{timestamp}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2)
        return path


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 4)
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return round(ordered[index], 4)
