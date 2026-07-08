from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterator


@dataclass
class TelemetryCollector:
    stages: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def record_stage(self, name: str, payload: Any = None, runtime_ms: float | None = None) -> None:
        stage = {"name": name}
        if payload is not None:
            stage["payload"] = _serialize_payload(payload)
        if runtime_ms is not None:
            stage["runtime_ms"] = runtime_ms
        self.stages.append(stage)

    @contextmanager
    def timed_stage(self, name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.record_stage(name, runtime_ms=(perf_counter() - start) * 1000)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": self.stages,
            "warnings": self.warnings,
        }


def _serialize_payload(payload: Any) -> Any:
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    if hasattr(payload, "dict"):
        return payload.dict()
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        return payload
    if isinstance(payload, (list, tuple)):
        return [_serialize_payload(item) for item in payload]
    if isinstance(payload, dict):
        return {str(key): _serialize_payload(value) for key, value in payload.items()}
    return str(payload)
