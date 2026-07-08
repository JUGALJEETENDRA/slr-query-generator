from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    id: str
    description: str
    strategy_ids: tuple[str, ...]
    benchmark_suite: str = "default"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")
    metadata: dict[str, Any] = field(default_factory=dict)
