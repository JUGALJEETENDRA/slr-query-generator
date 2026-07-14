from __future__ import annotations

import json
import time

from pydantic import BaseModel

from local_ai.engine import GenerationResult


class InjectedStructuredEngine:
    """Structured-output adapter for explicitly enabled external engines."""

    def __init__(self, inference_engine):
        self.inference_engine = inference_engine

    def generate(self, model: str, prompt: str, schema: type[BaseModel]) -> GenerationResult:
        started = time.perf_counter()
        raw = self.inference_engine.ask(prompt, model=model)
        return GenerationResult(
            value=json.loads(raw), model=model,
            elapsed_seconds=round(time.perf_counter() - started, 4),
        )

    def unload(self, model: str) -> None:
        return None
