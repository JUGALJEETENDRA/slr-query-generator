from __future__ import annotations

import json
import re
import time

from pydantic import BaseModel

from litsync_app.screening.local.engine import GenerationResult
from litsync_app.screening.local.engine import LocalAIOutputError


def parse_structured_model_output(raw: str, schema: type[BaseModel] | None = None) -> dict:
    """Extract one JSON object from API JSON, fenced JSON, or light UI prose."""
    text = str(raw or "").strip().lstrip("\ufeff")
    if not text:
        raise LocalAIOutputError("Gemini Web returned an empty response. Please retry the job.")

    candidates = [text]
    candidates.extend(
        block.strip()
        for block in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if block.strip()
    )

    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
            candidates.append(json.dumps(value))
        except json.JSONDecodeError:
            continue

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            if schema is None:
                return value
            try:
                schema.model_validate(value)
                return value
            except Exception:
                continue

    raise LocalAIOutputError(
        "Gemini Web answered, but the response was not valid structured JSON. "
        "LitSync will retry once with a fresh instruction."
    )


class InjectedStructuredEngine:
    """Structured-output adapter for explicitly enabled external engines."""

    def __init__(self, inference_engine):
        self.inference_engine = inference_engine

    def generate(self, model: str, prompt: str, schema: type[BaseModel]) -> GenerationResult:
        started = time.perf_counter()
        if hasattr(self.inference_engine, "ask_structured"):
            raw = self.inference_engine.ask_structured(prompt, schema=schema, model=model)
        else:
            raw = self.inference_engine.ask(prompt, model=model)
        return GenerationResult(
            value=parse_structured_model_output(raw, schema), model=model,
            elapsed_seconds=round(time.perf_counter() - started, 4),
        )

    def unload(self, model: str) -> None:
        return None
