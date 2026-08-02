"""Structured query-draft engines that are not part of the local AI stack."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from litsync_app.integrations.gemini_web_v24_automation import GeminiWebV24Automation
from litsync_app.screening.local.engine import (
    GenerationResult,
    LocalAIError,
    LocalAIOutputError,
)


_JSON_FENCE_RE = re.compile(
    r"\A```(?:json)?\s*\n?(.*?)\n?```\s*\Z", re.IGNORECASE | re.DOTALL,
)


class _MalformedGeminiJSON(ValueError):
    pass


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        match = _JSON_FENCE_RE.fullmatch(text)
        if match is None:
            raise _MalformedGeminiJSON("Gemini returned an invalid JSON Markdown fence.")
        text = match.group(1).strip()
    elif "```" in text:
        raise _MalformedGeminiJSON("Gemini returned text outside a JSON Markdown fence.")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _MalformedGeminiJSON(str(exc)) from exc
    if not isinstance(value, dict):
        raise _MalformedGeminiJSON("Gemini must return one JSON object.")
    return value


class GeminiWebQueryEngine:
    """Small Gemini Web adapter for the existing structured query generator."""

    engine_id = "gemini_web_v24"

    def __init__(
        self,
        browser_factory: Callable[[], Any] = GeminiWebV24Automation,
    ) -> None:
        self.browser_factory = browser_factory

    def generate(
        self,
        model: str,
        prompt: str,
        schema: type[BaseModel],
        *,
        timeout_seconds: float | None = None,
    ) -> GenerationResult:
        del timeout_seconds  # GeminiWebV24Automation owns its browser timeouts.
        started_at = time.perf_counter()
        request = (
            f"{prompt}\n\n{schema.__name__} JSON schema:\n"
            f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
        )
        last_error: LocalAIError | None = None

        for attempt in range(2):
            browser = self.browser_factory()
            browser_started = False
            parsed: dict[str, Any] | None = None
            attempt_error: LocalAIError | None = None
            try:
                browser.start()
                browser_started = True
                raw = browser.submit_prompt_and_get_response(request)
                value = _parse_json_object(raw)
                try:
                    parsed = schema.model_validate(value).model_dump(mode="json")
                except ValidationError as exc:
                    raise LocalAIOutputError(
                        f"Gemini returned invalid {schema.__name__} data: {exc}"
                    ) from exc
            except LocalAIOutputError:
                raise
            except _MalformedGeminiJSON as exc:
                attempt_error = LocalAIOutputError(f"Gemini returned malformed JSON: {exc}")
            except (TimeoutError, RuntimeError) as exc:
                attempt_error = LocalAIError(f"Gemini Web query generation failed: {exc}")
            except Exception as exc:
                attempt_error = LocalAIError("Gemini Web browser interaction failed.")
                attempt_error.__cause__ = exc
            finally:
                if browser_started:
                    try:
                        browser.close()
                    except Exception as exc:
                        if parsed is None and attempt_error is None:
                            attempt_error = LocalAIError(
                                f"Gemini Web browser cleanup failed: {exc}"
                            )

            if parsed is not None:
                elapsed = max(0.0001, time.perf_counter() - started_at)
                return GenerationResult(
                    value=parsed,
                    model=model or self.engine_id,
                    elapsed_seconds=round(elapsed, 4),
                    total_duration_seconds=round(elapsed, 4),
                )

            last_error = attempt_error or LocalAIError("Gemini Web query generation failed.")
            if attempt == 1:
                raise last_error

        raise last_error or LocalAIError("Gemini Web query generation failed.")


__all__ = ["GeminiWebQueryEngine"]
