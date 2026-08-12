"""Structured query-draft engines that are not part of the local AI stack."""

from __future__ import annotations

import json
import re
import time
from dataclasses import is_dataclass, replace
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from litsync_app.integrations.gemini_web_query_automation import GeminiWebQueryAutomation
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


class _QueryBudgetExceeded(TimeoutError):
    pass


def _budget_remaining(deadline: float | None) -> float | None:
    return None if deadline is None else deadline - time.monotonic()


def _require_budget(deadline: float | None) -> float | None:
    remaining = _budget_remaining(deadline)
    if remaining is not None and remaining <= 0:
        raise _QueryBudgetExceeded("Gemini Web query-generation budget expired.")
    return remaining


def _apply_supported_timeout(target: Any, remaining: float | None) -> None:
    if remaining is None:
        return
    timeout_ms = max(1, int(remaining * 1000))
    config = getattr(target, "config", None)
    if config is not None and is_dataclass(config):
        updates = {}
        for name in ("ready_timeout_ms", "response_timeout_ms", "no_container_timeout_ms"):
            current = getattr(config, name, None)
            if (
                isinstance(current, (int, float)) and not isinstance(current, bool)
                and current > 0
            ):
                updates[name] = min(current, timeout_ms)
        if updates:
            try:
                target.config = replace(config, **updates)
            except (AttributeError, TypeError, ValueError):
                pass
    page = getattr(target, "_page", None)
    if page is None:
        return
    for method_name in ("set_default_timeout", "set_default_navigation_timeout"):
        method = getattr(page, method_name, None)
        if callable(method):
            try:
                method(timeout_ms)
            except Exception:
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

    engine_id = "gemini_web"

    def __init__(
        self,
        browser_factory: Callable[[], Any] = GeminiWebQueryAutomation,
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
        started_at = time.perf_counter()
        deadline = (
            time.monotonic() + float(timeout_seconds)
            if timeout_seconds is not None and float(timeout_seconds) > 0 else None
        )
        request = (
            f"{prompt}\n\n{schema.__name__} JSON schema:\n"
            f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
        )
        last_error: LocalAIError | None = None

        for attempt in range(2):
            try:
                remaining = _require_budget(deadline)
            except _QueryBudgetExceeded as exc:
                raise LocalAIError(str(exc)) from exc
            browser = self.browser_factory()
            browser_started = False
            startup_attempted = False
            parsed: dict[str, Any] | None = None
            attempt_error: LocalAIError | None = None
            attempt_request = request
            if attempt == 1:
                attempt_request += (
                    "\n\nCORRECTION ATTEMPT: The previous response failed strict JSON "
                    "or query-contract validation. Return one corrected JSON object "
                    "that satisfies every schema constraint. Do not add commentary, "
                    "Markdown, or fields outside the schema."
                )
            try:
                remaining = _require_budget(deadline)
                _apply_supported_timeout(browser, remaining)
                startup_attempted = True
                browser.start()
                browser_started = True
                remaining = _require_budget(deadline)
                _apply_supported_timeout(browser, remaining)
                raw = browser.submit_prompt_and_get_response(attempt_request)
                _require_budget(deadline)
                value = _parse_json_object(raw)
                try:
                    parsed = schema.model_validate(value).model_dump(mode="json")
                except ValidationError as exc:
                    attempt_error = LocalAIOutputError(
                        f"Gemini returned invalid {schema.__name__} data: {exc}"
                    )
            except _MalformedGeminiJSON as exc:
                attempt_error = LocalAIOutputError(f"Gemini returned malformed JSON: {exc}")
            except (TimeoutError, RuntimeError) as exc:
                attempt_error = (
                    LocalAIError(str(exc)) if isinstance(exc, _QueryBudgetExceeded)
                    else LocalAIError(f"Gemini Web query generation failed: {exc}")
                )
            except Exception as exc:
                attempt_error = LocalAIError("Gemini Web browser interaction failed.")
                attempt_error.__cause__ = exc
            finally:
                if startup_attempted:
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
                if isinstance(last_error, LocalAIOutputError):
                    raise LocalAIOutputError(
                        "Gemini could not produce a valid query response after two attempts."
                    ) from last_error
                raise last_error
            try:
                _require_budget(deadline)
            except _QueryBudgetExceeded as exc:
                raise LocalAIError(str(exc)) from exc

        raise last_error or LocalAIError("Gemini Web query generation failed.")


__all__ = ["GeminiWebQueryEngine"]
