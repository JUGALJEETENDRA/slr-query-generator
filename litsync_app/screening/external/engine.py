from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from litsync_app.screening.local.engine import GenerationResult
from litsync_app.screening.local.engine import LocalAIOutputError


_MAX_DIAGNOSTIC_ERRORS = 10
_MAX_DIAGNOSTIC_KEYS = 20
_MAX_LOCATION_SEGMENTS = 6
_MAX_DIAGNOSTIC_NAME_LENGTH = 48
_MAX_DIAGNOSTIC_MESSAGE_LENGTH = 120
_MAX_DIAGNOSTIC_PAYLOAD_BYTES = 8192


def _safe_name(value: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or ""))
    return normalized[:_MAX_DIAGNOSTIC_NAME_LENGTH]


def _safe_validation_message(error_type: str) -> str:
    if error_type == "missing":
        return "Field required"
    if error_type == "extra_forbidden":
        return "Extra inputs are not permitted"
    if error_type == "literal_error":
        return "Value must match an allowed literal"
    if error_type.endswith("_type"):
        return "Value has an invalid type"
    if error_type in {"string_too_short", "too_short"}:
        return "Value does not meet the minimum length"
    if error_type in {"string_too_long", "too_long"}:
        return "Value exceeds the maximum length"
    if error_type in {
        "greater_than", "greater_than_equal", "less_than", "less_than_equal",
    }:
        return "Value is outside the allowed bounds"
    if error_type == "value_error":
        return "Value failed schema validation"
    return "Schema validation failed"


def _bounded_diagnostic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep optional diagnostic detail within a hard serialized-size limit."""
    bounded = dict(payload)
    list_fields = (
        "validation_error_messages",
        "validation_error_locations",
        "validation_error_types",
        "top_level_key_names",
    )
    while (
        len(json.dumps(bounded, ensure_ascii=True).encode("utf-8"))
        > _MAX_DIAGNOSTIC_PAYLOAD_BYTES
    ):
        field = next(
            (name for name in list_fields if bounded.get(name)),
            None,
        )
        if field is None:
            break
        bounded[field].pop()
    return bounded


def _emit_diagnostic(
    diagnostic_sink: Callable[[dict[str, Any]], None] | None,
    payload: dict[str, Any],
) -> None:
    if diagnostic_sink is None:
        return
    try:
        diagnostic_sink(_bounded_diagnostic_payload(payload))
    except Exception:
        # Diagnostics must never alter parsing, retries, or decisions.
        return


def _pydantic_error_details(
    exc: ValidationError,
) -> tuple[int, list[str], list[list[str | int]], list[str]]:
    errors = exc.errors(include_input=False, include_url=False)
    bounded = errors[:_MAX_DIAGNOSTIC_ERRORS]
    types: list[str] = []
    locations: list[list[str | int]] = []
    messages: list[str] = []
    for error in bounded:
        error_type = _safe_name(error.get("type", "validation_error"))
        location = [
            segment if isinstance(segment, int) else _safe_name(segment)
            for segment in error.get("loc", ())[:_MAX_LOCATION_SEGMENTS]
        ]
        types.append(error_type)
        locations.append(location)
        messages.append(
            _safe_validation_message(error_type)[:_MAX_DIAGNOSTIC_MESSAGE_LENGTH]
        )
    return len(errors), types, locations, messages


def _full_response_structure(text: str) -> dict[str, Any]:
    stripped = text.strip()
    brace_balance = 0
    bracket_balance = 0
    inside_string = False
    escape_pending = False
    for character in stripped:
        if inside_string:
            if escape_pending:
                escape_pending = False
            elif character == "\\":
                escape_pending = True
            elif character == '"':
                inside_string = False
            continue
        if character == '"':
            inside_string = True
        elif character == "{":
            brace_balance += 1
        elif character == "}":
            brace_balance -= 1
        elif character == "[":
            bracket_balance += 1
        elif character == "]":
            bracket_balance -= 1

    raw_decode_succeeded = False
    raw_decode_consumed_ratio = 0.0
    trailing_nonwhitespace_characters = 0
    if stripped:
        try:
            _, end = json.JSONDecoder().raw_decode(stripped)
            raw_decode_succeeded = True
            raw_decode_consumed_ratio = round(end / len(stripped), 6)
            trailing_nonwhitespace_characters = len(stripped[end:].strip())
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "full_response_starts_with_object": stripped.startswith("{"),
        "full_response_starts_with_array": stripped.startswith("["),
        "full_response_ends_with_object": stripped.endswith("}"),
        "full_response_ends_with_array": stripped.endswith("]"),
        "full_response_brace_balance": brace_balance,
        "full_response_bracket_balance": bracket_balance,
        "full_response_inside_string_at_end": inside_string,
        "full_response_escape_pending_at_end": escape_pending,
        "full_response_trailing_nonwhitespace_characters": (
            trailing_nonwhitespace_characters
        ),
        "full_response_raw_decode_succeeded": raw_decode_succeeded,
        "full_response_raw_decode_consumed_ratio": raw_decode_consumed_ratio,
    }


def parse_structured_model_output(
    raw: str,
    schema: type[BaseModel] | None = None,
    *,
    diagnostic_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict:
    """Extract one JSON object from API JSON, fenced JSON, or light UI prose."""
    raw_text = str(raw or "")
    text = raw_text.strip().lstrip("\ufeff")
    diagnostic: dict[str, Any] = {
        "failure_code": "",
        "raw_response_empty": not bool(text),
        "raw_response_utf8_bytes": len(raw_text.encode("utf-8")),
        "fenced_candidate_count": 0,
        "embedded_json_candidate_count": 0,
        "total_candidate_count": 0,
        "json_decodable_candidate_count": 0,
        "dictionary_candidate_count": 0,
        "non_dictionary_candidate_count": 0,
        "schema_validation_failure_count": 0,
        "candidate_source": "",
        "decoded_top_level_type": "",
        "top_level_key_names": [],
        "validation_error_count": 0,
        "validation_error_types": [],
        "validation_error_locations": [],
        "validation_error_messages": [],
        "validation_exception_type": "",
        "validation_exception_message": "",
        "full_response_json_decodable": False,
        "full_response_top_level_type": "",
        "full_response_schema_valid": None,
        "full_response_json_error_type": "",
        "full_response_json_error_message": "",
        "full_response_json_error_position": None,
        "full_response_json_error_line": None,
        "full_response_json_error_column": None,
        "full_response_json_error_position_ratio": None,
        "full_response_starts_with_object": False,
        "full_response_starts_with_array": False,
        "full_response_ends_with_object": False,
        "full_response_ends_with_array": False,
        "full_response_brace_balance": 0,
        "full_response_bracket_balance": 0,
        "full_response_inside_string_at_end": False,
        "full_response_escape_pending_at_end": False,
        "full_response_trailing_nonwhitespace_characters": 0,
        "full_response_raw_decode_succeeded": False,
        "full_response_raw_decode_consumed_ratio": 0.0,
    }
    try:
        diagnostic.update(_full_response_structure(text))
    except Exception:
        # Full-response diagnostics cannot affect parsing.
        pass
    if not text:
        diagnostic["failure_code"] = "empty_response"
        _emit_diagnostic(diagnostic_sink, diagnostic)
        raise LocalAIOutputError("Gemini Web returned an empty response. Please retry the job.")

    candidates: list[tuple[str, str]] = [("complete_response", text)]
    fenced_candidates = [
        ("fenced_block", block.strip())
        for block in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if block.strip()
    ]
    candidates.extend(fenced_candidates)
    diagnostic["fenced_candidate_count"] = len(fenced_candidates)

    decoder = json.JSONDecoder()
    embedded_candidate_count = 0
    for start, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
            candidates.append(("embedded_json", json.dumps(value)))
            embedded_candidate_count += 1
        except json.JSONDecodeError:
            continue
    diagnostic["embedded_json_candidate_count"] = embedded_candidate_count
    diagnostic["total_candidate_count"] = len(candidates)

    best_validation_error_count: int | None = None
    for candidate_source, candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            if candidate_source == "complete_response":
                try:
                    diagnostic["full_response_json_error_type"] = "JSONDecodeError"
                    diagnostic["full_response_json_error_message"] = (
                        str(exc.msg)[:_MAX_DIAGNOSTIC_MESSAGE_LENGTH]
                    )
                    diagnostic["full_response_json_error_position"] = int(exc.pos)
                    diagnostic["full_response_json_error_line"] = int(exc.lineno)
                    diagnostic["full_response_json_error_column"] = int(exc.colno)
                    diagnostic["full_response_json_error_position_ratio"] = round(
                        exc.pos / max(1, len(candidate)),
                        6,
                    )
                except Exception:
                    pass
            continue
        diagnostic["json_decodable_candidate_count"] += 1
        if candidate_source == "complete_response":
            diagnostic["full_response_json_decodable"] = True
            diagnostic["full_response_top_level_type"] = type(value).__name__
        if isinstance(value, dict):
            diagnostic["dictionary_candidate_count"] += 1
            if schema is None:
                if candidate_source == "complete_response":
                    diagnostic["full_response_schema_valid"] = True
                diagnostic["candidate_source"] = candidate_source
                diagnostic["decoded_top_level_type"] = "dict"
                diagnostic["top_level_key_names"] = [
                    _safe_name(key) for key in list(value)[:_MAX_DIAGNOSTIC_KEYS]
                ]
                _emit_diagnostic(diagnostic_sink, diagnostic)
                return value
            try:
                schema.model_validate(value)
                if candidate_source == "complete_response":
                    diagnostic["full_response_schema_valid"] = True
                diagnostic["candidate_source"] = candidate_source
                diagnostic["decoded_top_level_type"] = "dict"
                diagnostic["top_level_key_names"] = [
                    _safe_name(key) for key in list(value)[:_MAX_DIAGNOSTIC_KEYS]
                ]
                _emit_diagnostic(diagnostic_sink, diagnostic)
                return value
            except Exception as exc:
                if candidate_source == "complete_response":
                    diagnostic["full_response_schema_valid"] = False
                diagnostic["schema_validation_failure_count"] += 1
                try:
                    if isinstance(exc, ValidationError):
                        (
                            error_count,
                            error_types,
                            error_locations,
                            error_messages,
                        ) = _pydantic_error_details(exc)
                        exception_type = "ValidationError"
                        exception_message = ""
                    else:
                        error_count = 1
                        error_types = [_safe_name(type(exc).__name__)]
                        error_locations = []
                        error_messages = ["Schema validation raised an exception"]
                        exception_type = _safe_name(type(exc).__name__)
                        exception_message = "Schema validation raised an exception"
                    if (
                        best_validation_error_count is None
                        or error_count < best_validation_error_count
                    ):
                        best_validation_error_count = error_count
                        diagnostic["candidate_source"] = candidate_source
                        diagnostic["decoded_top_level_type"] = "dict"
                        diagnostic["top_level_key_names"] = [
                            _safe_name(key)
                            for key in list(value)[:_MAX_DIAGNOSTIC_KEYS]
                        ]
                        diagnostic["validation_error_count"] = error_count
                        diagnostic["validation_error_types"] = error_types
                        diagnostic["validation_error_locations"] = error_locations
                        diagnostic["validation_error_messages"] = error_messages
                        diagnostic["validation_exception_type"] = exception_type
                        diagnostic["validation_exception_message"] = (
                            exception_message[:_MAX_DIAGNOSTIC_MESSAGE_LENGTH]
                        )
                except Exception:
                    # Diagnostic extraction cannot affect candidate acceptance.
                    pass
                continue
        else:
            if candidate_source == "complete_response" and schema is not None:
                diagnostic["full_response_schema_valid"] = False
            diagnostic["non_dictionary_candidate_count"] += 1
            if not diagnostic["decoded_top_level_type"]:
                diagnostic["candidate_source"] = candidate_source
                diagnostic["decoded_top_level_type"] = type(value).__name__

    if diagnostic["schema_validation_failure_count"]:
        diagnostic["failure_code"] = "schema_validation_failed"
    elif not diagnostic["json_decodable_candidate_count"]:
        diagnostic["failure_code"] = "no_json_decodable_candidate"
    elif not diagnostic["dictionary_candidate_count"]:
        diagnostic["failure_code"] = "json_decoded_but_not_object"
    else:
        diagnostic["failure_code"] = "no_schema_matching_candidate"
    _emit_diagnostic(diagnostic_sink, diagnostic)

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
