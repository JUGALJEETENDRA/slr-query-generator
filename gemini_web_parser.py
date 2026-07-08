from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


class GeminiResponseParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedDecision:
    paper_id: str
    decision: str
    reason: str


_DECISION_MAP = {
    "include": "KEEP",
    "included": "KEEP",
    "keep": "KEEP",
    "exclude": "REJECT",
    "excluded": "REJECT",
    "reject": "REJECT",
    "rejected": "REJECT",
    "maybe": "MAYBE",
    "uncertain": "MAYBE",
    "unsure": "MAYBE",
}


def parse_gemini_screening_response(response_text: str, expected_ids: set[str]) -> list[ParsedDecision]:
    if not response_text or not response_text.strip():
        raise GeminiResponseParseError("Gemini returned an empty response.")

    parsed = _parse_json_response(response_text)
    if parsed is None:
        parsed = _parse_line_response(response_text)

    decisions = [_coerce_decision(item) for item in parsed]
    _validate_decisions(decisions, expected_ids)
    return decisions


def _parse_json_response(response_text: str) -> list[dict[str, Any]] | None:
    candidates = [response_text.strip()]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", response_text, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(block.strip() for block in fenced)

    object_match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0))

    array_match = re.search(r"\[.*\]", response_text, flags=re.DOTALL)
    if array_match:
        candidates.append(array_match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(data, dict):
            decisions = data.get("decisions") or data.get("results") or data.get("papers")
            if isinstance(decisions, list):
                return decisions
        if isinstance(data, list):
            return data

    return None


def _parse_line_response(response_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"(?:id|paper(?:\s*id)?)\s*[:#-]?\s*(?P<id>[A-Za-z0-9_.-]+)"
        r".{0,120}?\b(?P<decision>include|included|keep|exclude|excluded|reject|rejected|maybe|uncertain|unsure)\b"
        r"(?P<reason>.*)",
        flags=re.IGNORECASE,
    )
    for line in response_text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        reason = re.sub(r"^\s*[-:|,;]+\s*", "", match.group("reason")).strip()
        rows.append(
            {
                "id": match.group("id"),
                "decision": match.group("decision"),
                "reason": reason,
            }
        )
    if not rows:
        raise GeminiResponseParseError("Could not parse Gemini response as JSON or line-based decisions.")
    return rows


def _coerce_decision(item: dict[str, Any]) -> ParsedDecision:
    paper_id = str(item.get("id") or item.get("paper_id") or item.get("Paper ID") or "").strip()
    raw_decision = str(item.get("decision") or item.get("Decision") or "").strip().lower()
    reason = " ".join(str(item.get("reason") or item.get("Reason") or "").strip().split())

    decision = _DECISION_MAP.get(raw_decision)
    if not paper_id:
        raise GeminiResponseParseError(f"Malformed decision without paper id: {item}")
    if decision is None:
        raise GeminiResponseParseError(f"Unsupported decision for paper {paper_id}: {raw_decision}")
    if not reason:
        reason = "Gemini did not provide a rationale."

    return ParsedDecision(paper_id=paper_id, decision=decision, reason=reason)


def _validate_decisions(decisions: list[ParsedDecision], expected_ids: set[str]) -> None:
    found_ids = {decision.paper_id for decision in decisions}
    missing = expected_ids - found_ids
    extra = found_ids - expected_ids
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing ids: {sorted(missing)}")
        if extra:
            parts.append(f"unexpected ids: {sorted(extra)}")
        raise GeminiResponseParseError("Gemini response does not match the submitted batch; " + "; ".join(parts))
