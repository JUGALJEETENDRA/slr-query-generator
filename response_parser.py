"""Parse and validate structured Gemini batch responses."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List


class ResponseParseError(ValueError):
    pass


def _extract_json(text: str):
    decoder = json.JSONDecoder()
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        for index, character in enumerate(stripped):
            if character not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(stripped[index:])
                return value
            except json.JSONDecodeError:
                continue
    raise ResponseParseError("Gemini did not return valid JSON.")


def parse_batch_response(text: str, expected_ids: Iterable[str]) -> List[Dict]:
    expected = list(expected_ids)
    payload = _extract_json(text)
    results = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(results, list):
        raise ResponseParseError("Gemini response must contain a results list.")

    parsed: Dict[str, Dict] = {}
    for item in results:
        if not isinstance(item, dict):
            raise ResponseParseError("Every Gemini result must be an object.")
        paper_id = str(item.get("id", ""))
        decision = str(item.get("decision", "")).upper()
        if paper_id not in expected or paper_id in parsed:
            raise ResponseParseError(f"Unexpected or duplicate paper id: {paper_id!r}")
        if decision not in {"KEEP", "REJECT", "MAYBE"}:
            raise ResponseParseError(f"Invalid decision for {paper_id}: {decision!r}")
        parsed[paper_id] = {
            "id": paper_id,
            "decision": decision,
            "reason": str(item.get("reason") or "").strip(),
            "required_evidence": str(item.get("required_evidence") or "").strip(),
            "paper_contribution": str(item.get("paper_contribution") or "").strip(),
        }

    missing = [paper_id for paper_id in expected if paper_id not in parsed]
    if missing:
        raise ResponseParseError(f"Gemini omitted paper ids: {', '.join(missing)}")
    return [parsed[paper_id] for paper_id in expected]
