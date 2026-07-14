from __future__ import annotations

import re
from typing import Literal, TypedDict

from .contracts import EvidenceSpan


class EvidenceUnit(TypedDict):
    evidence_id: str
    source: Literal["title", "abstract"]
    text: str


def _text_units(text: str) -> list[str]:
    """Split source text into exact, human-readable units without changing it."""
    if not text or not text.strip():
        return []
    units: list[str] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+|\n+", text):
        unit = text[start:match.start()].strip()
        if unit:
            units.append(unit)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        units.append(tail)
    return units


def build_evidence_units(title: str, abstract: str) -> list[EvidenceUnit]:
    units: list[EvidenceUnit] = []
    for number, text in enumerate(_text_units(title), start=1):
        units.append({"evidence_id": f"title_{number:03d}", "source": "title", "text": text})
    for number, text in enumerate(_text_units(abstract), start=1):
        units.append({"evidence_id": f"abstract_{number:03d}", "source": "abstract", "text": text})
    return units


def evidence_lookup(title: str, abstract: str) -> dict[str, EvidenceUnit]:
    return {unit["evidence_id"]: unit for unit in build_evidence_units(title, abstract)}


def resolve_evidence(reference: EvidenceSpan, title: str, abstract: str) -> EvidenceUnit | None:
    unit = evidence_lookup(title, abstract).get(reference.evidence_id)
    if unit is None or unit["source"] != reference.source:
        return None
    return unit
