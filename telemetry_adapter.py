from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from task_ontology import RESEARCH_TASK_ONTOLOGY


def _normalize_label(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip().lower()
    return s or None


def _safe_float(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _matching_columns(row: pd.Series, candidates: list[str]) -> list[str]:
    matches: list[str] = []
    for candidate in candidates:
        needle = candidate.lower()
        for col in row.index:
            haystack = str(col).lower()
            if needle == haystack or needle in haystack:
                matches.append(col)
    return matches


def _row_value(row: pd.Series, candidates: list[str]) -> Any:
    for col in _matching_columns(row, candidates):
        value = row.get(col)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        if str(value).strip() != "":
            return value
    return None


def row_text(row: pd.Series, candidates: list[str]) -> str:
    value = _row_value(row, candidates)
    if value is None:
        return ""
    return str(value).strip()


def row_float(row: pd.Series, candidates: list[str]) -> Optional[float]:
    return _safe_float(_row_value(row, candidates))


def row_bool(row: pd.Series, candidates: list[str]) -> Optional[bool]:
    value = _row_value(row, candidates)
    if value is None:
        return None
    normalized = _normalize_label(value)
    if normalized in {"true", "t", "1", "yes", "y"}:
        return True
    if normalized in {"false", "f", "0", "no", "n"}:
        return False
    return None


def _known_canonical_task(value: str) -> bool:
    return str(value or "").strip() in RESEARCH_TASK_ONTOLOGY


@dataclass(frozen=True)
class TelemetryEvent:
    # Task
    canonical_task_left: str
    canonical_task_right: str
    task_identity_match: Optional[bool]
    task_similarity: Optional[float]

    # Technology
    technology_similarity: Optional[float]

    # Context
    context_similarity: Optional[float]

    # Study / review roles
    study_role_match: Optional[bool]
    review_role_match: Optional[bool]

    # Comparator task role signal (present in comparator output CSV)
    task_role_match: Optional[float]

    # Evidence
    evidence_type: str

    # Canonical reasoning strings (kept for downstream evidence/debugging)
    paper_review_role: str
    required_task: str
    paper_task: str
    rq_technology: str
    paper_technology: str
    required_evidence: str



def build_event(row: pd.Series) -> TelemetryEvent:
    # Maintain current evaluation_engine normalization semantics exactly.

    canonical_left = row_text(row, ["canonical_task_left"])
    canonical_right = row_text(row, ["canonical_task_right"])

    task_identity_match = row_bool(row, ["task_identity_match"])
    task_similarity = row_float(row, ["task_match"])

    technology_similarity = row_float(row, ["technology_match"])
    context_similarity = row_float(row, ["context_match"])

    study_role_match = row_bool(row, ["study_role_match"])
    review_role_match = row_bool(row, ["review_role_match"])
    task_role_match = row_float(row, ["task_role_match"])


    evidence_type = row_text(row, ["paper_evidence_type", "evidence_type"])

    # Maintain additional canonical strings used by evaluation_engine evidence/debug.
    paper_review_role = row_text(row, ["paper_review_role", "review_role"])
    required_task = row_text(row, ["rq_target_problem_or_task", "required_task"])
    paper_task = row_text(row, ["paper_target_problem_or_task", "target_problem_or_task"])
    rq_technology = row_text(row, ["rq_intervention_or_method", "required_technology"])
    paper_technology = row_text(row, ["paper_intervention_or_method", "intervention_or_method"])
    required_evidence = row_text(row, ["required_evidence", "rq_evidence_type"])

    # Ensure defaults are semantically identical to old helper behavior.
    # (No-op; kept to mirror prior adapter structure.)
    _ = _known_canonical_task(canonical_left)

    return TelemetryEvent(
        canonical_task_left=canonical_left,
        canonical_task_right=canonical_right,
        task_identity_match=task_identity_match,
        task_similarity=task_similarity,
        technology_similarity=technology_similarity,
        context_similarity=context_similarity,
        study_role_match=study_role_match,
        review_role_match=review_role_match,
        evidence_type=evidence_type,
        paper_review_role=paper_review_role,
        required_task=required_task,
        paper_task=paper_task,
        rq_technology=rq_technology,
        paper_technology=paper_technology,
        required_evidence=required_evidence,
    )


