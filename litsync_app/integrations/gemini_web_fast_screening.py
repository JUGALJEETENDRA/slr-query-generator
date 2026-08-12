from __future__ import annotations

import asyncio
import json
import os
import re
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from pydantic import ValidationError

from litsync_app.integrations.gemini_web_fast_browser import GeminiWebFastBrowser
from litsync_app.integrations.gemini_web_fast_prompt import (
    ARCHITECTURE_VERSION,
    PROMPT_VERSION,
    ScreeningBatchResult,
    ScreeningRubric,
    batch_prompt,
    criterion_entries,
    fallback_rubric,
    protocol_prompt,
)
from litsync_app.screening.local.contracts import SCHEMA_VERSION


GEMINI_WEB_FAST_ENGINE = "gemini_web_fast"


def _bounded_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))



def _normalize(value: str) -> str:
    return " ".join(str(value or "").split()).casefold()


def _clean_cell(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value or "").strip()


def _strip_quote_wrappers(value: str) -> str:
    text = str(value or "").strip()
    pairs = {
        '"': '"',
        "'": "'",
        "“": "”",
        "‘": "’",
    }
    if len(text) >= 2 and text[0] in pairs and text[-1] == pairs[text[0]]:
        return text[1:-1].strip()
    return text


def _validate_evidence(
    quote: str,
    *,
    title: str,
    abstract: str,
) -> tuple[bool, str, str]:
    candidate = _strip_quote_wrappers(quote)
    if not candidate:
        return False, "none", "empty_evidence_quote"
    if "..." in candidate or "…" in candidate:
        return False, "none", "ellipsis_not_allowed"

    normalized_quote = _normalize(candidate)
    if not normalized_quote:
        return False, "none", "empty_evidence_quote"

    if normalized_quote in _normalize(title):
        return True, "title", ""
    if normalized_quote in _normalize(abstract):
        return True, "abstract", ""
    return False, "none", "quote_not_found_in_source"


def _structural_valid(assessment: dict[str, Any] | None) -> bool:
    if not assessment:
        return False
    if "structural_valid" in assessment:
        return bool(assessment.get("structural_valid"))
    return bool(assessment.get("valid")) and not bool(assessment.get("technical_failure"))


def _semantic_valid(assessment: dict[str, Any] | None) -> bool:
    if not assessment:
        return False
    if "semantic_valid" in assessment:
        return bool(assessment.get("semantic_valid"))
    return bool(assessment.get("valid")) and not bool(assessment.get("failure_class"))


def _evidence_valid(assessment: dict[str, Any] | None) -> bool:
    if not assessment:
        return False
    if "evidence_valid" in assessment:
        return bool(assessment.get("evidence_valid"))
    return bool(assessment.get("valid"))


def _technical_invalid(assessment: dict[str, Any] | None) -> bool:
    if not assessment:
        return True
    if "structural_valid" in assessment or "technical_failure" in assessment:
        return bool(
            assessment.get("technical_failure")
            or not _structural_valid(assessment)
        )
    return bool(
        not assessment.get("valid")
        and assessment.get("failure_class")
    )


def _fully_valid(assessment: dict[str, Any] | None) -> bool:
    if not assessment:
        return False
    decision = str(
        assessment.get("model_decision")
        or assessment.get("decision")
        or ""
    )
    if not (_structural_valid(assessment) and _semantic_valid(assessment)):
        return False
    if decision in {"KEEP", "REJECT"}:
        return _evidence_valid(assessment)
    return True


def _json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Gemini must return one JSON object")
    return value


def _protocol_preserves_inputs(
    rubric: ScreeningRubric,
    inclusion: str,
    exclusion: str,
) -> bool:
    inclusion_text = " ".join(
        f"{item.text} {item.source_text}"
        for item in rubric.inclusion_criteria
    )
    exclusion_text = " ".join(
        f"{item.text} {item.source_text}"
        for item in rubric.exclusion_criteria
    )

    inputs_preserved = (
        all(
            _normalize(item) in _normalize(inclusion_text)
            for item in criterion_entries(inclusion)
        )
        and all(
            _normalize(item) in _normalize(exclusion_text)
            for item in criterion_entries(exclusion)
        )
    )
    if not inputs_preserved:
        return False

    normalized_inclusion = _normalize(inclusion)
    explicit_alternative_markers = (
        "at least one",
        "one or more",
        "any of",
        "either ",
        "one of",
    )
    has_explicit_alternative = any(
        marker in normalized_inclusion
        for marker in explicit_alternative_markers
    )
    has_alternative_group = any(
        group.operator in {"ANY", "AT_LEAST"}
        for group in rubric.criterion_groups
    )
    mandatory_count = sum(
        criterion.role == "MANDATORY"
        for criterion in rubric.inclusion_criteria
    )

    # Reject an obviously unsafe compilation that turns an explicit
    # alternative list into several mandatory AND conditions. The safe
    # fallback keeps the complete original criteria block authoritative.
    if (
        has_explicit_alternative
        and mandatory_count > 1
        and not has_alternative_group
    ):
        return False

    return True



def _validate_batch(
    raw: str,
    papers: list[dict[str, str]],
    rubric: ScreeningRubric,
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, str]]:
    expected = {paper["paper_id"]: paper for paper in papers}
    accepted: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}

    try:
        result = ScreeningBatchResult.model_validate(_json_object(raw))
    except (ValueError, json.JSONDecodeError, ValidationError):
        return {}, list(expected), {
            paper_id: "schema_invalid"
            for paper_id in expected
        }

    returned_ids = [item.paper_id for item in result.items]
    if any(paper_id not in expected for paper_id in returned_ids):
        return {}, list(expected), {
            paper_id: "unknown_id_in_response"
            for paper_id in expected
        }

    duplicate_ids = {
        paper_id
        for paper_id in returned_ids
        if returned_ids.count(paper_id) > 1
    }

    inclusion_ids = [
        item.criterion_id
        for item in rubric.inclusion_criteria
    ]
    exclusion_ids = [
        item.criterion_id
        for item in rubric.exclusion_criteria
    ]
    inclusion_by_id = {
        item.criterion_id: item
        for item in rubric.inclusion_criteria
    }

    grouped_ids = {
        member_id
        for group in rubric.criterion_groups
        for member_id in group.member_ids
    }

    for item in result.items:
        paper_id = item.paper_id

        if paper_id in duplicate_ids:
            failures[paper_id] = "duplicate_id"
            continue

        got_inclusion = [
            entry.criterion_id
            for entry in item.inclusion_assessments
        ]
        got_exclusion = [
            entry.criterion_id
            for entry in item.exclusion_assessments
        ]

        if (
            sorted(got_inclusion) != sorted(inclusion_ids)
            or len(got_inclusion) != len(set(got_inclusion))
        ):
            failures[paper_id] = "invalid_inclusion_assessments"
            continue

        if (
            sorted(got_exclusion) != sorted(exclusion_ids)
            or len(got_exclusion) != len(set(got_exclusion))
        ):
            failures[paper_id] = "invalid_exclusion_assessments"
            continue

        inclusion = {
            entry.criterion_id: entry.status
            for entry in item.inclusion_assessments
        }
        exclusion = {
            entry.criterion_id: entry.status
            for entry in item.exclusion_assessments
        }

        semantic_errors: list[str] = []
        semantic_warnings: list[str] = []

        explicit_mandatory_failures = [
            criterion_id
            for criterion_id, criterion in inclusion_by_id.items()
            if (
                criterion.role == "MANDATORY"
                and criterion_id not in grouped_ids
                and inclusion.get(criterion_id) == "NOT_MET"
            )
        ]
        explicit_mandatory_unclear = [
            criterion_id
            for criterion_id, criterion in inclusion_by_id.items()
            if (
                criterion.role == "MANDATORY"
                and criterion_id not in grouped_ids
                and inclusion.get(criterion_id) == "UNCLEAR"
            )
        ]

        failed_groups: list[str] = []
        unclear_groups: list[str] = []
        for group in rubric.criterion_groups:
            statuses = [
                inclusion.get(member_id, "UNCLEAR")
                for member_id in group.member_ids
            ]
            met_count = sum(status == "MET" for status in statuses)
            unclear_count = sum(status == "UNCLEAR" for status in statuses)
            required = (
                len(group.member_ids)
                if group.operator == "ALL"
                else int(group.minimum_required)
            )

            if met_count >= required:
                continue
            if met_count + unclear_count < required:
                failed_groups.append(group.group_id)
            else:
                unclear_groups.append(group.group_id)

        met_exclusions = [
            criterion_id
            for criterion_id, status in exclusion.items()
            if status == "MET"
        ]

        unresolved_not_met = [
            criterion_id
            for criterion_id, criterion in inclusion_by_id.items()
            if (
                criterion.role == "UNRESOLVED"
                and inclusion.get(criterion_id) == "NOT_MET"
            )
        ]
        unresolved_unclear = [
            criterion_id
            for criterion_id, criterion in inclusion_by_id.items()
            if (
                criterion.role == "UNRESOLVED"
                and inclusion.get(criterion_id) == "UNCLEAR"
            )
        ]

        if item.decision == "KEEP":
            if explicit_mandatory_failures:
                semantic_errors.append("keep_failed_mandatory_criterion")
            if failed_groups:
                semantic_errors.append("keep_failed_criterion_group")
            if met_exclusions:
                semantic_errors.append("keep_met_exclusion_criterion")
            if explicit_mandatory_unclear or unclear_groups:
                semantic_warnings.append("keep_mandatory_logic_unclear")
            if unresolved_not_met or unresolved_unclear:
                semantic_warnings.append("unresolved_inclusion_logic")

        elif item.decision == "REJECT":
            decisive_reject = bool(
                explicit_mandatory_failures
                or failed_groups
                or met_exclusions
            )
            if not decisive_reject:
                if unresolved_not_met or unresolved_unclear:
                    semantic_warnings.append("reject_under_unresolved_logic")
                else:
                    semantic_warnings.append("reject_without_locally_decisive_criterion")

        elif item.decision == "MAYBE":
            if (
                explicit_mandatory_failures
                or failed_groups
                or met_exclusions
            ):
                semantic_warnings.append("maybe_despite_decisive_criterion")

        evidence_valid = True
        evidence_source = "none"
        evidence_failure_reason = ""

        if item.decision in {"KEEP", "REJECT"}:
            (
                evidence_valid,
                evidence_source,
                evidence_failure_reason,
            ) = _validate_evidence(
                item.evidence_quote,
                title=expected[paper_id]["title"],
                abstract=expected[paper_id]["abstract"],
            )

        structural_valid = True
        semantic_valid = not semantic_errors
        fully_valid = bool(
            structural_valid
            and semantic_valid
            and (
                evidence_valid
                or item.decision == "MAYBE"
            )
        )

        risk_flags = list(dict.fromkeys([
            *item.risk_flags,
            *semantic_warnings,
            *(
                ["invalid_evidence_quote"]
                if not evidence_valid
                and item.decision in {"KEEP", "REJECT"}
                else []
            ),
        ]))

        validation_errors = [
            *semantic_errors,
            *(
                [evidence_failure_reason]
                if evidence_failure_reason
                else []
            ),
        ]

        if semantic_errors:
            validation_status = "semantic_contradiction"
            failure_class = "validation_contradiction"
        elif not evidence_valid and item.decision in {"KEEP", "REJECT"}:
            validation_status = "semantic_valid_evidence_invalid"
            failure_class = ""
        elif semantic_warnings:
            validation_status = "valid_with_verification_required"
            failure_class = ""
        else:
            validation_status = "valid"
            failure_class = ""

        accepted[paper_id] = {
            "paper_id": paper_id,
            "decision": item.decision if semantic_valid else "MAYBE",
            "model_decision": item.decision,
            "confidence": item.confidence,
            "reason": item.reason,
            "evidence_quote": item.evidence_quote,
            "inclusion_assessments": [
                entry.model_dump()
                for entry in item.inclusion_assessments
            ],
            "exclusion_assessments": [
                entry.model_dump()
                for entry in item.exclusion_assessments
            ],
            "risk_flags": risk_flags,
            "validation_errors": validation_errors,
            "validation_warnings": semantic_warnings,
            "validation_status": validation_status,
            "failure_class": failure_class,
            "structural_valid": structural_valid,
            "semantic_valid": semantic_valid,
            "evidence_valid": evidence_valid,
            "evidence_source": evidence_source,
            "evidence_failure_reason": evidence_failure_reason,
            "technical_failure": False,
            "valid": fully_valid,
        }

    unresolved = [
        paper_id
        for paper_id in expected
        if paper_id not in accepted
    ]
    for paper_id in unresolved:
        failures.setdefault(
            paper_id,
            "missing_or_unknown_id",
        )

    return accepted, unresolved, failures



def _technical_maybe(
    paper_id: str,
    reason: str,
    failure: str,
) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "decision": "MAYBE",
        "model_decision": "",
        "confidence": 0.0,
        "reason": reason,
        "evidence_quote": "",
        "inclusion_assessments": [],
        "exclusion_assessments": [],
        "risk_flags": [],
        "validation_errors": [failure],
        "validation_warnings": [],
        "validation_status": "safe_fallback",
        "failure_class": failure,
        "structural_valid": False,
        "semantic_valid": False,
        "evidence_valid": False,
        "evidence_source": "none",
        "evidence_failure_reason": failure,
        "technical_failure": True,
        "valid": False,
    }


def _requires_verification(
    assessment: dict[str, Any],
) -> bool:
    return bool(
        assessment.get("model_decision") in {"REJECT", "MAYBE"}
        or (
            assessment.get("model_decision") == "KEEP"
            and float(assessment.get("confidence") or 0) < 0.80
        )
        or not _fully_valid(assessment)
        or assessment.get("risk_flags")
    )


def _merge(
    primary: dict[str, Any],
    verifier: dict[str, Any] | None,
) -> tuple[str, float, str, str]:
    primary_model_decision = str(
        primary.get("model_decision")
        or primary.get("decision")
        or ""
    )

    if primary.get("failure_class") == "missing_abstract":
        return (
            "MAYBE",
            0.0,
            str(primary.get("reason") or "The abstract is missing."),
            "missing_abstract",
        )

    if _technical_invalid(primary):
        return (
            "MAYBE",
            0.0,
            (
                "The primary Gemini assessment was technically unusable. "
                "One later assessment alone is insufficient for a definitive decision."
            ),
            "primary_validation_failed",
        )

    if not _semantic_valid(primary):
        return (
            "MAYBE",
            0.0,
            (
                "The primary Gemini assessment contradicted an explicit "
                "mandatory or exclusion rule."
            ),
            "primary_semantic_contradiction",
        )

    verification_required = _requires_verification(primary)

    if verifier is None:
        if verification_required:
            return (
                "MAYBE",
                0.0,
                "Required independent validation did not complete successfully.",
                "verification_unfinished",
            )
        return (
            primary_model_decision,
            float(primary.get("confidence") or 0),
            str(primary.get("reason") or ""),
            "not_required",
        )

    if _technical_invalid(verifier) or not _semantic_valid(verifier):
        return (
            "MAYBE",
            0.0,
            "Independent validation was unavailable or contradictory.",
            "verification_failed",
        )

    verifier_model_decision = str(
        verifier.get("model_decision")
        or verifier.get("decision")
        or ""
    )

    if primary_model_decision == "MAYBE":
        if (
            verifier_model_decision in {"KEEP", "REJECT"}
            and _fully_valid(verifier)
        ):
            return (
                verifier_model_decision,
                float(verifier.get("confidence") or 0),
                str(verifier.get("reason") or ""),
                "resolved_by_verifier",
            )
        return (
            "MAYBE",
            min(
                float(primary.get("confidence") or 0),
                float(verifier.get("confidence") or 0),
            ),
            str(primary.get("reason") or ""),
            "both_maybe",
        )

    if verifier_model_decision != primary_model_decision:
        return (
            "MAYBE",
            0.0,
            "Primary and blind verification decisions disagreed.",
            "disagreement",
        )

    if _fully_valid(primary) and _fully_valid(verifier):
        return (
            primary_model_decision,
            min(
                float(primary.get("confidence") or 0),
                float(verifier.get("confidence") or 0),
            ),
            str(primary.get("reason") or ""),
            "agreement",
        )

    if (
        _semantic_valid(primary)
        and not _evidence_valid(primary)
        and _fully_valid(verifier)
    ):
        return (
            primary_model_decision,
            min(
                float(primary.get("confidence") or 0),
                float(verifier.get("confidence") or 0),
            ),
            str(verifier.get("reason") or ""),
            "agreement_recovered_by_verifier_evidence",
        )

    return (
        "MAYBE",
        0.0,
        (
            "The model decisions agreed, but the required independent "
            "evidence validation was incomplete."
        ),
        "verification_failed",
    )


def _review_protocol_id(inputs: dict[str, str]) -> str:
    payload = {
        "architecture_version": ARCHITECTURE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "research_question": inputs["question"],
        "research_context": inputs["context"],
        "inclusion_criteria": inputs["inclusion"],
        "exclusion_criteria": inputs["exclusion"],
    }
    digest = sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"gemini-web-fast-v1-{digest}"


def _record_transport_failure(
    stats: dict[str, Any], exc: Exception, *, screening_stage: str,
    batch_id: str, attempt: int, browser: Any,
) -> None:
    stats["transport_failure_count"] += 1
    diagnostics = stats.setdefault("transport_diagnostics", [])
    if len(diagnostics) >= 100:
        return
    diagnostics.append({
        "screening_stage": screening_stage,
        "batch_id": batch_id,
        "attempt_number": int(attempt),
        "failure_stage": str(getattr(exc, "stage", "unknown")),
        "exception_type": str(getattr(exc, "exception_type", type(exc).__name__)),
        "active_tab_count": int(getattr(exc, "active_tabs", getattr(browser, "active_pages", 0)) or 0),
    })


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)



def _row(
    paper: dict[str, Any],
    primary: dict[str, Any],
    verifier: dict[str, Any] | None,
    *,
    origin: str,
    protocol_id: str,
    resumed: bool = False,
) -> dict[str, Any]:
    decision, confidence, reason, agreement = _merge(
        primary,
        verifier,
    )

    route = "resumed" if resumed else (
        "missing_abstract"
        if primary.get("failure_class") == "missing_abstract"
        else "blind_verification"
        if verifier is not None
        else "safe_fallback"
        if _technical_invalid(primary)
        else "primary_only"
    )

    primary_model_decision = str(
        primary.get("model_decision")
        or primary.get("decision")
        or ""
    )
    verifier_model_decision = (
        ""
        if verifier is None
        else str(
            verifier.get("model_decision")
            or verifier.get("decision")
            or ""
        )
    )

    final_evidence_source = "none"
    final_evidence_location = "none"
    evidence_quote = ""

    if decision in {"KEEP", "REJECT"}:
        if (
            verifier is not None
            and verifier_model_decision == decision
            and _evidence_valid(verifier)
        ):
            evidence_quote = str(
                verifier.get("evidence_quote")
                or ""
            )
            final_evidence_source = "verifier"
            final_evidence_location = str(
                verifier.get("evidence_source")
                or "none"
            )
        elif (
            primary_model_decision == decision
            and _evidence_valid(primary)
        ):
            evidence_quote = str(
                primary.get("evidence_quote")
                or ""
            )
            final_evidence_source = "primary"
            final_evidence_location = str(
                primary.get("evidence_source")
                or "none"
            )
    else:
        if verifier is not None and verifier.get("evidence_quote"):
            evidence_quote = str(verifier.get("evidence_quote") or "")
            final_evidence_source = (
                "verifier"
                if _evidence_valid(verifier)
                else "none"
            )
            final_evidence_location = (
                str(verifier.get("evidence_source") or "none")
                if _evidence_valid(verifier)
                else "none"
            )
        elif primary.get("evidence_quote"):
            evidence_quote = str(primary.get("evidence_quote") or "")
            final_evidence_source = (
                "primary"
                if _evidence_valid(primary)
                else "none"
            )
            final_evidence_location = (
                str(primary.get("evidence_source") or "none")
                if _evidence_valid(primary)
                else "none"
            )

    definitive_evidence_valid = bool(
        decision not in {"KEEP", "REJECT"}
        or (
            evidence_quote
            and final_evidence_source != "none"
        )
    )

    final_safe = decision == "MAYBE" and agreement in {
        "primary_validation_failed",
        "primary_semantic_contradiction",
        "missing_abstract",
        "verification_unfinished",
        "verification_failed",
        "disagreement",
    }

    validation_status = (
        "validated"
        if not final_safe and definitive_evidence_valid
        else "safe_fallback"
    )

    failure_class = ""
    if validation_status != "validated":
        if agreement in {"disagreement", "primary_semantic_contradiction"}:
            failure_class = "validation_contradiction"
        elif agreement == "missing_abstract":
            failure_class = "missing_abstract"
        elif agreement == "primary_validation_failed":
            failure_class = str(
                primary.get("failure_class")
                or "primary_validation_failed"
            )
        elif agreement == "verification_failed":
            failure_class = str(
                (
                    verifier.get("failure_class")
                    if verifier
                    else ""
                )
                or "verification_failed"
            )
        elif agreement == "verification_unfinished":
            failure_class = "verification_unfinished"
        else:
            failure_class = str(
                primary.get("failure_class")
                or (
                    verifier.get("failure_class")
                    if verifier
                    else ""
                )
                or ""
            )

    result = dict(paper["original"])
    result.update({
        "Decision": decision,
        "Confidence": confidence,
        "Reason": reason,
        "Evidence_Quote": evidence_quote,
        "Route_Used": route,
        "Validation_Status": validation_status,
        "Failure_Class": failure_class,
        "Primary_Decision": primary_model_decision,
        "Primary_Confidence": primary.get("confidence", 0.0),
        "Verifier_Decision": verifier_model_decision,
        "Verifier_Confidence": (
            ""
            if verifier is None
            else verifier.get("confidence", 0.0)
        ),
        "Agreement_Status": agreement,
        "Prompt_Version": PROMPT_VERSION,
        "Architecture_Version": ARCHITECTURE_VERSION,
        "Protocol_ID": protocol_id,
        "Review_Protocol_ID": protocol_id,
        "Source_Row_Index": paper["paper_id"],
        "Execution_Origin": (
            "resume"
            if resumed
            else "direct_handling"
            if primary.get("failure_class") == "missing_abstract"
            else "technical_fallback"
            if _technical_invalid(primary)
            else origin
        ),
        "Primary_Structural_Valid": _structural_valid(primary),
        "Primary_Semantic_Valid": _semantic_valid(primary),
        "Primary_Evidence_Valid": _evidence_valid(primary),
        "Verifier_Structural_Valid": (
            ""
            if verifier is None
            else _structural_valid(verifier)
        ),
        "Verifier_Semantic_Valid": (
            ""
            if verifier is None
            else _semantic_valid(verifier)
        ),
        "Verifier_Evidence_Valid": (
            ""
            if verifier is None
            else _evidence_valid(verifier)
        ),
        "Final_Evidence_Source": final_evidence_source,
        "Final_Evidence_Location": final_evidence_location,
        "Primary_Assessment_JSON": json.dumps(
            primary,
            ensure_ascii=False,
        ),
        "Verifier_Assessment_JSON": (
            json.dumps(
                verifier,
                ensure_ascii=False,
            )
            if verifier
            else ""
        ),
    })

    return result



def _reusable_resume_row(row: dict[str, Any]) -> bool:
    """Return whether a Fast-v2 prompt checkpoint row is safe to reuse."""
    route = str(
        row.get("Route_Used")
        or ""
    ).strip().casefold()
    origin = str(
        row.get("Execution_Origin")
        or ""
    ).strip().casefold()
    validation = str(
        row.get("Validation_Status")
        or ""
    ).strip().casefold()
    failure = str(
        row.get("Failure_Class")
        or ""
    ).strip()
    decision = str(
        row.get("Decision")
        or ""
    ).strip().upper()

    if route == "missing_abstract":
        return True

    if route == "safe_fallback" or origin == "technical_fallback":
        return False

    if failure or validation != "validated":
        return False

    if decision not in {"KEEP", "MAYBE", "REJECT"}:
        return False

    if decision in {"KEEP", "REJECT"}:
        evidence_source = str(
            row.get("Final_Evidence_Source")
            or ""
        ).strip().casefold()
        evidence_quote = str(
            row.get("Evidence_Quote")
            or ""
        ).strip()
        if not evidence_quote or evidence_source in {"", "none"}:
            return False

    try:
        primary = json.loads(
            str(row.get("Primary_Assessment_JSON") or "")
        )
        verifier_raw = str(
            row.get("Verifier_Assessment_JSON")
            or ""
        ).strip()
        verifier = (
            json.loads(verifier_raw)
            if verifier_raw
            else None
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return False

    merged_decision, _, _, _ = _merge(
        primary,
        verifier,
    )
    return merged_decision == decision


async def _compile_protocol(browser, question: str, context: str, inclusion: str, exclusion: str, deadline: float, stats: dict[str, Any]) -> ScreeningRubric:
    prompt = protocol_prompt(question, context, inclusion, exclusion)
    for attempt in range(2):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            raw = await browser.submit_fresh(prompt, timeout_seconds=remaining)
        except Exception as exc:
            _record_transport_failure(
                stats, exc, screening_stage="protocol_compilation",
                batch_id="protocol", attempt=attempt + 1, browser=browser,
            )
        else:
            try:
                rubric = ScreeningRubric.model_validate(_json_object(raw))
            except Exception:
                rubric = None
            if rubric is not None and _protocol_preserves_inputs(rubric, inclusion, exclusion):
                return rubric
        if attempt == 0:
            stats["retry_count"] += 1
    return fallback_rubric(inclusion, exclusion)


async def _screen_batch(
    browser, papers: list[dict[str, Any]], rubric: ScreeningRubric,
    inputs: dict[str, str], deadline: float, stats: dict[str, Any], *, verification: bool,
    batch_id: str,
) -> dict[str, dict[str, Any]]:
    pending = list(papers)
    results: dict[str, dict[str, Any]] = {}
    failure_by_id: dict[str, str] = {}
    for attempt in range(2):
        if not pending:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            failure_by_id.update({paper["paper_id"]: "time_budget" for paper in pending})
            break
        payload = [{"paper_id": p["paper_id"], "title": p["title"], "abstract": p["abstract"]} for p in pending]
        try:
            raw = await browser.submit_fresh(
                batch_prompt(
                    question=inputs["question"], context=inputs["context"],
                    inclusion=inputs["inclusion"], exclusion=inputs["exclusion"],
                    rubric=rubric, papers=payload, verification=verification,
                ),
                timeout_seconds=remaining,
            )
        except Exception as exc:
            _record_transport_failure(
                stats, exc,
                screening_stage="blind_verification" if verification else "primary_screening",
                batch_id=batch_id, attempt=attempt + 1, browser=browser,
            )
            failure_by_id.update({paper["paper_id"]: "browser_or_transport_failure" for paper in pending})
        else:
            try:
                valid, unresolved, failures = _validate_batch(raw, payload, rubric)
            except Exception:
                failure_by_id.update({paper["paper_id"]: "structured_output_failure" for paper in pending})
            else:
                results.update(valid)
                failure_by_id.update(failures)
                pending = [paper for paper in pending if paper["paper_id"] in unresolved]
        if pending and attempt == 0:
            stats["retry_count"] += 1
    for paper in pending:
        paper_id = paper["paper_id"]
        results[paper_id] = _technical_maybe(
            paper_id,
            "Gemini screening did not return a complete valid assessment; manual review is required.",
            failure_by_id.get(paper_id, "structured_output_failure"),
        )
    return results


async def _run_fast(
    *, papers: list[dict[str, Any]], browser, inputs: dict[str, str],
    job_timeout: int, primary_batch_size: int, verification_batch_size: int,
    concurrency: int, progress, job_id: str, checkpoint: Path, checkpoint_meta: Path,
    checkpoint_identity: str, protocol_id: str, resume_rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], ScreeningRubric, dict[str, Any]]:
    started = time.monotonic()
    deadline = started + job_timeout
    finalization_deadline = deadline - min(90, max(1, job_timeout // 10))
    stats: dict[str, Any] = {
        "retry_count": 0, "primary_batches_submitted": 0,
        "primary_batches_completed": 0,
        "verification_batches_submitted": 0, "verification_batches_completed": 0,
        "primary_papers_requested": 0,
        "verification_papers_requested": 0, "stopped_by_time_budget": False,
        "browser_context_started": 0, "fresh_primary_count": 0,
        "transport_failure_count": 0, "transport_diagnostics": [],
        "scheduler_worker_count": max(1, int(concurrency)),
    }
    primary: dict[str, dict[str, Any]] = {}
    verifier: dict[str, dict[str, Any]] = {}

    # A complete validated checkpoint must resume without opening Gemini or
    # recompiling the protocol. This makes reload/resume deterministic even
    # when the browser or network is temporarily unavailable.
    for paper_id, row in resume_rows.items():
        try:
            primary[paper_id] = json.loads(row["Primary_Assessment_JSON"])
            if row.get("Verifier_Assessment_JSON"):
                verifier[paper_id] = json.loads(row["Verifier_Assessment_JSON"])
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
    if len(primary) == len(papers):
        progress.begin_batches(job_id, "restoring_checkpoint", len(papers), len(papers), 1)
        progress.update_batch(job_id, len(papers), len(papers))
        return primary, verifier, None, stats

    if hasattr(browser, "activity_callback"):
        browser.activity_callback = lambda active: progress.update_fast_runtime(
            job_id, active_tabs=active,
        )
    await browser.start()
    stats["browser_context_started"] = 1
    try:
        def persist_checkpoint_metadata() -> None:
            _atomic_json(checkpoint_meta, {
                "identity": checkpoint_identity,
                "architecture_version": ARCHITECTURE_VERSION,
                "prompt_version": PROMPT_VERSION,
                "protocol_id": protocol_id,
                "counters": {
                    "resumed_count": len(resume_rows),
                    "fresh_primary_count": stats["fresh_primary_count"],
                    "primary_batches_submitted": stats["primary_batches_submitted"],
                    "primary_batches_completed": stats["primary_batches_completed"],
                    "verification_batches_submitted": stats["verification_batches_submitted"],
                    "verification_batches_completed": stats["verification_batches_completed"],
                    "browser_context_started": stats["browser_context_started"],
                    "pages_opened": getattr(browser, "pages_opened", 0),
                    "pages_closed": getattr(browser, "pages_closed", 0),
                    "peak_simultaneous_tabs": getattr(browser, "peak_active_pages", 0),
                    "transport_failure_count": stats["transport_failure_count"],
                },
            })

        progress.begin_batches(job_id, "compiling_protocol", 1, 1, 1)
        rubric = await _compile_protocol(
            browser,
            deadline=finalization_deadline,
            stats=stats,
            **inputs,
        )
        progress.update_batch(job_id, 1, 1)

        for paper_id, row in resume_rows.items():
            try:
                primary[paper_id] = json.loads(row["Primary_Assessment_JSON"])
                if row.get("Verifier_Assessment_JSON"):
                    verifier[paper_id] = json.loads(row["Verifier_Assessment_JSON"])
            except (KeyError, TypeError, json.JSONDecodeError):
                continue

        missing = [
            paper
            for paper in papers
            if not paper["abstract"] and paper["paper_id"] not in primary
        ]
        for paper in missing:
            primary[paper["paper_id"]] = _technical_maybe(
                paper["paper_id"],
                "Insufficient title-and-abstract evidence because the abstract is missing.",
                "missing_abstract",
            )

        pending = [
            paper
            for paper in papers
            if paper["paper_id"] not in primary and paper["abstract"]
        ]
        stats["fresh_primary_count"] = len(pending)
        primary_batches = [
            pending[index:index + primary_batch_size]
            for index in range(0, len(pending), primary_batch_size)
        ]
        total_primary_batches = len(primary_batches)
        progress.begin_batches(
            job_id,
            "primary_screening",
            len(pending),
            total_primary_batches,
            primary_batch_size,
        )

        paper_by_id = {
            paper["paper_id"]: paper
            for paper in papers
        }
        decision_priority = {
            "REJECT": 0,
            "MAYBE": 1,
            "KEEP": 2,
            "": 3,
        }

        # A fixed worker pool replaces eager task creation. The priority queue
        # lets newly available blind-verification work use the next free slot
        # before lower-priority primary batches, while the browser concurrency
        # cap remains unchanged.
        work_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        state_lock = asyncio.Lock()
        queue_sequence = 0
        verification_batch_number = 0
        verification_buffer: list[dict[str, Any]] = []
        verification_scheduled: set[str] = set()
        primary_phase_complete = total_primary_batches == 0

        def enqueue_work(
            *,
            stage: str,
            batch: list[dict[str, Any]],
            batch_number: int,
            priority: int,
        ) -> None:
            nonlocal queue_sequence
            queue_sequence += 1
            work_queue.put_nowait((
                int(priority),
                queue_sequence,
                stage,
                batch_number,
                batch,
            ))

        def paper_priority(paper: dict[str, Any]) -> tuple[int, int]:
            assessment = primary.get(paper["paper_id"], {})
            model_decision = str(
                assessment.get("model_decision")
                or assessment.get("decision")
                or ""
            )
            return (
                decision_priority.get(model_decision, 3),
                int(paper["order"]),
            )

        def schedule_verification_batch(
            batch: list[dict[str, Any]],
        ) -> None:
            nonlocal verification_batch_number
            if not batch:
                return
            verification_batch_number += 1
            highest_risk = min(
                paper_priority(paper)[0]
                for paper in batch
            )
            enqueue_work(
                stage="verification",
                batch=batch,
                batch_number=verification_batch_number,
                priority=highest_risk,
            )

        def add_verification_candidates(
            candidates: list[dict[str, Any]],
            *,
            flush: bool,
        ) -> None:
            for paper in candidates:
                paper_id = paper["paper_id"]
                if (
                    paper_id in verification_scheduled
                    or paper_id in verifier
                ):
                    continue
                verification_scheduled.add(paper_id)
                verification_buffer.append(paper)

            verification_buffer.sort(key=paper_priority)
            while len(verification_buffer) >= verification_batch_size:
                batch = verification_buffer[:verification_batch_size]
                del verification_buffer[:verification_batch_size]
                schedule_verification_batch(batch)

            if flush and verification_buffer:
                batch = list(verification_buffer)
                verification_buffer.clear()
                schedule_verification_batch(batch)

        def verification_candidates(
            candidate_papers: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            return [
                paper
                for paper in candidate_papers
                if (
                    paper["paper_id"] in primary
                    and paper["paper_id"] not in verifier
                    and primary[paper["paper_id"]].get("failure_class")
                    != "missing_abstract"
                    and _requires_verification(
                        primary[paper["paper_id"]],
                    )
                )
            ]

        def checkpoint_rows() -> list[dict[str, Any]]:
            return [
                _row(
                    paper,
                    primary[paper["paper_id"]],
                    verifier.get(paper["paper_id"]),
                    origin=(
                        "fresh_verification"
                        if paper["paper_id"] in verifier
                        else "fresh_primary"
                    ),
                    protocol_id=protocol_id,
                )
                for paper in papers
                if paper["paper_id"] in primary
            ]

        def persist_live_state() -> list[dict[str, Any]]:
            rows = checkpoint_rows()
            _atomic_csv(checkpoint, rows)
            persist_checkpoint_metadata()
            counts = {
                decision: sum(
                    row["Decision"] == decision
                    for row in rows
                )
                for decision in ("KEEP", "MAYBE", "REJECT")
            }
            progress.update_counts(
                job_id,
                len(primary),
                counts["KEEP"],
                counts["MAYBE"],
                counts["REJECT"],
            )
            return rows

        # Resumed rows normally already contain every required verifier result.
        # If a compatible checkpoint contains a still-risky primary assessment,
        # schedule it through the same bounded pipeline instead of waiting for
        # all new primary work.
        add_verification_candidates(
            verification_candidates(list(paper_by_id.values())),
            flush=primary_phase_complete,
        )

        for batch_number, batch in enumerate(primary_batches, start=1):
            enqueue_work(
                stage="primary",
                batch=batch,
                batch_number=batch_number,
                priority=10,
            )

        if primary_phase_complete:
            progress.begin_batches(
                job_id,
                "blind_verification",
                len(verification_scheduled),
                verification_batch_number,
                verification_batch_size,
            )
            progress.update_batch(
                job_id,
                stats["verification_batches_completed"],
                len(verifier),
            )

        async def primary_worker(
            batch: list[dict[str, Any]],
            batch_number: int,
        ) -> None:
            nonlocal primary_phase_complete

            if time.monotonic() >= finalization_deadline:
                stats["stopped_by_time_budget"] = True
                progress.update_fast_runtime(
                    job_id,
                    safety_mode=True,
                )
                values = {
                    paper["paper_id"]: _technical_maybe(
                        paper["paper_id"],
                        "Primary screening could not finish within the job budget.",
                        "time_budget",
                    )
                    for paper in batch
                }
            else:
                stats["primary_batches_submitted"] += 1
                stats["primary_papers_requested"] += len(batch)
                values = await _screen_batch(
                    browser,
                    batch,
                    rubric,
                    inputs,
                    finalization_deadline,
                    stats,
                    verification=False,
                    batch_id=f"primary_{batch_number}",
                )

            async with state_lock:
                primary.update(values)
                stats["primary_batches_completed"] += 1

                add_verification_candidates(
                    verification_candidates(batch),
                    flush=(
                        stats["primary_batches_completed"]
                        == total_primary_batches
                    ),
                )

                primary_phase_complete = (
                    stats["primary_batches_completed"]
                    == total_primary_batches
                )
                if primary_phase_complete:
                    progress.begin_batches(
                        job_id,
                        "blind_verification",
                        len(verification_scheduled),
                        verification_batch_number,
                        verification_batch_size,
                    )
                    progress.update_batch(
                        job_id,
                        stats["verification_batches_completed"],
                        len(verifier),
                    )
                else:
                    progress.update_batch(
                        job_id,
                        stats["primary_batches_completed"],
                        len(primary),
                    )

                persist_live_state()

        async def verification_worker(
            batch: list[dict[str, Any]],
            batch_number: int,
        ) -> None:
            elapsed = time.monotonic() - started
            stop_all = (
                elapsed >= min(1620, job_timeout * 0.9)
                or time.monotonic() >= finalization_deadline
            )
            low_keep_only = all(
                str(
                    primary[paper["paper_id"]].get("model_decision")
                    or ""
                ) == "KEEP"
                for paper in batch
            )
            stop_low = elapsed >= min(1440, job_timeout * 0.8)

            if stop_all or (stop_low and low_keep_only):
                stats["stopped_by_time_budget"] = True
                progress.update_fast_runtime(
                    job_id,
                    safety_mode=True,
                )
                values: dict[str, dict[str, Any]] = {}
            else:
                stats["verification_batches_submitted"] += 1
                stats["verification_papers_requested"] += len(batch)
                values = await _screen_batch(
                    browser,
                    batch,
                    rubric,
                    inputs,
                    finalization_deadline,
                    stats,
                    verification=True,
                    batch_id=f"verification_{batch_number}",
                )

            async with state_lock:
                verifier.update(values)
                stats["verification_batches_completed"] += 1
                if primary_phase_complete:
                    progress.update_batch(
                        job_id,
                        stats["verification_batches_completed"],
                        len(verifier),
                    )
                persist_live_state()

        async def scheduler_worker() -> None:
            while True:
                (
                    _priority,
                    _sequence,
                    stage,
                    batch_number,
                    batch,
                ) = await work_queue.get()
                try:
                    if stage == "primary":
                        await primary_worker(
                            batch,
                            batch_number,
                        )
                    else:
                        await verification_worker(
                            batch,
                            batch_number,
                        )
                finally:
                    work_queue.task_done()

        worker_tasks = [
            asyncio.create_task(scheduler_worker())
            for _ in range(max(1, int(concurrency)))
        ]
        join_task = asyncio.create_task(work_queue.join())
        try:
            completed, _ = await asyncio.wait(
                [join_task, *worker_tasks],
                return_when=asyncio.FIRST_COMPLETED,
            )
            failed_worker = next(
                (
                    task
                    for task in worker_tasks
                    if (
                        task.done()
                        and not task.cancelled()
                        and task.exception() is not None
                    )
                ),
                None,
            )
            if failed_worker is not None:
                raise failed_worker.exception()

            if join_task not in completed:
                raise RuntimeError(
                    "Gemini Web Fast scheduler worker stopped unexpectedly."
                )
            await join_task
        finally:
            if not join_task.done():
                join_task.cancel()
            for task in worker_tasks:
                task.cancel()
            await asyncio.gather(
                *worker_tasks,
                return_exceptions=True,
            )

        progress.begin_batches(
            job_id,
            "finalizing",
            len(papers),
            1,
            len(papers),
        )
        return primary, verifier, rubric, stats
    finally:
        await browser.close()


def _run_event_loop(coro):
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            asyncio.set_event_loop(None)
            loop.close()


def screen_csv_with_gemini_web_fast(
    *, frame: pd.DataFrame, valid: pd.DataFrame, title_col: str, abstract_col: str,
    research_question: str, research_context: str, inclusion_criteria: str,
    exclusion_criteria: str, output_path: str, job_id: str, input_fingerprint: str,
    source_dataset_fingerprint: str, resume: bool, limit: int, progress,
    screening_session, browser_factory: Callable[[], Any] = GeminiWebFastBrowser,
) -> dict[str, Any]:
    started = time.monotonic()
    primary_batch_size = _bounded_env("GEMINI_WEB_FAST_BATCH_SIZE", 10, 5, 15)
    verification_batch_size = _bounded_env("GEMINI_WEB_FAST_VERIFICATION_BATCH_SIZE", 8, 5, 10)
    concurrency = _bounded_env("GEMINI_WEB_FAST_CONCURRENCY", 3, 1, 4)
    job_timeout = _bounded_env("GEMINI_WEB_FAST_JOB_TIMEOUT_SECONDS", 1800, 60, 1800)
    inputs = {
        "question": str(research_question or ""), "context": str(research_context or ""),
        "inclusion": str(inclusion_criteria or ""), "exclusion": str(exclusion_criteria or ""),
    }
    protocol_id = _review_protocol_id(inputs)
    papers: list[dict[str, Any]] = []
    for order, (source_index, record) in enumerate(valid.iterrows()):
        papers.append({
            "paper_id": str(source_index), "order": order,
            "title": _clean_cell(record.get(title_col)),
            "abstract": _clean_cell(record.get(abstract_col)),
            "original": record.to_dict(),
        })

    identity_payload = {
        "dataset": source_dataset_fingerprint, "input": input_fingerprint,
        **inputs, "architecture": ARCHITECTURE_VERSION, "prompt": PROMPT_VERSION,
    }
    identity = sha256(json.dumps(identity_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    output = Path(output_path)
    cache_root = output.parent.parent / "cache" / "gemini_web_fast" if output.parent.name == "runs" else output.parent / ".gemini_web_fast"
    checkpoint = cache_root / "checkpoints" / f"{identity}.csv"
    checkpoint_meta = checkpoint.with_suffix(".json")
    resume_rows: dict[str, dict[str, Any]] = {}
    selected_paper_ids = {paper["paper_id"] for paper in papers}
    if resume and checkpoint.exists() and checkpoint_meta.exists():
        try:
            meta = json.loads(checkpoint_meta.read_text(encoding="utf-8"))
            if meta.get("identity") == identity and meta.get("architecture_version") == ARCHITECTURE_VERSION:
                restored = pd.read_csv(checkpoint, dtype=str, keep_default_na=False, encoding="utf-8-sig")
                candidates = {
                    str(row["Source_Row_Index"]): row.to_dict()
                    for _, row in restored.iterrows()
                }
                resume_rows = {
                    paper_id: row for paper_id, row in candidates.items()
                    if paper_id in selected_paper_ids and _reusable_resume_row(row)
                }
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            resume_rows = {}
    progress.set_resumed_count(job_id, len(resume_rows))
    browser = browser_factory()
    primary, verifier, rubric, stats = _run_event_loop(_run_fast(
        papers=papers, browser=browser, inputs=inputs, job_timeout=job_timeout,
        primary_batch_size=primary_batch_size, verification_batch_size=verification_batch_size,
        concurrency=concurrency, progress=progress, job_id=job_id, checkpoint=checkpoint,
        checkpoint_meta=checkpoint_meta, checkpoint_identity=identity, resume_rows=resume_rows,
        protocol_id=protocol_id,
    ))
    ordered = [
        _row(
            paper, primary.get(paper["paper_id"], _technical_maybe(paper["paper_id"], "Paper was not completed within the screening budget.", "time_budget")),
            verifier.get(paper["paper_id"]),
            origin="fresh_verification" if paper["paper_id"] in verifier else "fresh_primary",
            protocol_id=protocol_id,
            resumed=paper["paper_id"] in resume_rows,
        )
        for paper in papers
    ]
    _atomic_csv(checkpoint, ordered)
    final_counters = {
        "resumed_count": len(resume_rows),
        "fresh_primary_count": stats["fresh_primary_count"],
        "primary_batches_submitted": stats["primary_batches_submitted"],
        "primary_batches_completed": stats["primary_batches_completed"],
        "verification_batches_submitted": stats["verification_batches_submitted"],
        "verification_batches_completed": stats["verification_batches_completed"],
        "browser_context_started": stats["browser_context_started"],
        "pages_opened": getattr(browser, "pages_opened", 0),
        "pages_closed": getattr(browser, "pages_closed", 0),
        "peak_simultaneous_tabs": getattr(browser, "peak_active_pages", 0),
        "transport_failure_count": stats["transport_failure_count"],
    }
    _atomic_json(checkpoint_meta, {
        "identity": identity,
        "architecture_version": ARCHITECTURE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "protocol_id": protocol_id,
        "counters": final_counters,
    })
    _atomic_csv(output, ordered)
    screening_session.set_results(ordered, job_id=job_id, output_path=output_path, architecture_version=ARCHITECTURE_VERSION)
    counts = screening_session.counts(ordered)
    progress.update_counts(job_id, len(ordered), counts["keep"], counts["maybe"], counts["reject"])
    progress.finish(job_id)
    runtime = round(time.monotonic() - started, 3)
    progress.set_fast_final_metadata(
        job_id,
        primary_batch_size=primary_batch_size,
        primary_batches_submitted=stats["primary_batches_submitted"],
        primary_batches_completed=stats["primary_batches_completed"],
        verification_batches_submitted=stats["verification_batches_submitted"],
        verification_batches_completed=stats["verification_batches_completed"],
        peak_simultaneous_tabs=final_counters["peak_simultaneous_tabs"],
        resumed_count=final_counters["resumed_count"],
        fresh_primary_count=final_counters["fresh_primary_count"],
        transport_failure_count=final_counters["transport_failure_count"],
        retry_count=stats["retry_count"],
        runtime_seconds=runtime,
    )
    return {
        **counts, "total_papers": len(ordered), "output_file": output_path,
        "architecture_version": ARCHITECTURE_VERSION, "screening_engine": GEMINI_WEB_FAST_ENGINE,
        "primary_batch_size": primary_batch_size,
        "primary_batches_submitted": stats["primary_batches_submitted"],
        "verification_batches_submitted": stats["verification_batches_submitted"],
        "verification_batches_completed": stats["verification_batches_completed"],
        "primary_batches_completed": stats["primary_batches_completed"],
        "primary_papers_requested": stats["primary_papers_requested"],
        "verification_papers_requested": stats["verification_papers_requested"],
        "missing_abstract_count": sum(not paper["abstract"] for paper in papers),
        "safe_fallback_count": sum(bool(row.get("Failure_Class")) for row in ordered),
        "retry_count": stats["retry_count"], "runtime_seconds": runtime,
        "stopped_by_time_budget": stats["stopped_by_time_budget"],
        "transport_diagnostics": list(stats["transport_diagnostics"]),
        **final_counters,
        "protocol_id": protocol_id, "schema_version": SCHEMA_VERSION,
        "input_total_rows": len(frame), "screened_total_rows": len(ordered),
        "row_limit_applied": bool(limit), "row_limit_value": limit or "",
        "model_tier": "gemini_web_fast",
        "resource_profile": "web",
        "rubric": rubric.model_dump(mode="json") if rubric is not None else {},
    }
