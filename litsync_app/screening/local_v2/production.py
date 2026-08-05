from __future__ import annotations

import re
from typing import Any

from .batch import BATCH_RUNNER_VERSION
from .contracts import CriterionAssessment
from .runner import LocalV2PaperRunResult


_CRITERION_SPLIT_RE = re.compile(r"[;\r\n]+")
_CRITERION_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def _criterion_entries(value: str) -> list[str]:
    entries: list[str] = []
    for raw in _CRITERION_SPLIT_RE.split(str(value or "")):
        cleaned = _CRITERION_PREFIX_RE.sub("", raw).strip()
        if cleaned:
            entries.append(cleaned[:800])
    return entries


def build_local_v2_protocol_draft(
    *,
    research_question: str,
    research_context: str = "",
    inclusion_criteria: str = "",
    exclusion_criteria: str = "",
) -> dict[str, Any]:
    """Build a deterministic protocol draft without performing semantic inference."""

    question = str(research_question or "").strip()
    context = str(research_context or "").strip()
    inclusions = _criterion_entries(inclusion_criteria)
    exclusions = _criterion_entries(exclusion_criteria)

    criteria: list[dict[str, Any]] = []
    if inclusions:
        for index, description in enumerate(inclusions, start=1):
            criteria.append({
                "label": f"Inclusion criterion {index}",
                "role": "REQUIRED_INCLUSION",
                "description": description,
                "expected_evidence": (
                    "Explicit title or abstract evidence resolving this required condition."
                ),
                "resolution_required": True,
            })
    else:
        criteria.append({
            "label": "Research question relevance",
            "role": "REQUIRED_INCLUSION",
            "description": (
                "The paper directly addresses the supplied research question: "
                + question
            )[:800],
            "expected_evidence": (
                "Explicit title or abstract evidence that the paper addresses the "
                "research question."
            ),
            "resolution_required": True,
        })

    for index, description in enumerate(exclusions, start=1):
        criteria.append({
            "label": f"Exclusion trigger {index}",
            "role": "EXCLUSION_TRIGGER",
            "description": description,
            "expected_evidence": (
                "Explicit title or abstract evidence establishing this exclusion trigger."
            ),
            "resolution_required": False,
        })

    return {
        "research_question": question,
        "research_context": context,
        "criteria": criteria,
        "model": "local-v2",
    }


def _selected_assessments(result: LocalV2PaperRunResult) -> list[CriterionAssessment]:
    orchestration = result.orchestration
    if orchestration is None:
        return []

    stages = [
        orchestration.validator,
        orchestration.reviewer,
        orchestration.primary,
    ]
    for stage in stages:
        if stage is None or not stage.usable:
            continue
        if stage.evidence_result is not None:
            return list(stage.evidence_result.assessments)
        if stage.parse_result is not None and stage.parse_result.success:
            return list(stage.parse_result.assessments)
    return []


def local_v2_result_to_public_result(
    result: LocalV2PaperRunResult,
    *,
    resource_profile: str,
    resumed: bool,
) -> dict[str, Any]:
    """Map a Local v2 paper result into the existing production CSV result contract."""

    orchestration = result.orchestration
    assessments = _selected_assessments(result)
    evidence = [
        citation.model_dump(mode="json")
        for assessment in assessments
        for citation in assessment.evidence
    ]
    policy = result.final_policy
    route = orchestration.route if orchestration is not None else result.status
    plan = orchestration.model_plan if orchestration is not None else None
    validation_errors = list(policy.policy_errors)
    if orchestration is not None:
        for stage in (
            orchestration.primary,
            orchestration.reviewer,
            orchestration.validator,
        ):
            if stage is None:
                continue
            if stage.generation_error:
                validation_errors.append(stage.generation_error)
            if stage.parse_result is not None:
                validation_errors.extend(
                    f"{issue.code}: {issue.message}"
                    for issue in stage.parse_result.issues
                )
            if stage.evidence_result is not None:
                validation_errors.extend(
                    f"{issue.code}: {issue.message}"
                    for issue in stage.evidence_result.issues
                )

    safe_fallback = bool(result.safe_fallback or policy.safe_fallback)
    decision = policy.decision
    confidence = 0.5 if decision == "MAYBE" else (0.75 if safe_fallback else 0.95)

    return {
        "decision": decision,
        "reason": policy.reason,
        "confidence": confidence,
        "protocol_id": result.protocol_id,
        "evidence": evidence,
        "criteria": [
            assessment.model_dump(mode="json")
            for assessment in assessments
        ],
        "uncertainty": list(dict.fromkeys(
            list(policy.unresolved_criterion_ids)
            + list(result.warnings)
            + list(policy.policy_errors)
        )),
        "escalated": bool(orchestration and orchestration.review_used),
        "validation_status": "safe_fallback" if safe_fallback else "validated",
        "validation_errors": list(dict.fromkeys(validation_errors)),
        "schema_version": result.runner_version,
        "model_tier": "local_v2",
        "resource_profile": str(resource_profile or "balanced"),
        "model": plan.primary_model if plan is not None else "",
        "prompt_version": BATCH_RUNNER_VERSION,
        "processing_seconds": 0.0 if resumed else result.elapsed_seconds,
        "original_processing_seconds": result.elapsed_seconds,
        "cache_hit": bool(resumed),
        "runtime_downgrades": list(result.warnings),
        "layer_trace": [{
            "name": "local_v2",
            "route": route,
            "model_calls": result.model_call_count,
            "safe_fallback": safe_fallback,
        }],
        "layer_metrics": [{
            "processing_seconds": result.elapsed_seconds,
            "model_calls": result.model_call_count,
        }],
        "decision_risk": (
            "HIGH" if decision == "REJECT"
            else "BORDERLINE" if decision == "MAYBE"
            else "LOW"
        ),
        "triage_basis": route,
        "rq_frame_id": result.protocol_id,
        "rq_frame_version": BATCH_RUNNER_VERSION,
        "rq_frame_source": "compiled_local_v2_protocol",
        "rq_frame_status": "validated",
        "rq_frame_validation_failures": [],
        "rq_group_coverage": {},
        "local_profile": "local-v2",
        "protocol_model": "",
        "deep_model": plan.review_model if plan is not None else "",
        "edge_model": plan.validator_model if plan is not None else "",
    }
