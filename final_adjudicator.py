from __future__ import annotations

import re
from typing import Any


FINAL_ADJUDICATION_FIELDS = {
    "final_adjudicated_decision": "",
    "final_adjudication_action": "not_run",
    "final_adjudication_reason": "",
    "final_relation": "",
    "final_relation_confidence": 0.0,
    "final_workflow_use": False,
    "final_external_domain": False,
    "final_uncertainty_reason": "",
    "relation_evidence_valid": False,
    "relation_validation_warning": "",
    "workflow_quote_valid": False,
    "external_domain_quote_valid": False,
    "task_object_type": "unclear",
    "validated_directional_relation": "",
    "validated_directional_confidence": 0.0,
}

REVIEW_OBJECT_TERMS = (
    "abstract", "abstracts", "citation", "citations", "paper", "papers",
    "article", "articles", "study selection", "selection of studies",
    "included studies", "excluded studies", "title/abstract", "title abstract",
    "screening", "literature search", "search strategy", "data extraction",
    "risk of bias", "quality assessment", "evidence synthesis", "review process",
    "systematic review process", "review workflow", "systematic review workflow",
    "review assistant", "research assistant for literature review",
    "slr automation", "systematic literature review automation",
    "systematic review automation", "review update", "review updates",
    "reviewer workload", "inclusion/exclusion",
)

EXTERNAL_OBJECT_TERMS = (
    "patient", "patients", "image", "images", "diagnosis", "diagnostic",
    "disease", "cancer", "breast cancer", "medicine", "medical", "clinical",
    "student", "students", "education", "financial", "finance", "fintech",
    "transaction", "transactions", "software artifact", "software artifacts",
    "software refactoring", "software engineering", "cybersecurity",
    "security event", "security events", "drug discovery", "healthcare",
    "product", "products", "code smell", "requirements engineering",
)

RECALL_GUARD_WORKFLOW_TERMS = (
    "systematic review automation",
    "systematic literature review automation",
    "literature review automation",
    "automating systematic reviews",
    "automating systematic literature reviews",
    "automate systematic reviews",
    "automate systematic literature reviews",
    "automating literature reviews",
    "automate literature reviews",
    "automate and facilitate the review process",
    "facilitate the review process",
    "automating the review process",
    "automate the review process",
    "review process automation",
    "title/abstract screening",
    "title abstract screening",
    "citation screening",
    "study selection",
    "selection of studies",
    "literature search",
    "search strategy",
    "review stages",
    "slr methodology",
    "llm-assisted methodology",
    "accelerating systematic reviews",
    "accelerate systematic reviews",
    "accelerating systematic literature reviews",
    "accelerate systematic literature reviews",
    "evaluating llm tools across key systematic literature review stages",
    "data extraction from studies",
    "screening sensitivity",
    "screening recall",
    "reviewer workload reduction",
    "screening phase",
    "identification of relevant articles",
)

RECALL_GUARD_AI_TERMS = (
    "artificial intelligence",
    "ai",
    "machine learning",
    "ml",
    "large language model",
    "large language models",
    "llm",
    "llms",
    "generative ai",
    "automation",
    "automated",
)

RECALL_GUARD_KEEP_TERMS = (
    "systematic review automation",
    "systematic literature review automation",
    "literature review automation",
    "automating systematic reviews",
    "automating systematic literature reviews",
    "automate systematic reviews",
    "automate systematic literature reviews",
    "automating literature reviews",
    "automate literature reviews",
    "automate and facilitate the review process",
    "facilitate the review process",
    "automating the review process",
    "automate the review process",
    "review process automation",
    "title/abstract screening",
    "title abstract screening",
    "citation screening",
    "study selection",
    "selection of studies",
    "literature search",
    "review stages",
    "slr methodology",
    "llm-assisted methodology",
    "accelerating systematic reviews",
    "accelerate systematic reviews",
    "accelerating systematic literature reviews",
    "accelerate systematic literature reviews",
    "screening sensitivity",
    "screening recall",
    "reviewer workload reduction",
)

REVIEW_METHOD_PROCESS_TERMS = (
    "review process",
    "review workflow",
    "review protocol",
    "title/abstract",
    "title abstract",
    "abstract screening",
    "citation screening",
    "study selection",
    "selection of studies",
    "selected studies",
    "screened articles",
    "screened papers",
    "screening over",
    "screening phase",
    "inclusion criteria",
    "exclusion criteria",
    "inclusion/exclusion",
    "data extraction",
    "risk of bias",
    "quality assessment",
    "literature search",
    "search strategy",
    "database search",
    "database searches",
    "prisma",
)


def adjudicate_row(row: dict[str, Any]) -> dict[str, Any]:
    """Validate directional relation evidence and make a final SLR decision pass."""
    current_decision = _decision(row.get("Decision"))
    output = dict(FINAL_ADJUDICATION_FIELDS)
    output["final_adjudicated_decision"] = current_decision

    if not _is_review_workflow_rq(row):
        output["final_adjudication_action"] = "not_applicable"
        output["final_adjudication_reason"] = "Final adjudication skipped for non-review-workflow RQ."
        return output

    candidates = [
        _validate_directional_evidence(row, "stage1_"),
        _validate_directional_evidence(row, "stage2_"),
    ]
    chosen = _choose_candidate(candidates)
    output.update({
        "relation_evidence_valid": chosen["relation_evidence_valid"],
        "relation_validation_warning": chosen["relation_validation_warning"],
        "workflow_quote_valid": chosen["workflow_quote_valid"],
        "external_domain_quote_valid": chosen["external_domain_quote_valid"],
        "task_object_type": chosen["task_object_type"],
        "validated_directional_relation": chosen["validated_directional_relation"],
        "validated_directional_confidence": chosen["validated_directional_confidence"],
        "final_relation": chosen["validated_directional_relation"],
        "final_relation_confidence": chosen["validated_directional_confidence"],
        "final_workflow_use": chosen["workflow_use"],
        "final_external_domain": chosen["external_domain"],
        "final_uncertainty_reason": chosen["uncertainty_reason"],
    })

    confidence = chosen["validated_directional_confidence"]
    if chosen["workflow_use"] and chosen["relation_evidence_valid"] and confidence >= 0.70:
        if current_decision == "REJECT":
            output["final_adjudicated_decision"] = "MAYBE"
            output["final_adjudication_action"] = "rescue_reject_to_maybe"
            output["final_adjudication_reason"] = (
                "Validated workflow-use evidence found after deterministic rejection."
            )
            return output
        if current_decision == "MAYBE" and confidence >= 0.80:
            output["final_adjudicated_decision"] = "KEEP"
            output["final_adjudication_action"] = "promote_maybe_to_keep"
            output["final_adjudication_reason"] = (
                "Validated strong evidence that AI/LLM is used for review workflow tasks."
            )
            return output
        output["final_adjudication_action"] = "confirm_workflow"
        output["final_adjudication_reason"] = "Validated workflow-use relation preserved."
        return output

    recall_guard = _strong_workflow_recall_guard(row, chosen, current_decision)
    if recall_guard:
        output["final_adjudicated_decision"] = recall_guard["decision"]
        output["final_adjudication_action"] = recall_guard["action"]
        output["final_adjudication_reason"] = recall_guard["reason"]
        return output

    if chosen["external_domain"] and chosen["relation_evidence_valid"] and confidence >= 0.70:
        if (
            current_decision == "REJECT"
            and chosen["task_object_type"] == "unclear"
            and chosen["external_domain_quote_valid"]
            and _has_review_process_ambiguity(row)
        ):
            output["final_adjudicated_decision"] = "MAYBE"
            output["final_adjudication_action"] = "external_domain_uncertain_reject_to_maybe"
            output["final_adjudication_reason"] = (
                "External-domain direction is likely, but task-object evidence is mixed; "
                "hard rejection softened to uncertainty."
            )
            return output
        if current_decision == "KEEP":
            output["final_adjudicated_decision"] = "MAYBE"
            output["final_adjudication_action"] = "demote_keep_to_maybe"
            output["final_adjudication_reason"] = (
                "Validated external-domain AI review evidence conflicts with workflow-use KEEP."
            )
            return output
        output["final_adjudication_action"] = "confirm_external_domain"
        output["final_adjudication_reason"] = "Validated external-domain relation preserved."
        return output

    if current_decision == "KEEP" and chosen["relation_validation_warning"]:
        output["final_adjudicated_decision"] = "MAYBE"
        output["final_adjudication_action"] = "demote_keep_uncertain_relation"
        output["final_adjudication_reason"] = chosen["relation_validation_warning"]
        return output

    output["final_adjudication_action"] = "preserve"
    output["final_adjudication_reason"] = (
        chosen["relation_validation_warning"]
        or "No validated directional evidence strong enough to alter final decision."
    )
    return output


def _strong_workflow_recall_guard(
    row: dict[str, Any],
    chosen: dict[str, Any],
    current_decision: str,
) -> dict[str, str] | None:
    if current_decision not in {"REJECT", "MAYBE"}:
        return None

    workflow_text = _workflow_recall_text(row)
    full_text = workflow_text.lower()
    if not _has_any(full_text, RECALL_GUARD_AI_TERMS):
        return None
    if not _has_any(full_text, RECALL_GUARD_WORKFLOW_TERMS):
        return None

    workflow_object_evidence = (
        chosen["workflow_quote_valid"]
        or chosen["task_object_type"] == "review_object"
        or _has_any(str(row.get("stage1_positive_workflow_candidate_terms") or ""), RECALL_GUARD_WORKFLOW_TERMS)
        or _has_any(full_text, RECALL_GUARD_WORKFLOW_TERMS)
    )
    if not workflow_object_evidence:
        return None

    strong_external_contradiction = (
        chosen["external_domain"]
        and chosen["relation_evidence_valid"]
        and chosen["external_domain_quote_valid"]
        and chosen["task_object_type"] == "external_domain_object"
        and not chosen["workflow_quote_valid"]
    )
    if strong_external_contradiction:
        return {
            "decision": current_decision,
            "action": "recall_guard_preserve_external_domain",
            "reason": "Strong workflow recall guard found an external-domain contradiction; decision preserved.",
        }

    if current_decision == "REJECT":
        return {
            "decision": "MAYBE",
            "action": "recall_guard_reject_to_maybe",
            "reason": "Strong AI-for-review-workflow evidence prevents hard rejection.",
        }

    very_strong = _has_any(
        " ".join(
            str(row.get(name) or "")
            for name in (
                "Title",
                "Abstract",
                "stage1_positive_workflow_candidate_terms",
                "stage1_workflow_evidence_quote",
                "stage1_llm_workflow_tasks_detected",
            )
        ),
        RECALL_GUARD_KEEP_TERMS,
    )
    if current_decision == "MAYBE" and very_strong:
        return {
            "decision": "KEEP",
            "action": "recall_guard_maybe_to_keep",
            "reason": "Very strong AI-for-review-workflow evidence supports KEEP.",
        }

    return None


def _has_review_process_ambiguity(row: dict[str, Any]) -> bool:
    text = " ".join(
        str(row.get(name) or "")
        for name in (
            "Title",
            "Abstract",
            "stage1_workflow_evidence_quote",
            "stage1_positive_workflow_candidate_terms",
            "stage1_llm_workflow_tasks_detected",
            "stage1_directional_reason",
            "stage1_llm_judge_reason",
            "stage1_paper_target_problem_or_task",
            "stage2_workflow_evidence_quote",
            "stage2_positive_workflow_candidate_terms",
            "stage2_llm_workflow_tasks_detected",
            "stage2_directional_reason",
            "stage2_llm_judge_reason",
            "stage2_paper_target_problem_or_task",
        )
    )
    return _has_any(text, REVIEW_METHOD_PROCESS_TERMS)


def _workflow_recall_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(name) or "")
        for name in (
            "Title",
            "Abstract",
            "stage1_positive_workflow_candidate_terms",
            "stage1_workflow_evidence_quote",
            "stage1_llm_workflow_tasks_detected",
            "stage1_directional_reason",
            "stage1_llm_judge_reason",
            "stage1_paper_target_problem_or_task",
            "stage1_paper_application_context",
            "stage2_positive_workflow_candidate_terms",
            "stage2_workflow_evidence_quote",
            "stage2_llm_workflow_tasks_detected",
            "stage2_directional_reason",
            "stage2_llm_judge_reason",
            "stage2_paper_target_problem_or_task",
            "stage2_paper_application_context",
        )
    )


def _validate_directional_evidence(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    relation = str(row.get(f"{prefix}directional_relation") or row.get(f"{prefix}llm_judge_relation") or "")
    confidence = _float(
        row.get(f"{prefix}directional_confidence"),
        row.get(f"{prefix}relation_confidence"),
        row.get(f"{prefix}llm_relation_confidence"),
    )
    workflow = _bool(row.get(f"{prefix}directional_uses_ai_for_review_workflow")) or relation == "ai_tool_for_review_workflow"
    external = _bool(row.get(f"{prefix}directional_is_review_about_ai_external_domain")) or relation == "review_about_ai_external_domain"
    workflow_text = " ".join(
        str(row.get(name) or "")
        for name in (
            f"{prefix}workflow_evidence_quote",
            f"{prefix}llm_workflow_tasks_detected",
            f"{prefix}positive_workflow_candidate_terms",
            f"{prefix}directional_reason",
            f"{prefix}llm_judge_reason",
        )
    )
    external_text = " ".join(
        str(row.get(name) or "")
        for name in (
            f"{prefix}external_domain_evidence_quote",
            f"{prefix}llm_external_domain_tasks_detected",
            f"{prefix}external_domain_candidate_terms",
            f"{prefix}directional_reason",
            f"{prefix}llm_judge_reason",
        )
    )
    task_object_type = str(row.get(f"{prefix}task_object_type") or row.get("task_object_type") or "unclear")
    workflow_quote_valid = _has_any(workflow_text, REVIEW_OBJECT_TERMS)
    external_quote_valid = _has_any(external_text, EXTERNAL_OBJECT_TERMS)

    if task_object_type not in {"review_object", "external_domain_object", "unclear"}:
        task_object_type = "unclear"
    if task_object_type == "unclear":
        if workflow_quote_valid and not external_quote_valid:
            task_object_type = "review_object"
        elif external_quote_valid and not workflow_quote_valid:
            task_object_type = "external_domain_object"

    warning = ""
    relation_valid = False
    if workflow:
        relation_valid = workflow_quote_valid or task_object_type == "review_object"
        if not relation_valid:
            warning = "Workflow relation lacks review-process object evidence."
        elif external_quote_valid and not workflow_quote_valid:
            warning = "Workflow relation conflicts with stronger external-domain object evidence."
            relation_valid = False
    elif external:
        relation_valid = external_quote_valid or task_object_type == "external_domain_object"
        if not relation_valid:
            warning = "External-domain relation lacks domain-object evidence."
        elif workflow_quote_valid and not external_quote_valid:
            warning = "External-domain relation conflicts with stronger review-workflow object evidence."
            relation_valid = False
    elif confidence > 0.0:
        warning = "Directional relation is unclear."

    if not relation:
        relation = "unclear"

    return {
        "prefix": prefix,
        "validated_directional_relation": relation,
        "validated_directional_confidence": confidence,
        "workflow_use": bool(workflow),
        "external_domain": bool(external),
        "workflow_quote_valid": bool(workflow_quote_valid),
        "external_domain_quote_valid": bool(external_quote_valid),
        "task_object_type": task_object_type,
        "relation_evidence_valid": bool(relation_valid),
        "relation_validation_warning": warning,
        "uncertainty_reason": str(row.get(f"{prefix}uncertainty_reason") or ""),
    }


def _choose_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in candidates if item["relation_evidence_valid"]]
    if valid:
        return max(valid, key=lambda item: (item["validated_directional_confidence"], item["prefix"] == "stage2_"))
    return max(candidates, key=lambda item: (item["validated_directional_confidence"], item["prefix"] == "stage2_"))


def _is_review_workflow_rq(row: dict[str, Any]) -> bool:
    values = " ".join(
        str(row.get(name) or "")
        for name in (
            "stage1_rq_review_question_type",
            "stage1_rq_question_type",
            "stage1_rq_type",
            "stage1_rq_desired_relation",
            "stage2_rq_review_question_type",
            "stage2_rq_question_type",
            "stage2_rq_type",
            "stage2_rq_desired_relation",
        )
    ).lower()
    if "review_workflow_automation" in values or "tool_used_for_workflow" in values:
        return True
    return _has_any(
        " ".join(str(row.get(name) or "") for name in ("Title", "Abstract")),
        ("systematic review automation", "systematic literature review automation", "review workflow"),
    )


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    normalized = str(text or "").lower()
    return any(re.search(r"\b" + re.escape(term.lower()) + r"\b", normalized) for term in terms)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _float(*values: Any) -> float:
    for value in values:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            continue
    return 0.0


def _decision(value: Any) -> str:
    decision = str(value or "").upper()
    return decision if decision in {"KEEP", "MAYBE", "REJECT"} else "MAYBE"
