import hashlib

from domain_vocabulary import split_terms


RQ_TYPE_MAP = {
    "method_for_task_in_domain_review": "method_for_task_in_domain",
    "method_comparison_review": "method_comparison_review",
    "technology_in_domain_review": "technology_for_outcome_in_domain",
    "review_workflow_automation": "review_workflow_automation",
    "application_mapping_review": "application_mapping_review",
    "security_risk_review": "security_risk_review",
    "mapping_scoping_review": "broad_scoping_review",
    "domain_literature_review": "broad_scoping_review",
}


def _items(value):
    return list(dict.fromkeys(split_terms(value)))


def build_rq_contract(rq_text, frame):
    rq_text = str(rq_text or "")
    source_type = str(
        frame.get("review_question_type") or frame.get("question_type") or ""
    )
    rq_type = RQ_TYPE_MAP.get(source_type, source_type or "broad_scoping_review")
    required = _items(frame.get("required_dimensions"))
    method_terms = _items(
        frame.get("method_or_technology") or frame.get("intervention_or_method")
    )
    task_terms = _items(
        frame.get("target_tasks_or_outcomes") or frame.get("target_problem_or_task")
    )
    context_terms = _items(frame.get("application_context") or frame.get("core_domain"))
    suspect = bool(
        not method_terms
        or not task_terms
        or not context_terms
        or str(frame.get("rq_extraction_suspect", "")).lower() == "true"
    )
    return {
        "rq_id": hashlib.sha1(rq_text.encode("utf-8")).hexdigest()[:12],
        "rq_text": rq_text,
        "rq_type": rq_type,
        "rq_desired_relation": frame.get("rq_desired_relation", "unclear_relation"),
        "rq_scope_width": "broad" if rq_type == "broad_scoping_review" else "focused",
        "rq_strictness": "strict" if rq_type == "review_workflow_automation" else "balanced",
        "rq_required_dimensions": required,
        "rq_optional_dimensions": ["evidence_type"],
        "rq_method_terms": method_terms,
        "rq_method_families": _items(frame.get("method_family")),
        "rq_task_terms": task_terms,
        "rq_task_families": [],
        "rq_context_terms": context_terms,
        "rq_context_families": [],
        "rq_outcome_terms": task_terms,
        "rq_evidence_types_expected": _items(frame.get("expected_evidence_types")),
        "rq_inclusion_concepts": _items(frame.get("required_inclusion_concepts")),
        "rq_exclusion_concepts": _items(frame.get("exclusion_concepts")),
        "rq_positive_relation_patterns": [frame.get("rq_desired_relation", "")],
        "rq_negative_relation_patterns": [
            "external_domain_review",
            "paper_type_only_review",
            "background_mention_only",
        ],
        "rq_ambiguity_policy": "preserve_maybe",
        "rq_stage2_policy": "explicit_contradiction_required_for_demotion",
        "rq_extraction_suspect": suspect,
    }


def build_paper_contract(title, abstract, frame, relation=None):
    relation = relation or {}
    title = title or frame.get("source_title", "")
    abstract = abstract or frame.get("source_abstract", "")
    text_length = len(str(title or "").strip()) + len(str(abstract or "").strip())
    return {
        "paper_id": hashlib.sha1(
            f"{title or ''}\n{abstract or ''}".encode("utf-8")
        ).hexdigest()[:12],
        "title": str(title or ""),
        "abstract": str(abstract or ""),
        "paper_methods": _items(
            frame.get("methods_or_technologies") or frame.get("intervention_or_method")
        ),
        "paper_method_families": _items(frame.get("method_family")),
        "paper_specific_models": _items(frame.get("specific_models_or_systems")),
        "paper_tasks": _items(
            frame.get("target_tasks_or_outcomes") or frame.get("target_problem_or_task")
        ),
        "paper_task_families": _items(frame.get("task_family")),
        "paper_contexts": _items(
            frame.get("application_contexts") or frame.get("application_context")
        ),
        "paper_context_families": _items(frame.get("context_family")),
        "paper_outcomes": _items(frame.get("target_tasks_or_outcomes")),
        "paper_evidence_type": frame.get("evidence_type", ""),
        "paper_study_role": frame.get("study_role", ""),
        "paper_contribution_type": frame.get("contribution_type", ""),
        "paper_relation_candidates": [relation.get("paper_observed_relation", "")],
        "paper_observed_relation": relation.get("paper_observed_relation", ""),
        "paper_external_domain_terms": _items(relation.get("external_domain_terms", "")),
        "paper_inclusion_cues": _items(frame.get("inclusion_cues")),
        "paper_exclusion_cues": _items(frame.get("exclusion_cues")),
        "paper_uncertainty_cues": [],
        "paper_text_quality_score": min(1.0, text_length / 500.0),
    }
