from domain_vocabulary import join_terms


RQ_RELATIONS = {
    "review_workflow_automation": "tool_used_for_workflow",
    "method_comparison_review": "method_compared_for_task",
    "method_for_task_in_domain_review": "method_used_for_task",
    "technology_in_domain_review": "technology_used_for_outcome",
    "application_mapping_review": "applications_mapped_in_domain",
}

POSITIVE_REVIEW_RELATIONS = {
    "review_workflow_tool": "tool_used_for_workflow",
    "review_workflow_methodology": "review_workflow_methodology",
    "review_workflow_evaluation": "review_workflow_evaluation",
    "evidence_synthesis_tool": "evidence_synthesis_tool",
}

CONFLICT_REVIEW_RELATIONS = {
    "technology_subject_review": "technology_reviewed_as_subject",
    "external_domain_review": "external_domain_review",
    "review_context_only": "paper_type_only_review",
    "unrelated_to_review_workflow": "background_mention_only",
}


def desired_relation_for_rq(rq_frame):
    explicit = str((rq_frame or {}).get("rq_desired_relation", "")).strip()
    if explicit:
        return explicit
    question_type = str(
        (rq_frame or {}).get("review_question_type")
        or (rq_frame or {}).get("question_type")
        or ""
    ).strip()
    return RQ_RELATIONS.get(question_type, "unclear_relation")


def classify_paper_relation(
    rq_frame,
    paper_frame,
    review_intent,
    method_score,
    task_score,
    context_score,
):
    desired = desired_relation_for_rq(rq_frame)
    intent_relation = str(review_intent.get("review_intent_relation", ""))
    evidence = []

    workflow_use_score = 1.0 if review_intent.get("ai_tool_for_review_workflow") else 0.0
    subject_only_score = (
        float(review_intent.get("review_relation_confidence", 0.0))
        if review_intent.get("technology_subject_review_detected")
        else 0.0
    )
    paper_type_only_score = (
        float(review_intent.get("review_relation_confidence", 0.0))
        if review_intent.get("review_context_only")
        else 0.0
    )
    external_domain_score = (
        float(review_intent.get("review_relation_confidence", 0.0))
        if review_intent.get("external_domain_review_detected")
        else 0.0
    )
    implied_workflow_score = max(
        float(review_intent.get("implied_workflow_intent_score", 0.0) or 0.0),
        0.45 if review_intent.get("review_automation_intent_detected") else 0.0,
    )

    if desired == "tool_used_for_workflow":
        if intent_relation in POSITIVE_REVIEW_RELATIONS:
            observed = POSITIVE_REVIEW_RELATIONS[intent_relation]
            alignment = 1.0
            conflict = False
        elif intent_relation in CONFLICT_REVIEW_RELATIONS:
            observed = CONFLICT_REVIEW_RELATIONS[intent_relation]
            alignment = 0.0
            conflict = True
        else:
            observed = "unclear_relation"
            alignment = max(0.45, implied_workflow_score)
            conflict = False
        evidence.extend([
            review_intent.get("ai_tool_terms", ""),
            review_intent.get("review_workflow_task_terms", ""),
            review_intent.get("review_automation_intent_terms", ""),
            review_intent.get("review_context_terms", ""),
        ])
    else:
        review_role = str(paper_frame.get("review_role", "")).strip()
        if method_score >= 0.60 and task_score >= 0.60 and context_score >= 0.60:
            observed = desired
            alignment = 1.0
            conflict = False
        elif review_role == "technology_being_reviewed":
            observed = "technology_reviewed_as_subject"
            alignment = 0.0
            conflict = desired != "technology_reviewed_as_subject"
        elif sum(score >= 0.60 for score in (method_score, task_score, context_score)) >= 2:
            observed = "unclear_relation"
            alignment = 0.55
            conflict = False
        else:
            observed = "background_mention_only"
            alignment = 0.0
            conflict = False
        evidence.extend([
            paper_frame.get("intervention_or_method", ""),
            paper_frame.get("target_problem_or_task", ""),
            paper_frame.get("application_context", ""),
        ])

    confidence = float(review_intent.get("review_relation_confidence", 0.0) or 0.0)
    if desired != "tool_used_for_workflow":
        confidence = 0.90 if alignment == 1.0 else 0.60 if alignment > 0 else 0.75

    contradiction_score = max(subject_only_score, paper_type_only_score, external_domain_score)
    mismatch_reason = ""
    if conflict:
        mismatch_reason = f"observed {observed}, but the RQ requires {desired}"

    return {
        "rq_desired_relation": desired,
        "paper_observed_relation": observed,
        "relation_match": alignment >= 0.75,
        "relation_conflict": conflict,
        "relation_alignment_score": round(alignment, 4),
        "relation_confidence": round(confidence, 4),
        "relation_evidence_terms": join_terms(evidence),
        "relation_negative_terms": join_terms([
            review_intent.get("external_domain_terms", ""),
            "technology subject only" if subject_only_score else "",
            "paper type only" if paper_type_only_score else "",
        ]),
        "relation_mismatch_reason": mismatch_reason,
        "workflow_use_score": workflow_use_score,
        "subject_only_score": round(subject_only_score, 4),
        "paper_type_only_score": round(paper_type_only_score, 4),
        "external_domain_topic_score": round(external_domain_score, 4),
        "implied_workflow_score": round(implied_workflow_score, 4),
        "uncertainty_preservation_score": round(
            max(alignment, implied_workflow_score) if not conflict else 0.0,
            4,
        ),
        "contradiction_score": round(contradiction_score, 4),
        "relation_decision_path": (
            "relation_exact_keep" if alignment == 1.0
            else "relation_uncertain_maybe" if alignment > 0.0
            else "relation_conflict_reject" if conflict
            else "relation_weak_reject"
        ),
        "review_automation_relevance_score": round(
            (0.35 * workflow_use_score)
            + (0.25 * implied_workflow_score)
            + (0.40 * alignment)
            - (0.50 * contradiction_score),
            4,
        ),
    }
