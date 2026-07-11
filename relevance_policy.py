from domain_vocabulary import (
    analyze_paper_text,
    contains_phrase,
    find_terms,
    join_terms,
    split_terms,
)
from review_workflow_ontology import classify_review_intent
from relation_classifier import classify_paper_relation
from decision_policy import DEFAULT_DECISION_POLICY
from evidence_ledger import build_evidence_ledger
from screening_contracts import build_paper_contract, build_rq_contract
from model_score_fusion import MODEL_DIAGNOSTIC_FIELDS, apply_model_score_fusion


USEFUL_EVIDENCE = {
    "application",
    "architecture",
    "benchmark",
    "case study",
    "empirical",
    "evaluation",
    "framework",
    "model",
    "platform",
    "proposal",
    "prototype",
    "review",
    "survey",
    "system",
}


def _overlap_score(left_terms, right_text):
    left = split_terms(left_terms)
    if not left:
        return 0.0, []
    matches = find_terms(right_text, left)
    return min(1.0, len(matches) / max(1, min(len(left), 5))), matches


def _max_score(*scores):
    return max([0.0, *[float(score or 0.0) for score in scores]])


def _dimension_evidence(rq_frame, paper_text, direct_fields, synonym_field, corpus_field):
    direct_terms = []
    for field in direct_fields:
        direct_terms.extend(split_terms(rq_frame.get(field, "")))
    synonym_terms = split_terms(rq_frame.get(synonym_field, ""))
    corpus_terms = split_terms(rq_frame.get(corpus_field, ""))
    direct_hits = find_terms(paper_text, direct_terms)
    synonym_hits = find_terms(paper_text, synonym_terms)
    corpus_hits = find_terms(paper_text, corpus_terms)
    hits = list(dict.fromkeys(direct_hits + synonym_hits + corpus_hits))
    if direct_hits:
        strength = 1.0
    elif synonym_hits:
        strength = 0.8
    elif corpus_hits:
        strength = 0.6
    else:
        strength = 0.0
    return strength, hits


def _required_dimension_names(rq_frame):
    dimensions = split_terms(rq_frame.get("required_dimensions", ""))
    return dimensions or ["method", "task_or_outcome", "context"]


def apply_relevance_policy(rq_frame, paper_frame, base_result):
    rich_rq_frame = any(
        rq_frame.get(field)
        for field in (
            "review_question_type",
            "method_synonyms",
            "domain_synonyms",
            "task_outcome_synonyms",
            "required_inclusion_concepts",
            "required_dimensions",
        )
    )
    title_text = str(paper_frame.get("primary_subject", ""))
    paper_text = " ".join(
        str(paper_frame.get(field, ""))
        for field in (
            "primary_subject",
            "intervention_or_method",
            "target_problem_or_task",
            "application_context",
            "evidence_type",
            "methods_or_technologies",
            "target_tasks_or_outcomes",
            "application_contexts",
            "inclusion_cues",
        )
    )
    paper_analysis = analyze_paper_text("", paper_text)

    method_text = " ".join(
        str(paper_frame.get(field, ""))
        for field in ("primary_subject", "intervention_or_method", "methods_or_technologies")
    )
    task_text = " ".join(
        str(paper_frame.get(field, ""))
        for field in ("primary_subject", "target_problem_or_task", "target_tasks_or_outcomes")
    )
    context_text = " ".join(
        str(paper_frame.get(field, ""))
        for field in ("primary_subject", "application_context", "application_contexts", "main_domain")
    )
    method_score, method_hits = _dimension_evidence(
        rq_frame,
        method_text,
        ("method_or_technology", "intervention_or_method"),
        "method_synonyms",
        "corpus_method_terms",
    )
    task_score, task_hits = _dimension_evidence(
        rq_frame,
        task_text,
        ("target_tasks_or_outcomes", "target_problem_or_task"),
        "task_outcome_synonyms",
        "corpus_task_terms",
    )
    context_score, context_hits = _dimension_evidence(
        rq_frame,
        context_text,
        ("application_context", "core_domain"),
        "context_synonyms",
        "corpus_context_terms",
    )
    evidence_score, evidence_hits = _overlap_score(
        rq_frame.get("expected_evidence_types", ""),
        paper_text,
    )
    negative_score, negative_hits = _overlap_score(
        rq_frame.get("negative_contexts", "") or rq_frame.get("exclusion_concepts", ""),
        paper_text,
    )

    # Embeddings can support partial evidence, but cannot erase explicit evidence.
    method_score = _max_score(method_score, 0.45 * _max_score(base_result.get("effective_technology_match")))
    task_score = _max_score(task_score, 0.45 * _max_score(base_result.get("task_match")))
    context_score = _max_score(context_score, 0.45 * _max_score(base_result.get("context_match")))
    evidence_score = _max_score(evidence_score, 0.50 if evidence_hits else 0.0)

    question_type = str(rq_frame.get("review_question_type") or rq_frame.get("question_type") or "")
    review_intent = classify_review_intent(paper_frame)
    if question_type == "review_workflow_automation":
        task_score = _max_score(
            task_score,
            1.0 if review_intent["review_workflow_task_detected"] else 0.0,
        )
        method_score = _max_score(
            method_score,
            1.0 if review_intent["ai_tool_terms"] else 0.0,
        )
        context_score = _max_score(
            context_score,
            1.0 if review_intent["review_context_terms"] else 0.0,
        )
    relation = classify_paper_relation(
        rq_frame,
        paper_frame,
        review_intent,
        method_score,
        task_score,
        context_score,
    )
    review_gate_violation = (
        question_type == "review_workflow_automation"
        and review_intent["review_intent_relation"] in {
            "technology_subject_review",
            "external_domain_review",
            "review_context_only",
        }
    )
    review_gate_reason = ""
    if review_gate_violation:
        review_gate_reason = review_intent["review_intent_relation"]
    elif question_type == "review_workflow_automation" and review_intent["ai_tool_for_review_workflow"]:
        review_gate_reason = "AI/tool is explicitly connected to a review workflow task"
    explicit_negative = bool(negative_hits)
    technology_only = bool(method_hits and not task_hits and not context_hits)
    contradiction_detected = review_gate_violation or explicit_negative or technology_only
    contradiction_reason = ""
    if review_gate_violation:
        contradiction_reason = "wrong review role"
    elif explicit_negative:
        contradiction_reason = "explicit exclusion concept: " + join_terms(negative_hits)
    elif technology_only:
        contradiction_reason = "technology mentioned without target task or application context"

    inclusion_evidence = []
    if method_hits:
        inclusion_evidence.append("method: " + join_terms(method_hits))
    if task_hits:
        inclusion_evidence.append("task/outcome: " + join_terms(task_hits))
    if context_hits:
        inclusion_evidence.append("context: " + join_terms(context_hits))
    if evidence_hits:
        inclusion_evidence.append("evidence: " + join_terms(evidence_hits))

    exclusion_evidence = []
    if negative_hits:
        exclusion_evidence.append("negative context: " + join_terms(negative_hits))
    if review_gate_violation:
        exclusion_evidence.append("review-role mismatch")

    required_dimensions = _required_dimension_names(rq_frame)
    dimension_scores = {
        "method": method_score,
        "method_tool": method_score,
        "task_or_outcome": task_score,
        "context": context_score,
        "review_workflow_task": task_score,
        "review_workflow_process": task_score,
        "systematic_review_context": context_score,
        "review_context": context_score,
        "relation": relation["relation_alignment_score"],
    }
    met = [name for name in required_dimensions if dimension_scores.get(name, 0.0) >= 0.60]
    missing = [name for name in required_dimensions if name not in met]
    coverage = len(met)
    strongest_inclusion = max(
        (
            ("method: " + join_terms(method_hits), method_score),
            ("task/outcome: " + join_terms(task_hits), task_score),
            ("context: " + join_terms(context_hits), context_score),
        ),
        key=lambda item: item[1],
    )[0]
    strongest_exclusion = contradiction_reason

    final_score = (
        (0.35 * method_score)
        + (0.25 * task_score)
        + (0.25 * context_score)
        + (0.15 * evidence_score)
        - (0.40 * negative_score)
    )
    final_score = max(0.0, min(1.0, final_score))

    decision = None
    path = ""
    uncertainty = ""
    rejection_category = ""

    broad_review = question_type in {"mapping_scoping_review", "domain_literature_review"}
    if review_gate_violation:
        decision = "REJECT"
        path = "evidence_first_review_gate_reject"
        relation_reasons = {
            "technology_subject_review": "AI is only the subject being reviewed",
            "external_domain_review": "external-domain AI systematic review",
            "review_context_only": "systematic review paper type only",
        }
        rejection_category = relation_reasons.get(
            review_intent["review_intent_relation"],
            "review-role mismatch",
        )
    elif (
        question_type == "review_workflow_automation"
        and review_intent["ai_tool_for_review_workflow"]
        and coverage == len(required_dimensions)
    ):
        decision = "KEEP"
        path = "evidence_first_review_workflow_tool_keep"
    elif question_type == "review_workflow_automation" and coverage == 2:
        decision = "MAYBE"
        path = "evidence_first_review_workflow_unclear_maybe"
        uncertainty = "AI/tool connection to the review workflow is incomplete"
    elif question_type == "review_workflow_automation" and coverage <= 1:
        decision = "REJECT"
        path = "evidence_first_review_workflow_insufficient_reject"
        missing_reasons = {
            "method_tool": "no AI/tool method evidence",
            "review_workflow_process": "no review workflow process evidence",
            "review_context": "no review context",
        }
        rejection_category = join_terms(
            missing_reasons.get(name, "insufficient evidence")
            for name in missing
        )
    elif explicit_negative and coverage < len(required_dimensions):
        decision = "REJECT"
        path = "evidence_first_explicit_contradiction_reject"
        rejection_category = "explicit contradiction"
    elif coverage == len(required_dimensions):
        decision = "KEEP"
        path = "evidence_first_all_dimensions_keep"
    elif coverage == len(required_dimensions) - 1:
        decision = "MAYBE"
        path = "evidence_first_two_dimensions_maybe"
        uncertainty = "required dimension unclear: " + join_terms(missing)
    elif broad_review and coverage >= 1:
        decision = "MAYBE"
        path = "evidence_first_broad_review_adjacent_maybe"
        uncertainty = "broad review has adjacent evidence but incomplete dimension coverage"
    elif coverage <= 1:
        decision = "REJECT"
        path = "evidence_first_insufficient_dimensions_reject"
        rejection_category = "missing required dimensions: " + join_terms(missing)

    desired_tool_relation = relation["rq_desired_relation"] == "tool_used_for_workflow"
    external_domain_subject_only = bool(
        review_intent["external_domain_review_detected"]
        and not review_intent["ai_tool_for_review_workflow"]
    )
    workflow_use_missing = relation["workflow_use_score"] < 0.75
    keep_required_relation_missing = not relation["relation_match"]
    subject_only_overrides_keep = bool(
        review_intent["technology_subject_review_detected"]
        and workflow_use_missing
    )
    relation_keep_gate_passed = bool(
        not desired_tool_relation
        or (
            relation["relation_match"]
            and relation["workflow_use_score"] >= 0.75
            and not external_domain_subject_only
            and not subject_only_overrides_keep
        )
    )
    false_keep_risk = 0.0
    if desired_tool_relation:
        false_keep_risk = max(
            relation["subject_only_score"],
            relation["paper_type_only_score"],
            relation["external_domain_topic_score"],
            0.65 if workflow_use_missing else 0.0,
            0.60 if keep_required_relation_missing else 0.0,
        )
    keep_suppression_applied = bool(
        desired_tool_relation
        and not relation_keep_gate_passed
        and false_keep_risk >= 0.60
    )
    keep_suppression_reason = ""
    if keep_suppression_applied:
        if external_domain_subject_only:
            keep_suppression_reason = (
                "The paper is an external-domain AI review without evidence that "
                "AI is used to conduct the review workflow."
            )
        elif review_intent["review_context_only"]:
            keep_suppression_reason = (
                "Systematic review is only the paper type; workflow-tool use is absent."
            )
        elif subject_only_overrides_keep:
            keep_suppression_reason = (
                "AI/LLM is the reviewed subject rather than a tool used for the review workflow."
            )
        else:
            keep_suppression_reason = (
                "The required tool-used-for-workflow relation is missing or unclear."
            )

    if decision == "KEEP" and keep_suppression_applied:
        if external_domain_subject_only:
            decision = "REJECT"
            path = "keep_downgraded_external_domain_reject"
            rejection_category = "external-domain AI systematic review"
        elif review_intent["review_context_only"] or subject_only_overrides_keep:
            decision = "REJECT"
            path = (
                "keep_downgraded_paper_type_only_reject"
                if review_intent["review_context_only"]
                else "keep_downgraded_subject_only_maybe"
            )
            if path == "keep_downgraded_subject_only_maybe":
                decision = "MAYBE"
                uncertainty = keep_suppression_reason
            else:
                rejection_category = "systematic review paper type only"
        else:
            decision = "MAYBE"
            path = "keep_downgraded_missing_workflow_use_maybe"
            uncertainty = keep_suppression_reason
    elif decision == "KEEP" and desired_tool_relation:
        path = "relation_keep_confirmed"

    rq_contract = build_rq_contract(rq_frame.get("rq_text", ""), rq_frame)
    dimension_terms = {
        "method": join_terms(method_hits),
        "method_tool": join_terms(method_hits),
        "task": join_terms(task_hits),
        "task_or_outcome": join_terms(task_hits),
        "review_workflow_process": review_intent.get("review_workflow_task_terms", ""),
        "context": join_terms(context_hits),
        "review_context": review_intent.get("review_context_terms", ""),
        "outcome": join_terms(task_hits),
        "relation": relation.get("relation_evidence_terms", ""),
        "evidence_type": join_terms(evidence_hits),
    }
    ledger = build_evidence_ledger(
        rq_contract.get("rq_required_dimensions", required_dimensions),
        {**dimension_scores, "evidence_type": evidence_score},
        dimension_terms,
        relation,
        {
            "contradiction_score": relation.get("contradiction_score", 0.0),
            "contradiction_reason": contradiction_reason or relation.get("relation_mismatch_reason", ""),
            "external_domain_score": relation.get("external_domain_topic_score", 0.0),
            "subject_only_score": relation.get("subject_only_score", 0.0),
            "paper_type_only_score": relation.get("paper_type_only_score", 0.0),
            "background_only_score": 1.0 if relation.get("paper_observed_relation") == "background_mention_only" else 0.0,
            "explicit_exclusion_score": negative_score,
        },
    )
    policy_result = DEFAULT_DECISION_POLICY.decide(
        rq_contract,
        ledger,
        relation,
        review_intent,
        stage=1,
    )
    decision = policy_result["decision"]
    path = policy_result["decision_path"]
    rejection_category = policy_result["rejection_reason_category"]
    uncertainty = ledger.get("uncertainty_reason", "")
    false_keep_risk = policy_result["false_keep_risk_score"]
    keep_suppression_applied = policy_result["keep_suppression_applied"]
    keep_suppression_reason = policy_result["keep_suppression_reason"]
    keep_required_relation_missing = policy_result["keep_required_relation_missing"]
    external_domain_subject_only = policy_result["external_domain_subject_only"]
    workflow_use_missing = policy_result["workflow_use_evidence_missing"]
    subject_only_overrides_keep = policy_result["subject_only_overrides_keep"]
    relation_keep_gate_passed = policy_result["relation_keep_gate_passed"]
    paper_contract = build_paper_contract("", "", paper_frame, relation)

    diagnostics = {
        **MODEL_DIAGNOSTIC_FIELDS,
        "paper_methods": paper_frame.get("methods_or_technologies", "") or paper_analysis.get("methods_or_technologies", ""),
        "paper_tasks_or_outcomes": paper_frame.get("target_tasks_or_outcomes", "") or paper_analysis.get("target_tasks_or_outcomes", ""),
        "paper_contexts": paper_frame.get("application_contexts", "") or paper_analysis.get("application_contexts", ""),
        "paper_evidence_type": paper_frame.get("evidence_type", "") or paper_analysis.get("evidence_type", ""),
        "paper_inclusion_cues": paper_frame.get("inclusion_cues", "") or paper_analysis.get("inclusion_cues", ""),
        "paper_exclusion_cues": join_terms(split_terms(paper_frame.get("exclusion_cues", "")) + negative_hits),
        "method_evidence_terms": join_terms(method_hits),
        "task_evidence_terms": join_terms(task_hits),
        "context_evidence_terms": join_terms(context_hits),
        "evidence_type_terms": join_terms(evidence_hits),
        "exclusion_evidence_terms": join_terms(negative_hits),
        "method_evidence_strength": round(method_score, 4),
        "task_evidence_strength": round(task_score, 4),
        "context_evidence_strength": round(context_score, 4),
        "evidence_coverage_count": coverage,
        "required_dimensions_met": join_terms(met),
        "required_dimensions_missing": join_terms(missing),
        "strongest_inclusion_evidence": strongest_inclusion,
        "strongest_exclusion_evidence": strongest_exclusion,
        "contradiction_detected": contradiction_detected,
        "contradiction_reason": contradiction_reason,
        "method_alignment_score": round(method_score, 4),
        "task_alignment_score": round(task_score, 4),
        "context_alignment_score": round(context_score, 4),
        "evidence_alignment_score": round(evidence_score, 4),
        "negative_signal_score": round(negative_score, 4),
        "final_relevance_score": round(final_score, 4),
        "inclusion_evidence": "; ".join(inclusion_evidence),
        "exclusion_evidence": "; ".join(exclusion_evidence),
        "uncertainty_reason": uncertainty,
        "rejection_reason_category": rejection_category,
        "false_keep_risk_score": round(false_keep_risk, 4),
        "keep_suppression_applied": keep_suppression_applied,
        "keep_suppression_reason": keep_suppression_reason,
        "keep_required_relation_missing": keep_required_relation_missing,
        "external_domain_subject_only": external_domain_subject_only,
        "workflow_use_required_for_keep": desired_tool_relation,
        "workflow_use_evidence_missing": workflow_use_missing,
        "subject_only_overrides_keep": subject_only_overrides_keep,
        "relation_keep_gate_passed": relation_keep_gate_passed,
        "suspicious_reject": decision == "REJECT" and coverage == len(required_dimensions),
        "suspicious_keep": policy_result["suspicious_keep"],
        "workflow_intent_rescue_applied": policy_result["workflow_intent_rescue_applied"],
        "workflow_intent_rescue_reason": policy_result["workflow_intent_rescue_reason"],
        "weak_subject_review_specificity": policy_result.get(
            "weak_subject_review_specificity",
            False,
        ),
        "evidence_coverage_ratio": ledger["evidence_coverage_ratio"],
        "outcome_evidence_strength": ledger["outcome_evidence_strength"],
        "outcome_evidence_terms": ledger["outcome_evidence_terms"],
        "uncertainty_score": ledger["uncertainty_score"],
        "abstract_insufficient": ledger["abstract_insufficient"],
        "relation_unclear": ledger["relation_unclear"],
        "weak_context": ledger["weak_context"],
        "weak_method": ledger["weak_method"],
        "weak_task": ledger["weak_task"],
        "paper_contract_id": paper_contract["paper_id"],
        "paper_method_families": join_terms(paper_contract["paper_method_families"]),
        "paper_specific_models": join_terms(paper_contract["paper_specific_models"]),
        "paper_task_families": join_terms(paper_contract["paper_task_families"]),
        "paper_context_families": join_terms(paper_contract["paper_context_families"]),
        "paper_outcomes": join_terms(paper_contract["paper_outcomes"]),
        "paper_text_quality_score": paper_contract["paper_text_quality_score"],
        **review_intent,
        **relation,
        "review_role_gate_applied": review_gate_violation,
        "review_role_gate_reason": review_gate_reason,
    }
    diagnostics["relation_evidence_strength"] = relation["relation_alignment_score"]
    diagnostics["relation_dimension_met"] = relation["relation_alignment_score"] >= 0.60
    diagnostics["relation_dimension_missing"] = relation["relation_alignment_score"] < 0.60
    diagnostics["suspicious_reject"] = bool(
        decision == "REJECT"
        and (
            coverage == len(required_dimensions)
            or relation["relation_match"]
        )
    )

    if not rich_rq_frame:
        return None, diagnostics

    if decision is None:
        return None, diagnostics

    reason = _reason(decision, diagnostics, rejection_category, uncertainty)
    result = dict(base_result)
    result.update(diagnostics)
    result.update(
        {
            "decision": decision,
            "decision_path": path,
            "reason": reason,
            "comparison_diagnostic": (
                f"Evidence policy path: {path}. "
                f"method={method_score:.3f}; task={task_score:.3f}; "
                f"context={context_score:.3f}; evidence={evidence_score:.3f}; "
                f"negative={negative_score:.3f}; dimensions={coverage}/{len(required_dimensions)}."
            ),
        }
    )
    fused_result, model_diagnostics = apply_model_score_fusion(
        rq_frame=rq_frame,
        paper_frame=paper_frame,
        deterministic_result=result,
        research_question=rq_frame.get("rq_text", ""),
    )
    diagnostics.update(model_diagnostics)
    result.update(model_diagnostics)
    result.update({
        key: fused_result[key]
        for key in ("decision", "decision_path", "reason")
        if key in fused_result
    })
    return result, diagnostics


def _reason(decision, diagnostics, rejection_category, uncertainty):
    if decision == "KEEP":
        if diagnostics.get("ai_tool_for_review_workflow"):
            return (
                "Kept because AI/tool evidence is explicitly connected to "
                f"{diagnostics.get('review_workflow_task_terms') or 'a review workflow task'}."
            )
        return "Kept because structured evidence aligns on method, target outcome, and application context."
    if decision == "MAYBE":
        return f"Marked maybe because {uncertainty or 'one required relevance signal is incomplete'}."
    return f"Rejected because of {rejection_category or 'insufficient evidence'}."
