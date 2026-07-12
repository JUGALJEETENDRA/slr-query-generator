import json
import traceback

from ollama_client import ask_ollama
from semantic_frame import extract_research_question_frame, extract_semantic_frame
from semantic_comparator import compare_semantic_frames
from model_score_fusion import apply_model_score_fusion
from runtime_config import get_model_judge_config


RQ_DIAGNOSTIC_FIELDS = (
    "review_question_type",
    "core_domain",
    "method_or_technology",
    "method_family",
    "target_tasks_or_outcomes",
    "required_inclusion_concepts",
    "optional_related_concepts",
    "exclusion_concepts",
    "expected_evidence_types",
    "domain_synonyms",
    "method_synonyms",
    "task_outcome_synonyms",
    "context_synonyms",
    "negative_contexts",
    "required_dimensions",
    "minimum_inclusion_rule",
    "rq_extraction_suspect",
    "rq_desired_relation",
    "rq_id",
    "rq_text",
    "rq_type",
    "rq_scope_width",
    "rq_strictness",
    "rq_required_dimensions",
    "rq_optional_dimensions",
    "rq_method_terms",
    "rq_method_families",
    "rq_task_terms",
    "rq_task_families",
    "rq_context_terms",
    "rq_context_families",
    "rq_outcome_terms",
    "rq_evidence_types_expected",
    "rq_inclusion_concepts",
    "rq_exclusion_concepts",
    "rq_positive_relation_patterns",
    "rq_negative_relation_patterns",
    "rq_ambiguity_policy",
    "rq_stage2_policy",
    "corpus_profile_terms",
    "corpus_method_terms",
    "corpus_task_terms",
    "corpus_context_terms",
    "corpus_evidence_terms",
    "corpus_domain_specific_synonyms",
    "corpus_review_context_terms",
    "corpus_workflow_task_terms",
    "corpus_ai_tool_terms",
    "corpus_external_domain_terms",
    "corpus_technology_subject_terms",
    "corpus_automation_intent_terms",
    "corpus_review_workflow_terms",
    "corpus_tool_use_terms",
    "corpus_subject_review_terms",
    "corpus_relation_clusters",
)

PAPER_DIAGNOSTIC_FIELDS = (
    "paper_methods",
    "paper_tasks_or_outcomes",
    "paper_contexts",
    "paper_evidence_type",
    "paper_inclusion_cues",
    "paper_exclusion_cues",
)

ALIGNMENT_DIAGNOSTIC_FIELDS = (
    "paper_observed_relation",
    "relation_match",
    "relation_conflict",
    "relation_alignment_score",
    "relation_confidence",
    "relation_evidence_terms",
    "relation_negative_terms",
    "relation_mismatch_reason",
    "workflow_use_score",
    "subject_only_score",
    "paper_type_only_score",
    "external_domain_topic_score",
    "implied_workflow_score",
    "uncertainty_preservation_score",
    "contradiction_score",
    "relation_decision_path",
    "review_automation_relevance_score",
    "relation_evidence_strength",
    "relation_dimension_met",
    "relation_dimension_missing",
    "false_keep_risk_score",
    "keep_suppression_applied",
    "keep_suppression_reason",
    "keep_required_relation_missing",
    "external_domain_subject_only",
    "workflow_use_required_for_keep",
    "workflow_use_evidence_missing",
    "subject_only_overrides_keep",
    "relation_keep_gate_passed",
    "outcome_evidence_strength",
    "outcome_evidence_terms",
    "evidence_coverage_ratio",
    "uncertainty_score",
    "abstract_insufficient",
    "relation_unclear",
    "weak_context",
    "weak_method",
    "weak_task",
    "suspicious_keep",
    "paper_contract_id",
    "paper_method_families",
    "paper_specific_models",
    "paper_task_families",
    "paper_context_families",
    "paper_outcomes",
    "paper_text_quality_score",
    "review_intent_relation",
    "review_relation_confidence",
    "review_workflow_task_detected",
    "review_workflow_task_terms",
    "review_workflow_task_families",
    "review_automation_intent_detected",
    "review_automation_intent_terms",
    "review_context_detected",
    "review_context_terms",
    "ai_tool_terms",
    "external_domain_terms",
    "ai_tool_for_review_workflow",
    "technology_subject_review_detected",
    "external_domain_review_detected",
    "review_context_only",
    "strong_workflow_use_evidence",
    "workflow_task_term_detected",
    "workflow_task_object_detected",
    "workflow_task_object_linked",
    "workflow_task_object_terms",
    "workflow_task_sense",
    "workflow_task_sense_confidence",
    "external_domain_task_detected",
    "external_domain_task_terms",
    "task_object_mismatch",
    "task_object_mismatch_reason",
    "implied_workflow_intent_score",
    "implied_workflow_intent_terms",
    "strong_implied_workflow_intent",
    "medium_implied_workflow_intent",
    "workflow_direction_detected",
    "subject_review_direction_detected",
    "relation_direction",
    "workflow_intent_rescue_applied",
    "workflow_intent_rescue_reason",
    "workflow_task_object_required",
    "workflow_task_object_missing",
    "review_role_gate_reason",
    "method_evidence_terms",
    "task_evidence_terms",
    "context_evidence_terms",
    "evidence_type_terms",
    "exclusion_evidence_terms",
    "method_evidence_strength",
    "task_evidence_strength",
    "context_evidence_strength",
    "evidence_coverage_count",
    "required_dimensions_met",
    "required_dimensions_missing",
    "strongest_inclusion_evidence",
    "strongest_exclusion_evidence",
    "contradiction_detected",
    "contradiction_reason",
    "suspicious_reject",
    "method_alignment_score",
    "task_alignment_score",
    "context_alignment_score",
    "evidence_alignment_score",
    "negative_signal_score",
    "final_relevance_score",
    "inclusion_evidence",
    "exclusion_evidence",
    "uncertainty_reason",
    "rejection_reason_category",
)


def _clean_reason(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _fallback_reason(decision, paper_frame, comparison_result):
    task = paper_frame.get("target_problem_or_task", "") or "the paper's stated task"
    method = paper_frame.get("intervention_or_method", "") or "the reported method"
    role = paper_frame.get("review_role", "") or paper_frame.get("study_role", "")
    task_score = float(comparison_result.get("task_role_match", 0.0) or 0.0)

    if decision == "KEEP":
        return (
            f"Included because the abstract indicates that {method} is used for "
            f"{task}, which aligns with the review question. "
            f"The semantic task-role match is {task_score:.2f}."
        )
    if decision == "MAYBE":
        return (
            f"Marked maybe because the abstract has partial relevance to {task}, "
            f"but the match to the review question is not strong enough for automatic inclusion. "
            f"The semantic task-role match is {task_score:.2f}."
        )
    if role == "technology_being_reviewed":
        return (
            "Excluded because the abstract appears to review the technology itself "
            "rather than evaluate it as evidence for the review question."
        )
    return (
        f"Excluded because the abstract's main task, {task}, does not sufficiently "
        "match the task required by the review question."
    )


def generate_screening_reason(
    title,
    abstract,
    research_question,
    decision,
    rq_frame,
    paper_frame,
    comparison_result,
    model="qwen2.5:3b",
    inference_engine=None,
):
    prompt = f"""
Research Question:
{research_question}

Paper Title:
{title}

Paper Abstract:
{abstract}

Decision:
{decision}

Research Question Semantic Frame:
{json.dumps(rq_frame, ensure_ascii=True)}

Paper Semantic Frame:
{json.dumps(paper_frame, ensure_ascii=True)}

Semantic Match Signals:
{json.dumps(comparison_result, ensure_ascii=True)}

Write the screening rationale for this decision.

Requirements:
- Base the rationale on the title and abstract.
- Explain why the paper is included, excluded, or marked maybe.
- Mention the strongest relevant task/evidence match or mismatch.
- Do not expose internal threshold names, rule labels, JSON keys, or raw scores unless they are essential.
- Keep it to one concise sentence.

Return ONLY JSON:

{{
  "reason": ""
}}
"""

    try:
        ask = inference_engine.ask if inference_engine is not None else ask_ollama
        response = ask(prompt, model=model)
        parsed = json.loads(response)
        reason = _clean_reason(parsed.get("reason", ""))
        if reason:
            return reason
    except Exception:
        pass

    return _fallback_reason(decision, paper_frame, comparison_result)


def screen_paper(
    title,
    abstract,
    research_question,
    rq_frame=None,
    model="qwen2.5:3b",
    mode="local",
    inference_engine=None,
):
    rq_frame_for_fallback = rq_frame if isinstance(rq_frame, dict) else {}
    paper_frame_for_fallback = {
        "source_title": title,
        "source_abstract": abstract,
        "primary_subject": title,
    }
    try:
        if rq_frame is None:
            rq_frame = extract_research_question_frame(
                research_question=research_question,
                model=model,
                inference_engine=inference_engine,
            )
        rq_frame_for_fallback = rq_frame

        paper_frame = extract_semantic_frame(
            title=title,
            abstract=abstract,
            model=model,
            inference_engine=inference_engine,
        )
        paper_frame_for_fallback = paper_frame

        comparison_result = compare_semantic_frames(
            rq_frame,
            paper_frame,
        )
        decision = comparison_result.get("decision", "ERROR")
        reason = _fallback_reason(decision, paper_frame, comparison_result)

        result = {
            "decision": decision,
            "reason": reason,
            "required_evidence": "",
            "paper_contribution": "",
            "paper_primary_subject": paper_frame.get("primary_subject", ""),
            "paper_intervention_or_method": paper_frame.get("intervention_or_method", ""),
            "paper_target_problem_or_task": paper_frame.get("target_problem_or_task", ""),
            "paper_application_context": paper_frame.get("application_context", ""),
            "paper_evidence_type": paper_frame.get("evidence_type", ""),
            "paper_study_role": paper_frame.get("study_role", ""),
            "paper_review_role": paper_frame.get("review_role", ""),
            "cache_schema_complete": paper_frame.get("cache_schema_complete", True),
            "cache_missing_adjudication_fields": paper_frame.get("cache_missing_adjudication_fields", ""),
            "fast_recomputed_due_to_incomplete_cache": paper_frame.get("fast_recomputed_due_to_incomplete_cache", False),
            "rq_primary_subject": rq_frame.get("primary_subject", ""),
            "rq_intervention_or_method": rq_frame.get("intervention_or_method", ""),
            "rq_target_problem_or_task": rq_frame.get("target_problem_or_task", ""),
            "rq_application_context": rq_frame.get("application_context", ""),
            "rq_evidence_type": rq_frame.get("evidence_type", ""),
            "rq_study_role": rq_frame.get("study_role", ""),
            "rq_review_role": rq_frame.get("review_role", ""),
            "rq_question_type": rq_frame.get("question_type", ""),
            "rq_frame_source": rq_frame.get("frame_source", ""),
            "rq_frame_diagnostic": rq_frame.get("frame_diagnostic", ""),
            "technology_match": comparison_result.get("technology_match", 0.0),
            "effective_technology_match": comparison_result.get("effective_technology_match", 0.0),
            "method_family_left": comparison_result.get("method_family_left", ""),
            "method_family_right": comparison_result.get("method_family_right", ""),
            "method_family_compatible": comparison_result.get("method_family_compatible", False),
            "method_family_match": comparison_result.get("method_family_match", ""),
            "method_family_reason": comparison_result.get("method_family_reason", ""),
            "method_family_confidence": comparison_result.get("method_family_confidence", 0.0),
            "broad_method_query_detected": comparison_result.get("broad_method_query_detected", False),
            "task_match": comparison_result.get("task_match", 0.0),
            "task_subject_match": comparison_result.get("task_subject_match", 0.0),
            "task_role_match": comparison_result.get("task_role_match", 0.0),
            "subject_match": comparison_result.get("subject_match", 0.0),
            "context_match": comparison_result.get("context_match", 0.0),
            "study_role_match": comparison_result.get("study_role_match", False),
            "review_role_match": comparison_result.get("review_role_match", False),
            "canonical_task_left": comparison_result.get("canonical_task_left", ""),
            "canonical_task_right": comparison_result.get("canonical_task_right", ""),
            "task_identity_match": comparison_result.get("task_identity_match", False),
            "task_identity_conflict": comparison_result.get("task_identity_conflict", False),
            "task_family_compatible": comparison_result.get("task_family_compatible", False),
            "task_family_match": comparison_result.get("task_family_match", ""),
            "task_family_score": comparison_result.get("task_family_score", 0.0),
            "decision_path": comparison_result.get("decision_path", ""),
            "semantic_rescue_applied": comparison_result.get("semantic_rescue_applied", False),
            "semantic_rescue_reason": comparison_result.get("semantic_rescue_reason", ""),
            "reject_blocked_by_family_compatibility": comparison_result.get(
                "reject_blocked_by_family_compatibility",
                False,
            ),
            "rejected_despite_task_family_compatibility": comparison_result.get(
                "rejected_despite_task_family_compatibility",
                False,
            ),
            "review_workflow_gate_applied": comparison_result.get("review_role_gate_applied", comparison_result.get("review_workflow_gate_applied", False)),
            "comparison_diagnostic": comparison_result.get("comparison_diagnostic", ""),
        }
        for field in RQ_DIAGNOSTIC_FIELDS:
            result[f"rq_{field}"] = rq_frame.get(field, "")
        for field in PAPER_DIAGNOSTIC_FIELDS:
            result[field] = comparison_result.get(field, "")
        for field in ALIGNMENT_DIAGNOSTIC_FIELDS:
            result[field] = comparison_result.get(field, 0.0 if field.endswith("_score") else "")
        if get_model_judge_config()["enable_model_judges"] and not result.get("model_judges_enabled"):
            fused_result, model_diagnostics = apply_model_score_fusion(
                rq_frame=rq_frame,
                paper_frame=paper_frame,
                deterministic_result=result,
                research_question=research_question,
                model=model,
                inference_engine=inference_engine,
            )
            result.update(model_diagnostics)
            result.update({
                key: fused_result[key]
                for key in ("decision", "decision_path", "reason")
                if key in fused_result
            })
        return result

    except Exception as e:
        trace_short = "".join(
            traceback.format_exception(type(e), e, e.__traceback__, limit=6)
        ).strip()
        result = {
            "decision": "MAYBE",
            "reason": "Internal deterministic screening error; fallback uncertainty decision used.",
            "required_evidence": "",
            "paper_contribution": "",
            "paper_primary_subject": paper_frame_for_fallback.get("primary_subject", title),
            "paper_intervention_or_method": paper_frame_for_fallback.get("intervention_or_method", ""),
            "paper_target_problem_or_task": paper_frame_for_fallback.get("target_problem_or_task", ""),
            "paper_application_context": paper_frame_for_fallback.get("application_context", ""),
            "paper_evidence_type": paper_frame_for_fallback.get("evidence_type", ""),
            "paper_study_role": paper_frame_for_fallback.get("study_role", ""),
            "paper_review_role": paper_frame_for_fallback.get("review_role", ""),
            "rq_primary_subject": rq_frame_for_fallback.get("primary_subject", ""),
            "rq_intervention_or_method": rq_frame_for_fallback.get("intervention_or_method", ""),
            "rq_target_problem_or_task": rq_frame_for_fallback.get("target_problem_or_task", ""),
            "rq_application_context": rq_frame_for_fallback.get("application_context", ""),
            "rq_evidence_type": rq_frame_for_fallback.get("evidence_type", ""),
            "rq_study_role": rq_frame_for_fallback.get("study_role", ""),
            "rq_review_role": rq_frame_for_fallback.get("review_role", ""),
            "rq_question_type": rq_frame_for_fallback.get("question_type", ""),
            "rq_frame_source": rq_frame_for_fallback.get("frame_source", ""),
            "rq_frame_diagnostic": rq_frame_for_fallback.get("frame_diagnostic", ""),
            "technology_match": 0.0,
            "effective_technology_match": 0.0,
            "method_family_left": "",
            "method_family_right": "",
            "method_family_compatible": False,
            "method_family_match": "",
            "method_family_reason": "",
            "method_family_confidence": 0.0,
            "broad_method_query_detected": False,
            "task_match": 0.0,
            "task_subject_match": 0.0,
            "task_role_match": 0.0,
            "subject_match": 0.0,
            "context_match": 0.0,
            "study_role_match": False,
            "review_role_match": False,
            "canonical_task_left": "",
            "canonical_task_right": "",
            "task_identity_match": False,
            "task_identity_conflict": False,
            "task_family_compatible": False,
            "task_family_match": "",
            "task_family_score": 0.0,
            "decision_path": "deterministic_error_fallback_maybe",
            "semantic_rescue_applied": False,
            "semantic_rescue_reason": "",
            "reject_blocked_by_family_compatibility": False,
            "rejected_despite_task_family_compatibility": False,
            "review_workflow_gate_applied": False,
            "comparison_diagnostic": str(e),
            "stage1_error_type": type(e).__name__,
            "stage1_error_message": str(e),
            "stage1_error_trace_short": trace_short,
        }
        for field in RQ_DIAGNOSTIC_FIELDS:
            result[f"rq_{field}"] = ""
        for field in PAPER_DIAGNOSTIC_FIELDS:
            result[field] = ""
        for field in ALIGNMENT_DIAGNOSTIC_FIELDS:
            result[field] = 0.0 if field.endswith("_score") else ""
        if get_model_judge_config()["enable_model_judges"]:
            fused_result, model_diagnostics = apply_model_score_fusion(
                rq_frame=rq_frame_for_fallback,
                paper_frame=paper_frame_for_fallback,
                deterministic_result=result,
                research_question=research_question,
                model=model,
                inference_engine=inference_engine,
            )
            result.update(model_diagnostics)
            result.update({
                key: fused_result[key]
                for key in ("decision", "decision_path", "reason")
                if key in fused_result
            })
        return result
