import semantic_comparator
from bulk_screen import _stage2_keep_promotion_allowed
from semantic_comparator import (
    _decision_details,
    _hierarchical_decision,
    _review_workflow_gate_applies,
    _task_match_score,
)
from task_ontology import canonicalize_task, compatible_task_families


def test_prediction_and_detection_share_domain_neutral_task_family():
    families = compatible_task_families(
        "prediction",
        "detection",
        "clinical cardiovascular disease assessment",
    )

    assert families == {"predictive_assessment"}
    score, identity_match, identity_conflict = _task_match_score(
        "prediction",
        "detection",
        0.42,
        task_family_compatible=True,
    )
    assert score == 0.50
    assert identity_match is False
    assert identity_conflict is True


def test_task_family_is_domain_neutral_but_keeps_unrelated_tasks_separate():
    assert compatible_task_families(
        "prediction",
        "classification",
        "financial market analysis",
    ) == {"predictive_assessment"}
    assert not compatible_task_families(
        "prediction",
        "summarization",
        "clinical healthcare",
    )


def test_strong_clinical_family_match_is_keep_and_borderline_is_maybe():
    common = {
        "task_match": 0.50,
        "task_identity_match": False,
        "task_identity_conflict": True,
        "task_family_compatible": True,
        "task_family_score": 0.70,
        "study_role_compatible": True,
        "review_role_gate": True,
        "evidence_compatible": True,
    }

    assert _hierarchical_decision(
        **common,
        technology_match=0.80,
        context_match=0.70,
        subject_match=0.75,
    ) == "KEEP"
    assert _hierarchical_decision(
        **{**common, "task_family_score": 0.49},
        technology_match=0.50,
        context_match=0.48,
        subject_match=0.50,
    ) == "MAYBE"


def test_weak_family_match_and_unrelated_task_are_rejected():
    family_decision, family_path = _decision_details(
        task_match=0.50,
        task_identity_match=False,
        task_identity_conflict=True,
        task_family_compatible=True,
        task_family_score=0.38,
        study_role_compatible=True,
        review_role_gate=True,
        evidence_compatible=True,
        technology_match=0.30,
        context_match=0.35,
        subject_match=0.40,
    )
    unrelated_decision, unrelated_path = _decision_details(
        task_match=0.20,
        task_identity_match=False,
        task_identity_conflict=True,
        task_family_compatible=False,
        task_family_score=0.0,
        study_role_compatible=True,
        review_role_gate=True,
        evidence_compatible=True,
        technology_match=0.20,
        context_match=0.30,
        subject_match=0.30,
    )

    assert (family_decision, family_path) == ("REJECT", "task_family_weak_reject")
    assert (unrelated_decision, unrelated_path) == ("REJECT", "canonical_conflict_reject")


def test_exact_task_match_and_slr_role_gate_have_explicit_paths():
    exact_decision, exact_path = _decision_details(
        task_match=1.0,
        task_identity_match=True,
        task_identity_conflict=False,
        task_family_compatible=False,
        task_family_score=0.0,
        study_role_compatible=True,
        review_role_gate=True,
        evidence_compatible=True,
        technology_match=0.80,
        context_match=0.70,
        subject_match=0.75,
    )
    gated_decision, gated_path = _decision_details(
        task_match=1.0,
        task_identity_match=True,
        task_identity_conflict=False,
        task_family_compatible=False,
        task_family_score=0.0,
        study_role_compatible=True,
        review_role_gate=False,
        evidence_compatible=True,
        technology_match=0.90,
        context_match=0.90,
        subject_match=0.90,
    )

    assert (exact_decision, exact_path) == ("KEEP", "exact_task_strong_keep")
    assert (gated_decision, gated_path) == ("REJECT", "review_workflow_gate_reject")


def test_exact_task_identity_is_not_downgraded_by_conservative_cosines():
    decision, path = _decision_details(
        task_match=1.0,
        task_identity_match=True,
        task_identity_conflict=False,
        task_family_compatible=False,
        task_family_score=0.0,
        study_role_compatible=True,
        review_role_gate=True,
        evidence_compatible=True,
        technology_match=0.38,
        context_match=0.40,
        subject_match=0.42,
    )

    assert (decision, path) == ("KEEP", "exact_task_strong_keep")


def test_risk_tasks_canonicalize_and_slr_review_gate_remains_active():
    assert canonicalize_task("risk estimation and risk stratification") == "prediction"
    assert _review_workflow_gate_applies(
        {"question_type": "review_workflow_automation"}
    )
    assert not _review_workflow_gate_applies(
        {"question_type": "domain_literature_review"}
    )


def test_clinical_risk_screening_is_predictive_assessment_compatible():
    families = compatible_task_families(
        "prediction and diagnosis",
        "cardiovascular disease risk screening using routinely collected health data",
        "clinical medical healthcare patient risk assessment",
    )

    assert families == {"predictive_assessment"}


def test_review_workflow_screening_stays_separate_from_disease_screening():
    assert not compatible_task_families(
        "prediction and diagnosis",
        "title abstract screening for systematic reviews",
        "systematic literature review study selection paper screening",
    )
    assert not compatible_task_families(
        "study screening",
        "disease screening",
        "systematic literature review workflow",
    )


def test_generic_or_security_screening_does_not_join_predictive_assessment():
    assert not compatible_task_families(
        "prediction",
        "generic screening",
        "general document triage",
    )
    assert not compatible_task_families(
        "prediction",
        "security screening",
        "cybersecurity threat access control",
    )


def test_comparator_exposes_strong_family_decision_diagnostics(monkeypatch):
    similarities = {
        (0, 1): 0.80,
        (2, 3): 0.45,
        (4, 5): 0.60,
        (6, 7): 0.75,
        (8, 9): 0.70,
        (10, 11): 0.60,
    }
    monkeypatch.setattr(
        semantic_comparator,
        "_encode_by_index",
        lambda texts: {index: text for index, text in enumerate(texts) if text},
    )
    monkeypatch.setattr(
        semantic_comparator,
        "_pair_similarity",
        lambda embeddings, left, right: similarities.get((left, right), 0.0),
    )

    result = semantic_comparator.compare_semantic_frames(
        {
            "primary_subject": "disease outcome assessment",
            "intervention_or_method": "machine learning",
            "target_problem_or_task": "prediction and diagnosis",
            "application_context": "clinical decision support",
            "evidence_type": "methods",
            "question_type": "domain_literature_review",
        },
        {
            "primary_subject": "disease outcome assessment",
            "intervention_or_method": "deep learning",
            "target_problem_or_task": "classification",
            "application_context": "clinical decision support",
            "evidence_type": "empirical study",
        },
    )

    assert result["decision"] == "KEEP"
    assert result["decision_path"] == "task_family_method_rescue_keep"
    assert result["task_family_compatible"] is True
    assert result["task_family_match"] == "predictive_assessment"
    assert result["task_family_score"] >= 0.60
    assert result["rejected_despite_task_family_compatibility"] is False
    assert "broader method family" in result["reason"]


def test_stage2_keep_promotion_uses_new_semantic_evidence():
    allowed, reason = _stage2_keep_promotion_allowed(
        {"question_type": "domain_literature_review"},
        {
            "decision": "KEEP",
            "confidence": 0.40,
            "task_family_compatible": True,
            "method_family_compatible": True,
            "task_family_score": 0.44,
            "subject_match": 0.02,
            "context_match": 0.26,
            "decision_path": "semantic_rescue_floor_maybe",
        },
    )

    assert allowed is True
    assert "compatible method and task evidence" in reason


def test_stage2_keep_promotion_blocks_review_role_gate_violation():
    allowed, reason = _stage2_keep_promotion_allowed(
        {"question_type": "review_workflow_automation"},
        {
            "decision": "KEEP",
            "confidence": 0.99,
            "task_family_compatible": True,
            "method_family_compatible": True,
            "task_family_score": 0.90,
            "subject_match": 0.90,
            "context_match": 0.90,
            "paper_review_role": "technology_being_reviewed",
        },
    )

    assert allowed is False
    assert "review-workflow task" in reason


def test_stage2_keep_promotion_blocks_method_or_domain_contradiction():
    method_allowed, method_reason = _stage2_keep_promotion_allowed(
        {"question_type": "domain_literature_review"},
        {
            "decision": "KEEP",
            "confidence": 0.99,
            "task_family_compatible": True,
            "method_family_compatible": False,
            "effective_technology_match": 0.20,
            "task_family_score": 0.90,
            "subject_match": 0.80,
            "context_match": 0.80,
        },
    )
    domain_allowed, domain_reason = _stage2_keep_promotion_allowed(
        {"question_type": "domain_literature_review"},
        {
            "decision": "KEEP",
            "confidence": 0.99,
            "task_family_compatible": True,
            "method_family_compatible": True,
            "task_family_score": 0.90,
            "subject_match": 0.02,
            "context_match": 0.04,
        },
    )

    assert method_allowed is False
    assert "method family is unrelated" in method_reason
    assert domain_allowed is False
    assert "domain mismatch" in domain_reason
