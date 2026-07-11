import semantic_comparator
from method_ontology import compare_method_families
from semantic_comparator import _decision_details, _review_workflow_gate_applies


def test_broad_ml_query_accepts_specific_ml_algorithms():
    for method in ("XGBoost", "Random Forest", "SVM", "Logistic Regression"):
        result = compare_method_families(
            "machine learning methods",
            method,
            "disease prediction",
        )
        assert result["method_family_compatible"] is True
        assert result["method_family_match"] == "machine_learning"


def test_broad_deep_learning_query_accepts_transformer_cnn_and_lstm():
    for method in ("Swin Transformer", "CNN", "LSTM"):
        result = compare_method_families(
            "deep learning methods",
            method,
            "image classification",
        )
        assert result["method_family_compatible"] is True
        assert result["broad_method_query_detected"] is True


def test_xai_and_optimization_bridge_only_with_compatible_task():
    xai = compare_method_families(
        "machine learning",
        "Kernel SHAP explainability",
        "risk prediction",
    )
    optimization = compare_method_families(
        "deep learning",
        "particle swarm optimization",
        "classification model optimization",
    )
    unrelated = compare_method_families(
        "machine learning",
        "Kernel SHAP explainability",
        "document summarization",
    )

    assert xai["method_family_compatible"] is True
    assert optimization["method_family_compatible"] is True
    assert unrelated["method_family_compatible"] is False


def test_method_family_cannot_rescue_unrelated_task():
    decision, path = _decision_details(
        task_match=0.20,
        task_identity_match=False,
        task_identity_conflict=True,
        task_family_compatible=False,
        task_family_score=0.70,
        method_family_compatible=True,
        study_role_compatible=True,
        review_role_gate=True,
        evidence_compatible=True,
        technology_match=0.90,
        context_match=0.30,
        subject_match=0.25,
    )

    assert (decision, path) == ("REJECT", "canonical_conflict_reject")


def test_method_rescue_keeps_compatible_task_and_context():
    decision, path = _decision_details(
        task_match=0.50,
        task_identity_match=False,
        task_identity_conflict=True,
        task_family_compatible=True,
        task_family_score=0.52,
        method_family_compatible=True,
        study_role_compatible=True,
        review_role_gate=True,
        evidence_compatible=True,
        technology_match=0.90,
        context_match=0.55,
        subject_match=0.05,
    )

    assert (decision, path) == ("KEEP", "task_family_method_rescue_keep")


def test_review_gate_only_applies_to_review_workflow_questions():
    assert _review_workflow_gate_applies(
        {"question_type": "review_workflow_automation"}
    )
    assert not _review_workflow_gate_applies(
        {
            "question_type": "domain_literature_review",
            "review_role": "technology_being_reviewed",
        }
    )


def _family_rescue_decision(score, subject, context, **overrides):
    arguments = {
        "task_match": 0.50,
        "task_identity_match": False,
        "task_identity_conflict": True,
        "task_family_compatible": True,
        "task_family_score": score,
        "method_family_compatible": True,
        "study_role_compatible": True,
        "review_role_gate": True,
        "evidence_compatible": True,
        "technology_match": 0.90,
        "context_match": context,
        "subject_match": subject,
    }
    arguments.update(overrides)
    return _decision_details(**arguments)


def test_semantic_rescue_has_keep_maybe_and_clear_mismatch_paths():
    assert _family_rescue_decision(0.50, 0.20, 0.35) == (
        "KEEP",
        "semantic_rescue_strong_keep",
    )
    assert _family_rescue_decision(0.44, 0.15, 0.25) == (
        "MAYBE",
        "semantic_rescue_floor_maybe",
    )
    assert _family_rescue_decision(0.38, 0.05, 0.12) == (
        "MAYBE",
        "semantic_rescue_floor_maybe",
    )
    assert _family_rescue_decision(0.36, 0.04, 0.09) == (
        "REJECT",
        "family_compatibility_domain_mismatch_reject",
    )
    assert _family_rescue_decision(
        0.50,
        0.30,
        0.35,
        evidence_compatible=False,
    ) == ("REJECT", "family_compatibility_role_mismatch_reject")


def test_semantic_rescue_diagnostics_are_returned(monkeypatch):
    similarities = {
        (0, 1): 0.10,
        (2, 3): 0.35,
        (4, 5): 0.30,
        (6, 7): 0.05,
        (8, 9): 0.25,
        (10, 11): 0.40,
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
            "primary_subject": "target domain",
            "intervention_or_method": "machine learning",
            "target_problem_or_task": "prediction",
            "application_context": "target application",
            "evidence_type": "methods",
            "question_type": "domain_literature_review",
        },
        {
            "primary_subject": "specialized target",
            "intervention_or_method": "XGBoost",
            "target_problem_or_task": "classification",
            "application_context": "specialized application",
            "evidence_type": "empirical study",
        },
    )

    assert result["decision"] == "MAYBE"
    assert result["decision_path"] == "semantic_rescue_floor_maybe"
    assert result["semantic_rescue_applied"] is True
    assert result["reject_blocked_by_family_compatibility"] is True
    assert result["rejected_despite_task_family_compatibility"] is False
    assert "Semantic rescue blocked a hard reject" in result["comparison_diagnostic"]
