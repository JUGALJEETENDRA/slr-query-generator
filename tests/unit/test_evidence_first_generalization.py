import semantic_comparator
from benchmark_local_screening import summarize_screening_rows
from semantic_frame import _heuristic_research_question_frame


def _disable_embeddings(monkeypatch):
    monkeypatch.setattr(
        semantic_comparator,
        "_encode_by_index",
        lambda texts: {index: text for index, text in enumerate(texts) if text},
    )
    monkeypatch.setattr(
        semantic_comparator,
        "_pair_similarity",
        lambda embeddings, left, right: 0.01,
    )


def _compare(monkeypatch, rq, method, task, context, subject="direct application"):
    _disable_embeddings(monkeypatch)
    frame = _heuristic_research_question_frame(rq)
    paper = {
        "primary_subject": subject,
        "intervention_or_method": method,
        "target_problem_or_task": task,
        "application_context": context,
        "evidence_type": "empirical evaluation",
        "review_role": "",
    }
    return frame, semantic_comparator.compare_semantic_frames(frame, paper)


def test_cybersecurity_ml_intrusion_detection_iot_is_keep(monkeypatch):
    rq = "How are machine learning methods used for intrusion detection and threat classification in IoT networks?"
    frame, result = _compare(
        monkeypatch, rq, "random forest machine learning", "intrusion detection",
        "IoT network cybersecurity",
    )
    assert frame["review_question_type"] == "security_risk_review"
    assert result["decision"] == "KEEP"


def test_education_dropout_learning_analytics_is_keep(monkeypatch):
    rq = "What learning analytics methods are used to predict student dropout and academic performance?"
    _, result = _compare(
        monkeypatch, rq, "learning analytics", "student dropout prediction",
        "university education",
    )
    assert result["decision"] == "KEEP"


def test_software_defect_prediction_deep_learning_is_keep(monkeypatch):
    rq = "How are deep learning methods used for software defect prediction and bug detection?"
    _, result = _compare(
        monkeypatch, rq, "deep learning neural network",
        "software defect prediction and bug detection", "software engineering",
    )
    assert result["decision"] == "KEEP"


def test_legal_transformer_information_extraction_is_keep(monkeypatch):
    rq = "How are transformer models used for information extraction in legal documents?"
    _, result = _compare(
        monkeypatch, rq, "BERT transformer model", "information extraction",
        "legal documents and contracts",
    )
    assert result["decision"] == "KEEP"


def test_missing_two_dimensions_rejects(monkeypatch):
    rq = "How are transformer models used for information extraction in legal documents?"
    _, result = _compare(
        monkeypatch, rq, "transformer benchmark", "language modeling", "general NLP",
    )
    assert result["decision"] == "REJECT"
    assert result["required_dimensions_missing"]


def test_two_dimensions_and_one_unclear_is_maybe(monkeypatch):
    rq = "How are deep learning methods used for software defect prediction and bug detection?"
    _, result = _compare(
        monkeypatch, rq, "deep learning", "software defect prediction", "industrial dataset",
    )
    assert result["decision"] == "MAYBE"
    assert result["evidence_coverage_count"] == 2


def test_medical_xai_prediction_is_not_false_reject(monkeypatch):
    rq = "What machine learning and deep learning methods have been proposed for heart disease prediction and diagnosis?"
    _, result = _compare(
        monkeypatch, rq, "machine learning with SHAP explainability",
        "heart disease prediction", "clinical cardiovascular healthcare",
    )
    assert result["decision"] in {"KEEP", "MAYBE"}


def test_benchmark_flags_reject_with_complete_coverage():
    summary = summarize_screening_rows([
        {
            "Decision": "REJECT",
            "Title": "false reject",
            "stage1_evidence_coverage_count": 3,
            "stage1_suspicious_reject": True,
            "stage1_corpus_method_terms": "method",
            "stage1_corpus_task_terms": "task",
            "stage1_corpus_context_terms": "context",
        }
    ])
    assert "rejects_with_all_required_dimensions_met" in summary["warnings"]
    assert summary["top_suspicious_false_rejects"] == ["false reject"]
