import semantic_comparator
import pytest
from bulk_screen import _stage2_reject_demotion_allowed
from semantic_frame import _heuristic_research_question_frame


RQ = "Can large language models and artificial intelligence tools help automate systematic literature reviews?"


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


def _screen(monkeypatch, subject, method, task, context, review_role=""):
    _disable_embeddings(monkeypatch)
    rq = _heuristic_research_question_frame(RQ)
    paper = {
        "primary_subject": subject,
        "intervention_or_method": method,
        "target_problem_or_task": task,
        "application_context": context,
        "evidence_type": "empirical evaluation",
        "review_role": review_role,
    }
    return rq, semantic_comparator.compare_semantic_frames(rq, paper)


def test_rq_analyzer_extracts_review_workflow_intent():
    rq = _heuristic_research_question_frame(RQ)
    assert rq["review_question_type"] == "review_workflow_automation"
    assert "large language model" in rq["method_or_technology"]
    assert "systematic literature review" in rq["application_context"]
    assert "title abstract screening" in rq["target_tasks_or_outcomes"]
    assert rq["rq_extraction_suspect"] == "False"
    assert rq["rq_desired_relation"] == "tool_used_for_workflow"


def test_desired_relations_for_medical_and_blockchain():
    medical = _heuristic_research_question_frame(
        "What machine learning and deep learning methods have been proposed for heart disease prediction and diagnosis?"
    )
    blockchain = _heuristic_research_question_frame(
        "How is blockchain technology used to improve transparency, traceability, trust, and security in supply chain management?"
    )
    assert medical["rq_desired_relation"] == "method_used_for_task"
    assert blockchain["rq_desired_relation"] == "technology_used_for_outcome"


def test_llm_title_abstract_screening_is_keep_despite_model_role(monkeypatch):
    _, result = _screen(
        monkeypatch,
        "Cal-X enhancing systematic review screening with LLMs",
        "large language model",
        "title and abstract screening and study selection",
        "systematic reviews",
        "technology_being_reviewed",
    )
    assert result["decision"] == "KEEP"
    assert result["ai_tool_for_review_workflow"] is True
    assert result["review_role_gate_applied"] is False
    assert result["relation_match"] is True
    assert result["paper_observed_relation"] == "tool_used_for_workflow"
    assert result["relation_keep_gate_passed"] is True
    assert result["keep_suppression_applied"] is False


def test_active_learning_review_screening_is_keep_or_maybe(monkeypatch):
    _, result = _screen(
        monkeypatch,
        "active learning for citation screening",
        "machine learning active learning classifier",
        "citation screening",
        "systematic literature review",
    )
    assert result["decision"] in {"KEEP", "MAYBE"}


def test_llm_data_extraction_is_keep(monkeypatch):
    _, result = _screen(
        monkeypatch,
        "LLM extraction assistant for evidence reviews",
        "GPT-4 large language model",
        "data extraction and PICO extraction",
        "systematic review workflow",
    )
    assert result["decision"] == "KEEP"


def test_llm_evidence_synthesis_is_keep_or_maybe(monkeypatch):
    _, result = _screen(
        monkeypatch,
        "AI review assistant",
        "artificial intelligence tool",
        "evidence synthesis and review assistance",
        "systematic reviews",
    )
    assert result["decision"] in {"KEEP", "MAYBE"}


def test_systematic_review_of_ai_in_cardiology_is_not_keep(monkeypatch):
    _, result = _screen(
        monkeypatch,
        "systematic review of artificial intelligence in cardiology",
        "artificial intelligence and machine learning",
        "heart disease diagnosis",
        "systematic review in cardiology",
        "technology_being_reviewed",
    )
    assert result["decision"] != "KEEP"
    assert result["technology_subject_review_detected"] is True


def test_general_llm_survey_is_not_keep(monkeypatch):
    _, result = _screen(
        monkeypatch,
        "general survey of large language models",
        "large language models",
        "language generation",
        "natural language processing",
        "technology_being_reviewed",
    )
    assert result["decision"] != "KEEP"


def test_review_paper_type_alone_is_insufficient(monkeypatch):
    _, result = _screen(
        monkeypatch,
        "systematic review of nutrition interventions",
        "",
        "dietary outcomes",
        "systematic review in nutrition",
    )
    assert result["decision"] == "REJECT"
    assert result["review_intent_relation"] == "review_context_only"


def test_disease_screening_not_confused_with_review_screening(monkeypatch):
    _, result = _screen(
        monkeypatch,
        "systematic review of AI for disease screening",
        "machine learning",
        "clinical disease screening",
        "systematic review in healthcare",
        "technology_being_reviewed",
    )
    assert result["decision"] != "KEEP"
    assert result["review_workflow_task_detected"] is False


def test_medical_and_blockchain_regressions_stay_keep(monkeypatch):
    _disable_embeddings(monkeypatch)
    medical = _heuristic_research_question_frame(
        "What machine learning and deep learning methods have been proposed for heart disease prediction and diagnosis?"
    )
    blockchain = _heuristic_research_question_frame(
        "How is blockchain technology used to improve transparency, traceability, trust, and security in supply chain management?"
    )
    medical_result = semantic_comparator.compare_semantic_frames(medical, {
        "primary_subject": "heart disease prediction",
        "intervention_or_method": "deep learning neural network",
        "target_problem_or_task": "prediction and diagnosis",
        "application_context": "cardiovascular healthcare",
        "review_role": "",
    })
    blockchain_result = semantic_comparator.compare_semantic_frames(blockchain, {
        "primary_subject": "blockchain supply chain traceability",
        "intervention_or_method": "blockchain smart contracts",
        "target_problem_or_task": "traceability and transparency",
        "application_context": "supply chain management",
        "review_role": "",
    })
    assert medical_result["decision"] == "KEEP"
    assert blockchain_result["decision"] == "KEEP"
    assert medical_result["keep_suppression_applied"] is False
    assert blockchain_result["keep_suppression_applied"] is False


def test_automated_review_pipeline_using_nlp_is_keep(monkeypatch):
    _, result = _screen(
        monkeypatch,
        "automated systematic review pipeline",
        "natural language processing automation tool",
        "review pipeline supporting reviewers",
        "systematic literature reviews",
    )
    assert result["decision"] == "KEEP"
    assert result["review_intent_relation"] == "review_workflow_tool"


def test_external_domain_xai_fintech_review_is_not_keep(monkeypatch):
    _, result = _screen(
        monkeypatch,
        "application of explainable artificial intelligence in fintech: a systematic review",
        "artificial intelligence",
        "financial applications",
        "systematic review in fintech",
        "technology_being_reviewed",
    )
    assert result["decision"] != "KEEP"
    assert result["external_domain_review_detected"] is True
    assert result["keep_suppression_applied"] is True
    assert result["external_domain_subject_only"] is True


def test_telemedicine_ai_systematic_review_is_not_keep(monkeypatch):
    _, result = _screen(
        monkeypatch,
        "impact of generative AI and large language models on telemedicine diagnostics",
        "generative artificial intelligence and large language models",
        "telemedicine diagnosis",
        "systematic review in healthcare",
    )
    assert result["decision"] != "KEEP"
    assert result["external_domain_subject_only"] is True


@pytest.mark.parametrize(
    ("subject", "task", "context"),
    [
        (
            "systematic literature review on AI approaches for security operations centers",
            "classification and screening",
            "systematic review in cybersecurity",
        ),
        (
            "applications of large language models in breast cancer diagnosis",
            "diagnosis and information extraction",
            "systematic review in healthcare",
        ),
        (
            "impact of generative AI in K-12 education",
            "academic performance classification",
            "systematic literature review in education",
        ),
        (
            "explainable artificial intelligence in financial risk management",
            "risk explanation",
            "systematic review in finance",
        ),
        (
            "large language models in regional dialect analysis",
            "language classification",
            "systematic literature review",
        ),
    ],
)
def test_external_ai_subject_patterns_cannot_be_workflow_keep(
    monkeypatch,
    subject,
    task,
    context,
):
    _, result = _screen(
        monkeypatch,
        subject,
        "artificial intelligence and large language models",
        task,
        context,
        "technology_being_reviewed",
    )
    assert result["decision"] != "KEEP"
    assert result["paper_observed_relation"] in {
        "external_domain_review",
        "technology_reviewed_as_subject",
    }
    assert result["workflow_use_score"] == 0.0
    assert result["relation_keep_gate_passed"] is False


def test_explicit_workflow_language_overrides_subject_review_syntax(monkeypatch):
    _, result = _screen(
        monkeypatch,
        "LLM for title and abstract screening in systematic reviews",
        "large language model",
        "title and abstract screening and study selection",
        "systematic reviews",
    )
    assert result["decision"] == "KEEP"
    assert result["paper_observed_relation"] == "tool_used_for_workflow"
    assert result["workflow_use_score"] == 1.0


def test_method_and_review_context_without_workflow_use_is_maybe(monkeypatch):
    _, result = _screen(
        monkeypatch,
        "large language models in systematic reviews",
        "large language models",
        "general capabilities",
        "systematic literature reviews",
    )
    assert result["decision"] == "MAYBE"
    assert result["workflow_use_evidence_missing"] is True
    assert result["relation_keep_gate_passed"] is False


def test_ai_method_without_review_workflow_is_not_keep(monkeypatch):
    _, result = _screen(
        monkeypatch,
        "large language model benchmark",
        "large language model",
        "question answering",
        "natural language processing",
    )
    assert result["decision"] != "KEEP"


def test_review_process_without_ai_is_not_keep(monkeypatch):
    _, result = _screen(
        monkeypatch,
        "manual study selection for systematic reviews",
        "",
        "study selection",
        "systematic review",
    )
    assert result["decision"] in {"MAYBE", "REJECT"}


def test_implied_llm_review_automation_is_not_hard_reject(monkeypatch):
    _, result = _screen(
        monkeypatch,
        "large language models in systematic reviews",
        "large language models",
        "support systematic reviews",
        "systematic literature reviews",
        "technology_being_reviewed",
    )
    assert result["decision"] in {"KEEP", "MAYBE"}
    assert result["review_automation_intent_detected"] is True


def test_review_evaluation_relation_is_detected(monkeypatch):
    _, result = _screen(
        monkeypatch,
        "assess LLM performance in systematic reviews",
        "large language model",
        "probability rating and screening prioritization",
        "systematic reviews",
    )
    assert result["decision"] == "KEEP"
    assert result["review_intent_relation"] == "review_workflow_evaluation"


@pytest.mark.parametrize(
    ("subject", "task", "context"),
    [
        (
            "impact of artificial intelligence tools on competitive intelligence processes",
            "business intelligence methodology and decision support",
            "systematic literature review of competitive intelligence",
        ),
        (
            "bibliometric landscape of text and document classification research",
            "NLP document classification methodology",
            "literature review of natural language processing",
        ),
    ],
)
def test_generic_methodology_process_is_not_review_workflow_methodology(
    monkeypatch,
    subject,
    task,
    context,
):
    _, result = _screen(
        monkeypatch,
        subject,
        "artificial intelligence and natural language processing",
        task,
        context,
    )
    assert result["decision"] != "KEEP"
    assert result["review_intent_relation"] != "review_workflow_methodology"


def test_review_methodology_boilerplate_does_not_make_ai_a_review_tool(monkeypatch):
    _disable_embeddings(monkeypatch)
    rq = _heuristic_research_question_frame(RQ)
    paper = {
        "source_title": "AI applications in supply chain management: a systematic literature review",
        "source_abstract": (
            "The review searched databases and performed data extraction from "
            "included studies before synthesizing AI applications."
        ),
        "primary_subject": "AI applications in supply chains",
        "intervention_or_method": "artificial intelligence",
        "target_problem_or_task": "supply chain resilience and forecasting",
        "application_context": "supply chain management",
        "evidence_type": "systematic review",
        "review_role": "technology_being_reviewed",
    }
    result = semantic_comparator.compare_semantic_frames(rq, paper)
    assert result["decision"] != "KEEP"
    assert result["ai_tool_for_review_workflow"] is False
    assert result["relation_keep_gate_passed"] is False


def test_review_methodology_with_study_object_remains_relevant(monkeypatch):
    _, result = _screen(
        monkeypatch,
        "AI methodology for conducting systematic reviews",
        "artificial intelligence tool",
        "review methodology for study selection and included studies",
        "systematic reviews",
    )
    assert result["decision"] in {"KEEP", "MAYBE"}
    assert result["review_workflow_methodology_validated"] is True


def test_llm_assisted_slr_methodology_is_not_subject_only_review(monkeypatch):
    _disable_embeddings(monkeypatch)
    rq = _heuristic_research_question_frame(RQ)
    paper = {
        "source_title": (
            "A Tri-Level Semantic and LLM-Assisted Methodology for "
            "Systematic Literature Reviews"
        ),
        "source_abstract": (
            "We propose an LLM-assisted methodology to support systematic "
            "literature reviews by screening for relevant literature."
        ),
        "primary_subject": "LLM-assisted methodology for systematic literature reviews",
        "intervention_or_method": "large language model",
        "target_problem_or_task": "screening for relevant literature",
        "application_context": "systematic literature reviews",
        "evidence_type": "method evaluation",
    }
    result = semantic_comparator.compare_semantic_frames(rq, paper)
    assert result["decision"] == "KEEP"
    assert result["paper_observed_relation"] == "tool_used_for_workflow"
    assert result["relation_direction"] == "ai_tool_for_review"
    assert result["subject_review_direction_detected"] is False


def test_systematic_review_of_llm_applications_in_industry_stays_external(
    monkeypatch,
):
    _, result = _screen(
        monkeypatch,
        "systematic literature review of large language model applications in industry",
        "large language models",
        "business applications and decision support",
        "systematic literature review in industry",
        "technology_being_reviewed",
    )
    assert result["decision"] != "KEEP"
    assert result["paper_observed_relation"] in {
        "external_domain_review",
        "technology_reviewed_as_subject",
    }
    assert result["relation_keep_gate_passed"] is False


def test_weakly_specific_ai_subject_review_is_triaged_not_hard_rejected(
    monkeypatch,
):
    _, result = _screen(
        monkeypatch,
        "systematic review of large language models and future directions",
        "large language models",
        "general capabilities and future directions",
        "systematic literature review",
        "technology_being_reviewed",
    )
    assert result["decision"] == "MAYBE"
    assert result["paper_observed_relation"] in {
        "external_domain_review",
        "technology_reviewed_as_subject",
    }
    assert result["weak_subject_review_specificity"] is True
    assert result["relation_keep_gate_passed"] is False


@pytest.mark.parametrize(
    ("subject", "task", "context"),
    [
        (
            "impact of AI implementation in financial management",
            "enhance efficiency and manage risk in financial management processes",
            "systematic literature review in financial management",
        ),
        (
            "generative artificial intelligence in K-12 education",
            "analyzing relevant publications on generative AI applications in K-12 education",
            "systematic literature review in education",
        ),
    ],
)
def test_external_domain_review_methodology_does_not_create_workflow_keep(
    monkeypatch,
    subject,
    task,
    context,
):
    _, result = _screen(
        monkeypatch,
        subject,
        "artificial intelligence",
        task,
        context,
        "technology_being_reviewed",
    )
    assert result["decision"] != "KEEP"
    assert result["ai_tool_for_review_workflow"] is False
    assert result["relation_keep_gate_passed"] is False


def test_stage2_preserves_uncertainty_without_explicit_contradiction():
    allowed, reason = _stage2_reject_demotion_allowed(
        {"review_question_type": "review_workflow_automation"},
        {
            "paper_observed_relation": "unclear_relation",
            "relation_conflict": False,
            "contradiction_score": 0.0,
            "evidence_coverage_count": 2,
        },
    )
    assert allowed is False
    assert "preserved" in reason


def test_stage2_allows_explicit_external_domain_demotion():
    allowed, reason = _stage2_reject_demotion_allowed(
        {"review_question_type": "review_workflow_automation"},
        {
            "paper_observed_relation": "external_domain_review",
            "relation_conflict": True,
            "contradiction_score": 0.9,
            "relation_mismatch_reason": "external-domain subject-only review",
            "evidence_coverage_count": 2,
        },
    )
    assert allowed is True
    assert "external-domain" in reason


def test_review_workflow_diagnostics_exist(monkeypatch):
    _, result = _screen(
        monkeypatch,
        "LLM screening tool for systematic reviews",
        "LLM",
        "study selection and screening",
        "systematic review",
    )
    for field in (
        "review_intent_relation", "review_workflow_task_detected",
        "review_workflow_task_terms", "ai_tool_for_review_workflow",
        "technology_subject_review_detected", "review_role_gate_applied",
        "review_role_gate_reason", "review_relation_confidence",
        "review_automation_intent_detected", "review_automation_intent_terms",
        "review_context_detected", "external_domain_review_detected",
        "review_context_only",
        "paper_observed_relation", "relation_match", "relation_conflict",
        "relation_alignment_score", "relation_confidence",
        "relation_evidence_terms", "relation_mismatch_reason",
        "workflow_use_score", "subject_only_score", "paper_type_only_score",
        "external_domain_topic_score", "implied_workflow_score",
        "uncertainty_preservation_score", "contradiction_score",
        "relation_decision_path", "review_automation_relevance_score",
        "false_keep_risk_score", "keep_suppression_applied",
        "keep_suppression_reason", "keep_required_relation_missing",
        "external_domain_subject_only", "workflow_use_required_for_keep",
        "workflow_use_evidence_missing", "subject_only_overrides_keep",
        "relation_keep_gate_passed",
        "workflow_task_term_detected", "workflow_task_object_detected",
        "workflow_task_object_terms", "workflow_task_sense",
        "workflow_task_sense_confidence", "external_domain_task_detected",
        "external_domain_task_terms", "task_object_mismatch",
        "task_object_mismatch_reason", "implied_workflow_intent_score",
        "implied_workflow_intent_terms", "strong_implied_workflow_intent",
        "workflow_task_object_required", "workflow_task_object_missing",
        "medium_implied_workflow_intent", "workflow_direction_detected",
        "subject_review_direction_detected", "relation_direction",
        "workflow_intent_rescue_applied", "workflow_intent_rescue_reason",
        "weak_subject_review_specificity",
    ):
        assert field in result
