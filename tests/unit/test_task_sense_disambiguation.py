import pytest

import semantic_comparator
from semantic_frame import _heuristic_research_question_frame
from stage2_arbitration import arbitrate_stage2
from task_sense_disambiguator import classify_task_sense


def _frame(subject, task="", context=""):
    return {
        "primary_subject": subject,
        "target_problem_or_task": task,
        "application_context": context,
    }


@pytest.mark.parametrize(
    ("subject", "task"),
    [
        ("LLM screening abstracts", "screening abstracts for systematic reviews"),
        ("data extraction tool", "data extraction from included studies"),
        ("evidence assistant", "summarizing study findings for evidence synthesis"),
        ("automated search", "literature search automation for evidence reviews"),
    ],
)
def test_review_objects_produce_review_workflow_task(subject, task):
    result = classify_task_sense(_frame(subject, task, "systematic review"))
    assert result["workflow_task_sense"] == "review_workflow_task"
    assert result["workflow_task_object_detected"] is True


@pytest.mark.parametrize(
    ("subject", "task"),
    [
        ("patient classifier", "classification of patients"),
        ("breast cancer model", "diagnosis of breast cancer"),
        ("medical report OCR", "text extraction from medical reports"),
    ],
)
def test_external_objects_produce_external_domain_task(subject, task):
    result = classify_task_sense(_frame(subject, task, "healthcare"))
    assert result["workflow_task_sense"] == "external_domain_task"
    assert result["task_object_mismatch"] is True


def test_ai_assisted_review_pipeline_has_strong_implied_intent():
    result = classify_task_sense(_frame(
        "AI-assisted systematic review",
        "review automation pipeline",
        "systematic review",
    ))
    assert result["strong_implied_workflow_intent"] is True
    assert result["workflow_task_sense"] == "review_workflow_task"


@pytest.mark.parametrize(
    "text",
    [
        "Generative AI to accelerate systematic literature reviews",
        "automated systematic literature reviews",
        "implementation of a research assistant for literature review with LLMs",
    ],
)
def test_strong_implied_workflow_phrases(text):
    result = classify_task_sense(_frame(text, "", "systematic review"))
    assert result["implied_workflow_intent_score"] >= 0.85
    assert result["strong_implied_workflow_intent"] is True
    assert result["relation_direction"] == "ai_tool_for_review"


def test_llm_assisted_methodology_for_slr_is_workflow_direction():
    result = classify_task_sense({
        "source_title": (
            "A Tri-Level Semantic and LLM-Assisted Methodology for "
            "Systematic Literature Reviews"
        ),
        "source_abstract": (
            "The methodology supports screening for relevant literature in "
            "systematic reviews."
        ),
        "primary_subject": "LLM-assisted methodology for systematic literature reviews",
        "target_problem_or_task": "screening for relevant literature",
        "application_context": "systematic literature reviews",
    })
    assert result["strong_implied_workflow_intent"] is True
    assert result["workflow_task_sense"] == "review_workflow_task"
    assert result["relation_direction"] == "ai_tool_for_review"


def test_raw_source_title_drives_intent_when_model_subject_is_generic():
    result = classify_task_sense({
        "source_title": "Implementation of a research assistant for literature review with LLMs",
        "source_abstract": "The assistant supports reviewers across review stages.",
        "primary_subject": "academic research",
        "target_problem_or_task": "analysis",
        "application_context": "literature reviews",
    })
    assert result["strong_implied_workflow_intent"] is True
    assert result["relation_direction"] == "ai_tool_for_review"


def test_llms_for_systematic_reviews_has_medium_tool_direction():
    result = classify_task_sense(_frame(
        "LLMs for systematic reviews",
        "",
        "systematic reviews",
    ))
    assert 0.60 <= result["implied_workflow_intent_score"] < 0.85
    assert result["medium_implied_workflow_intent"] is True
    assert result["relation_direction"] == "ai_tool_for_review"


def test_systematic_review_of_llms_has_subject_direction():
    result = classify_task_sense(_frame(
        "systematic review of LLMs in medicine",
        "diagnosis",
        "medicine",
    ))
    assert result["subject_review_direction_detected"] is True
    assert result["relation_direction"] == "review_about_ai"


def test_ambiguous_review_task_stays_ambiguous():
    result = classify_task_sense(_frame(
        "AI and literature reviews",
        "classification and analysis",
        "literature review",
    ))
    assert result["workflow_task_sense"] == "ambiguous_task"


def test_stage2_uses_task_sense_for_preservation_and_demotion():
    rq = {"rq_type": "review_workflow_automation"}
    ambiguous = arbitrate_stage2("MAYBE", {
        "decision": "REJECT",
        "workflow_task_sense": "ambiguous_task",
        "relation_conflict": False,
        "contradiction_score": 0.0,
    }, rq)
    external = arbitrate_stage2("MAYBE", {
        "decision": "REJECT",
        "workflow_task_sense": "external_domain_task",
        "strong_implied_workflow_intent": False,
        "paper_observed_relation": "external_domain_review",
        "relation_conflict": False,
        "contradiction_score": 0.0,
    }, rq)
    assert ambiguous["decision"] == "MAYBE"
    assert ambiguous["stage2_preserved_due_to_ambiguous_task_sense"] is True
    assert external["decision"] == "REJECT"
    assert external["stage2_task_sense_conflict"] is True


def test_stage2_rescues_medium_and_promotes_strong_workflow_intent():
    rq = {"rq_type": "review_workflow_automation"}
    medium = arbitrate_stage2("REJECT", {
        "decision": "REJECT",
        "medium_implied_workflow_intent": True,
        "workflow_direction_detected": True,
        "subject_review_direction_detected": False,
        "relation_direction": "ai_tool_for_review",
        "relation_conflict": False,
        "contradiction_score": 0.0,
    }, rq)
    strong = arbitrate_stage2("MAYBE", {
        "decision": "MAYBE",
        "strong_implied_workflow_intent": True,
        "medium_implied_workflow_intent": True,
        "workflow_direction_detected": True,
        "subject_review_direction_detected": False,
        "relation_direction": "ai_tool_for_review",
        "relation_conflict": False,
        "contradiction_score": 0.0,
    }, rq)
    assert medium["decision"] == "MAYBE"
    assert medium["stage2_workflow_intent_rescue"] is True
    assert strong["decision"] == "KEEP"
    assert strong["stage2_workflow_intent_rescue"] is True


def test_stage2_raw_keep_with_strong_intent_bypasses_legacy_guard():
    result = arbitrate_stage2("MAYBE", {
        "decision": "KEEP",
        "implied_workflow_intent_score": 0.95,
        "strong_implied_workflow_intent": True,
        "workflow_direction_detected": True,
        "subject_review_direction_detected": False,
        "relation_conflict": False,
        "contradiction_score": 0.0,
    }, {"rq_type": "review_workflow_automation"}, keep_allowed=False)
    assert result["decision"] == "KEEP"
    assert result["stage2_promoted_to_keep"] is True
    assert "strong implied" in result["stage2_override_reason"].lower()


def test_medical_and_blockchain_tasks_are_not_affected(monkeypatch):
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
    medical = _heuristic_research_question_frame(
        "What machine learning and deep learning methods have been proposed for heart disease prediction and diagnosis?"
    )
    blockchain = _heuristic_research_question_frame(
        "How is blockchain technology used to improve transparency, traceability, trust, and security in supply chain management?"
    )
    medical_result = semantic_comparator.compare_semantic_frames(medical, {
        "primary_subject": "heart disease classification",
        "intervention_or_method": "deep learning",
        "target_problem_or_task": "diagnosis and prediction",
        "application_context": "cardiovascular healthcare",
    })
    blockchain_result = semantic_comparator.compare_semantic_frames(blockchain, {
        "primary_subject": "blockchain supply chain",
        "intervention_or_method": "blockchain",
        "target_problem_or_task": "traceability and transparency",
        "application_context": "supply chain management",
    })
    assert medical_result["decision"] == "KEEP"
    assert blockchain_result["decision"] == "KEEP"
