import pandas as pd

import semantic_comparator
from benchmark_local_screening import summarize_screening_rows
from corpus_profiler import profile_corpus
from domain_vocabulary import analyze_research_question
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
        lambda embeddings, left, right: 0.05,
    )


def test_blockchain_supply_chain_rq_extraction():
    frame = _heuristic_research_question_frame(
        "How is blockchain technology used to improve transparency, traceability, trust, and security in supply chain management?"
    )

    assert frame["review_question_type"] == "technology_in_domain_review"
    assert "blockchain" in frame["method_or_technology"]
    assert "supply chain" in frame["application_context"]
    assert "traceability" in frame["target_tasks_or_outcomes"]
    assert "provenance" in frame["task_outcome_synonyms"]


def test_medical_and_slr_rq_extraction_still_work():
    medical = _heuristic_research_question_frame(
        "What machine learning and deep learning methods have been proposed for heart disease prediction and diagnosis?"
    )
    slr = _heuristic_research_question_frame(
        "Can large language models help automate systematic literature reviews?"
    )

    assert "machine learning" in medical["method_or_technology"]
    assert "cardiovascular" in medical["core_domain"]
    assert "prediction" in medical["target_tasks_or_outcomes"]
    assert slr["review_question_type"] == "review_workflow_automation"
    assert "large language" in slr["method_or_technology"]
    assert "study selection" in slr["task_outcome_synonyms"]


def test_corpus_profiler_extracts_blockchain_supply_chain_vocabulary():
    rows = pd.DataFrame(
        [
            {
                "Title": "Blockchain traceability for agri-food supply chain",
                "Abstract": "Smart contracts improve provenance, transparency, and trust in food supply chain logistics.",
            },
            {
                "Title": "Hyperledger platform for supply chain data integrity",
                "Abstract": "A blockchain platform supports anti-counterfeit tracking and visibility.",
            },
        ]
    )

    profile = profile_corpus(rows, "Title", "Abstract")

    assert "blockchain" in profile["corpus_method_terms"]
    assert "smart contracts" in profile["corpus_method_terms"]
    assert "traceability" in profile["corpus_task_terms"]
    assert "supply chain" in profile["corpus_context_terms"]


def test_blockchain_supply_chain_traceability_paper_is_keep(monkeypatch):
    _disable_embeddings(monkeypatch)
    rq = _heuristic_research_question_frame(
        "How is blockchain technology used to improve transparency, traceability, trust, and security in supply chain management?"
    )
    paper = {
        "primary_subject": "blockchain supply chain traceability",
        "intervention_or_method": "blockchain and smart contracts",
        "target_problem_or_task": "traceability, transparency, provenance, and trust",
        "application_context": "food supply chain management",
        "evidence_type": "framework and platform evaluation",
        "review_role": "",
    }

    result = semantic_comparator.compare_semantic_frames(rq, paper)

    assert result["decision"] == "KEEP"
    assert result["decision_path"] == "relation_keep_confirmed"
    assert result["method_alignment_score"] >= 0.45
    assert result["context_alignment_score"] >= 0.35


def test_smart_contract_food_supply_chain_transparency_is_keep(monkeypatch):
    _disable_embeddings(monkeypatch)
    rq = _heuristic_research_question_frame(
        "How is blockchain technology used to improve transparency, traceability, trust, and security in supply chain management?"
    )
    paper = {
        "primary_subject": "food supply chain platform",
        "intervention_or_method": "smart contracts on Ethereum",
        "target_problem_or_task": "transparency and data integrity",
        "application_context": "food supply chain",
        "evidence_type": "prototype system",
        "review_role": "",
    }

    assert semantic_comparator.compare_semantic_frames(rq, paper)["decision"] == "KEEP"


def test_blockchain_iot_agro_food_traceability_is_keep_or_maybe(monkeypatch):
    _disable_embeddings(monkeypatch)
    rq = _heuristic_research_question_frame(
        "How is blockchain technology used to improve transparency, traceability, trust, and security in supply chain management?"
    )
    paper = {
        "primary_subject": "agro food traceability",
        "intervention_or_method": "blockchain and IoT",
        "target_problem_or_task": "traceability",
        "application_context": "agri-food supply chain",
        "evidence_type": "architecture",
        "review_role": "",
    }

    assert semantic_comparator.compare_semantic_frames(rq, paper)["decision"] in {"KEEP", "MAYBE"}


def test_negative_blockchain_contexts_are_not_kept(monkeypatch):
    _disable_embeddings(monkeypatch)
    rq = _heuristic_research_question_frame(
        "How is blockchain technology used to improve transparency, traceability, trust, and security in supply chain management?"
    )
    for paper in (
        {
            "primary_subject": "cryptocurrency trading",
            "intervention_or_method": "blockchain",
            "target_problem_or_task": "bitcoin price prediction",
            "application_context": "cryptocurrency trading",
            "evidence_type": "empirical",
            "review_role": "",
        },
        {
            "primary_subject": "consensus protocol",
            "intervention_or_method": "blockchain consensus algorithm",
            "target_problem_or_task": "proof of work scalability",
            "application_context": "generic blockchain security",
            "evidence_type": "benchmark",
            "review_role": "",
        },
        {
            "primary_subject": "cybersecurity",
            "intervention_or_method": "blockchain",
            "target_problem_or_task": "access control",
            "application_context": "cybersecurity",
            "evidence_type": "framework",
            "review_role": "",
        },
    ):
        assert semantic_comparator.compare_semantic_frames(rq, paper)["decision"] in {"REJECT", "MAYBE"}


def test_blockchain_supply_chain_review_is_keep_or_maybe(monkeypatch):
    _disable_embeddings(monkeypatch)
    rq = _heuristic_research_question_frame(
        "How is blockchain technology used to improve transparency, traceability, trust, and security in supply chain management?"
    )
    paper = {
        "primary_subject": "survey of blockchain for supply chain traceability",
        "intervention_or_method": "blockchain",
        "target_problem_or_task": "traceability and provenance",
        "application_context": "supply chain management",
        "evidence_type": "survey",
        "review_role": "technology_being_reviewed",
    }

    assert semantic_comparator.compare_semantic_frames(rq, paper)["decision"] in {"KEEP", "MAYBE"}


def test_general_llm_survey_does_not_pass_slr_review_gate(monkeypatch):
    _disable_embeddings(monkeypatch)
    rq = _heuristic_research_question_frame(
        "Can large language models help automate systematic literature reviews?"
    )
    paper = {
        "primary_subject": "survey of large language models",
        "intervention_or_method": "large language models",
        "target_problem_or_task": "general capabilities",
        "application_context": "natural language processing",
        "evidence_type": "survey",
        "review_role": "technology_being_reviewed",
    }

    result = semantic_comparator.compare_semantic_frames(rq, paper)

    assert result["decision"] == "REJECT"
    assert result["decision_path"] == "relation_conflict_reject"


def test_strong_structured_evidence_overrides_weak_embedding(monkeypatch):
    _disable_embeddings(monkeypatch)
    rq = _heuristic_research_question_frame(
        "How is blockchain technology used to improve transparency, traceability, trust, and security in supply chain management?"
    )
    paper = {
        "primary_subject": "blockchain supply chain",
        "intervention_or_method": "Hyperledger blockchain",
        "target_problem_or_task": "provenance and traceability",
        "application_context": "pharmaceutical supply chain",
        "evidence_type": "case study",
        "review_role": "",
    }

    result = semantic_comparator.compare_semantic_frames(rq, paper)

    assert result["decision"] == "KEEP"
    assert result["technology_match"] == 0.05


def test_weak_context_prevents_structured_keep(monkeypatch):
    _disable_embeddings(monkeypatch)
    rq = _heuristic_research_question_frame(
        "How is blockchain technology used to improve transparency, traceability, trust, and security in supply chain management?"
    )
    paper = {
        "primary_subject": "blockchain transparency",
        "intervention_or_method": "blockchain",
        "target_problem_or_task": "transparency and trust",
        "application_context": "electronic voting",
        "evidence_type": "system",
        "review_role": "",
    }

    assert semantic_comparator.compare_semantic_frames(rq, paper)["decision"] != "KEEP"


def test_diagnostic_fields_exist(monkeypatch):
    _disable_embeddings(monkeypatch)
    rq = _heuristic_research_question_frame(
        "How is blockchain technology used to improve transparency, traceability, trust, and security in supply chain management?"
    )
    paper = {
        "primary_subject": "blockchain supply chain",
        "intervention_or_method": "blockchain",
        "target_problem_or_task": "traceability",
        "application_context": "supply chain",
        "evidence_type": "framework",
        "review_role": "",
    }

    result = semantic_comparator.compare_semantic_frames(rq, paper)

    for field in (
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
    ):
        assert field in result


def test_benchmark_harness_flags_suspicious_all_reject():
    summary = summarize_screening_rows(
        [
            {"Decision": "REJECT", "Title": "A"},
            {"Decision": "REJECT", "Title": "B"},
        ]
    )

    assert "suspicious all-reject warning" in summary["warnings"]
