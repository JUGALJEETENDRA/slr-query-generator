from benchmark_local_screening import summarize_screening_rows
from decision_policy import DEFAULT_DECISION_POLICY
from evidence_ledger import build_evidence_ledger
from screening_contracts import build_paper_contract, build_rq_contract
from semantic_frame import _heuristic_research_question_frame
from stage2_arbitration import arbitrate_stage2
import semantic_comparator
from bulk_screen import _stage2_escalation_candidate


def _ledger(required, scores, relation):
    return build_evidence_ledger(
        required,
        scores,
        {
            "method": "LLM",
            "task": "screening",
            "context": "systematic review",
            "relation": "LLM used for screening",
        },
        relation,
        {
            "contradiction_score": relation.get("contradiction_score", 0.0),
            "contradiction_reason": relation.get("relation_mismatch_reason", ""),
        },
    )


def test_rq_contracts_for_three_supported_question_types():
    cases = (
        (
            "What machine learning and deep learning methods have been proposed for heart disease prediction and diagnosis?",
            "method_for_task_in_domain",
            "method_used_for_task",
        ),
        (
            "How is blockchain technology used to improve transparency, traceability, trust, and security in supply chain management?",
            "technology_for_outcome_in_domain",
            "technology_used_for_outcome",
        ),
        (
            "Can large language models and artificial intelligence tools help automate systematic literature reviews?",
            "review_workflow_automation",
            "tool_used_for_workflow",
        ),
    )
    for text, rq_type, relation in cases:
        frame = _heuristic_research_question_frame(text)
        contract = build_rq_contract(text, frame)
        assert contract["rq_type"] == rq_type
        assert contract["rq_desired_relation"] == relation
        assert contract["rq_required_dimensions"]
        assert contract["rq_extraction_suspect"] is False


def test_paper_evidence_contract_supports_arrays():
    contract = build_paper_contract("Title", "Abstract", {
        "methods_or_technologies": "LLM; GPT-4",
        "target_tasks_or_outcomes": "screening; study selection",
        "application_contexts": "systematic review; evidence review",
    }, {"paper_observed_relation": "tool_used_for_workflow"})
    assert contract["paper_methods"] == ["LLM", "GPT-4"]
    assert contract["paper_tasks"] == ["screening", "study selection"]
    assert contract["paper_observed_relation"] == "tool_used_for_workflow"


def test_evidence_ledger_tracks_coverage_and_missing_relation():
    relation = {"relation_match": False, "relation_conflict": False}
    ledger = _ledger(
        ["method_tool", "review_workflow_process", "review_context", "relation"],
        {
            "method_tool": 1.0,
            "review_workflow_process": 1.0,
            "review_context": 1.0,
            "relation": 0.4,
        },
        relation,
    )
    assert ledger["evidence_coverage_count"] == 3
    assert "relation" in ledger["required_dimensions_missing"]
    assert ledger["relation_unclear"] is True


def test_decision_policy_keep_maybe_and_reject():
    rq = {
        "rq_type": "review_workflow_automation",
        "rq_required_dimensions": [
            "method_tool", "review_workflow_process", "review_context", "relation"
        ],
    }
    matched = {
        "relation_match": True, "relation_conflict": False,
        "workflow_use_score": 1.0,
    }
    unclear = {
        "relation_match": False, "relation_conflict": False,
        "workflow_use_score": 0.0,
    }
    conflict = {
        "relation_match": False, "relation_conflict": True,
        "workflow_use_score": 0.0, "contradiction_score": 0.9,
        "external_domain_topic_score": 0.9,
        "relation_mismatch_reason": "external domain review",
    }
    full = _ledger(rq["rq_required_dimensions"], {
        "method_tool": 1.0, "review_workflow_process": 1.0,
        "review_context": 1.0, "relation": 1.0,
    }, matched)
    partial = _ledger(rq["rq_required_dimensions"], {
        "method_tool": 1.0, "review_workflow_process": 0.0,
        "review_context": 1.0, "relation": 0.45,
    }, unclear)
    rejected = _ledger(rq["rq_required_dimensions"], {
        "method_tool": 1.0, "review_workflow_process": 0.0,
        "review_context": 1.0, "relation": 0.0,
    }, conflict)
    assert DEFAULT_DECISION_POLICY.decide(rq, full, matched)["decision"] == "KEEP"
    assert DEFAULT_DECISION_POLICY.decide(rq, partial, unclear)["decision"] == "MAYBE"
    assert DEFAULT_DECISION_POLICY.decide(rq, rejected, conflict)["decision"] == "REJECT"


def test_decision_policy_triages_weak_subject_review_specificity_to_maybe():
    rq = {
        "rq_type": "review_workflow_automation",
        "rq_required_dimensions": [
            "method_tool", "review_workflow_process", "review_context", "relation"
        ],
    }
    conflict = {
        "paper_observed_relation": "external_domain_review",
        "relation_match": False,
        "relation_conflict": True,
        "workflow_use_score": 0.0,
        "contradiction_score": 0.9,
        "external_domain_topic_score": 0.9,
        "relation_mismatch_reason": "external domain review",
    }
    ledger = _ledger(rq["rq_required_dimensions"], {
        "method_tool": 1.0,
        "review_workflow_process": 0.0,
        "review_context": 1.0,
        "relation": 0.0,
    }, conflict)
    weak_subject_review = {
        "workflow_task_sense": "no_task_detected",
        "external_domain_terms": "",
        "external_domain_task_detected": False,
        "workflow_direction_detected": False,
    }
    explicit_external_task = {
        **weak_subject_review,
        "workflow_task_sense": "external_domain_task",
        "external_domain_task_detected": True,
    }

    triaged = DEFAULT_DECISION_POLICY.decide(
        rq,
        ledger,
        conflict,
        weak_subject_review,
    )
    rejected = DEFAULT_DECISION_POLICY.decide(
        rq,
        ledger,
        conflict,
        explicit_external_task,
    )

    assert triaged["decision"] == "MAYBE"
    assert triaged["weak_subject_review_specificity"] is True
    assert rejected["decision"] == "REJECT"


def test_stage2_arbitration_preserves_uncertainty_and_requires_contradiction():
    rq = {"rq_type": "review_workflow_automation"}
    preserved = arbitrate_stage2(
        "MAYBE",
        {"decision": "REJECT", "relation_conflict": False, "contradiction_score": 0.0},
        rq,
    )
    demoted = arbitrate_stage2(
        "MAYBE",
        {
            "decision": "REJECT", "relation_conflict": True,
            "contradiction_score": 0.9, "relation_mismatch_reason": "explicit conflict",
        },
        rq,
    )
    assert preserved["decision"] == "MAYBE"
    assert preserved["stage2_uncertainty_preserved"] is True
    assert demoted["decision"] == "REJECT"
    assert demoted["stage2_explicit_contradiction"] is True


def test_benchmark_reports_ranges_suspicious_keeps_and_demotions():
    summary = summarize_screening_rows(
        [
            {
                "Decision": "KEEP", "Title": "A",
                "stage1_relation_match": False,
                "stage1_suspicious_keep": True,
                "stage1_false_keep_risk_score": 0.8,
                "stage2_demoted_to_reject": False,
            },
            {
                "Decision": "REJECT", "Title": "B",
                "stage1_relation_match": False,
                "stage2_demoted_to_reject": True,
            },
        ],
        expected_ranges={"keep": [2, 3], "maybe": [0, 1], "reject": [0, 1]},
    )
    assert summary["expected_ranges_passed"] is False
    assert summary["top_suspicious_keeps"] == ["A"]
    assert summary["stage2_demotion_summary"]["count"] == 1


def test_embedding_cache_reuses_rq_text_and_only_encodes_misses(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.calls = []

        def encode(self, texts):
            self.calls.append(list(texts))
            return [[float(index), 1.0] for index, _ in enumerate(texts)]

    model = FakeModel()
    monkeypatch.setattr(semantic_comparator, "MODEL", model)
    semantic_comparator.clear_embedding_cache()

    semantic_comparator._encode_by_index(["same rq", "paper one"])
    semantic_comparator._encode_by_index(["same rq", "paper two"])

    assert model.calls == [["same rq", "paper one"], ["paper two"]]
    semantic_comparator.clear_embedding_cache()


def test_strong_stage2_intent_promotes_even_when_legacy_guard_blocks():
    result = arbitrate_stage2(
        "MAYBE",
        {
            "decision": "KEEP",
            "implied_workflow_intent_score": 0.95,
            "workflow_direction_detected": True,
            "subject_review_direction_detected": False,
            "relation_conflict": False,
            "contradiction_score": 0.0,
        },
        {"rq_type": "review_workflow_automation"},
        keep_allowed=False,
    )
    assert result["decision"] == "KEEP"
    assert result["stage2_promoted_to_keep"] is True


def test_adaptive_stage2_skips_stable_uncertainty_but_escalates_actionable_rows():
    rq = {"rq_type": "review_workflow_automation"}
    skipped, skip_reason = _stage2_escalation_candidate(rq, {
        "workflow_task_sense": "ambiguous_task",
        "evidence_coverage_ratio": 0.5,
        "relation_conflict": False,
        "medium_implied_workflow_intent": False,
        "external_domain_task_detected": False,
    })
    selected, select_reason = _stage2_escalation_candidate(rq, {
        "workflow_task_sense": "review_workflow_task",
        "evidence_coverage_ratio": 0.75,
        "relation_conflict": False,
        "medium_implied_workflow_intent": True,
        "external_domain_task_detected": False,
    })
    assert skipped is False
    assert "preserved" in skip_reason
    assert selected is True
    assert "workflow" in select_reason


def test_adaptive_stage2_skips_weak_subject_review_specificity():
    skipped, reason = _stage2_escalation_candidate(
        {"rq_type": "review_workflow_automation"},
        {
            "weak_subject_review_specificity": True,
            "relation_conflict": True,
            "medium_implied_workflow_intent": False,
            "workflow_task_sense": "no_task_detected",
            "evidence_coverage_ratio": 0.5,
        },
    )
    assert skipped is False
    assert "Weakly specific" in reason
