from __future__ import annotations

import json

import pandas as pd
import pytest

import bulk_screen
from benchmark_local_screening import (
    compare_screening_runs, save_blinded_disagreements, summarize_screening_rows,
)
from local_ai.contracts import CriterionEvidence, EvidenceSpan, PaperAssessment, ReviewProtocol
from local_ai.hardware import HardwareSnapshot, resolve_runtime_profile
from local_ai.profiles import resolve_local_screening_profile
from local_ai.rq_frame import build_screening_rq_frame, question_fingerprint
from local_ai.three_layer import (
    GroundedAssessmentBatch, LayerResult, ThreeLayerLocalOrchestrator, critic_batch_prompt,
)
from local_ai.validator import validate_assessment


QUESTION = "How are glimmers used in norvax processes?"


def _submitted(question=QUESTION):
    return {
        "question": question,
        "question_fingerprint": question_fingerprint(question),
        "groups": [
            {"label": "Method", "role": "technology", "terms": ["glimmers"], "source_spans": ["glimmers"]},
            {"label": "Context", "role": "domain", "terms": ["norvax processes"], "source_spans": ["norvax processes"]},
        ],
        "term_details": [
            {"term": "glimmers", "group": "Method", "source": "literal", "supporting_paper_ids": []},
            {"term": "GLM", "group": "Method", "source": "source_acronym", "supporting_paper_ids": []},
            {"term": "norvax process", "group": "Context", "source": "corpus", "supporting_paper_ids": ["p1"]},
        ],
        "model": "qwen3.5:4b", "generation_status": "full",
    }


def _protocol():
    return ReviewProtocol.model_validate({
        "research_question": QUESTION, "objective": "Find direct use.",
        "scope_interpretation": "Require the stated relationship.",
        "criteria": [{
            "id": "rq_core_relationship", "kind": "inclusion", "required": True,
            "description": "Glimmers are used in norvax processes.",
            "expected_evidence": "Direct use evidence.", "source": "research_question",
        }],
    }).with_identity()


def test_generated_frame_preserves_roles_requirements_variants_and_provenance():
    frame = build_screening_rq_frame(QUESTION, submitted=_submitted())
    assert frame.source == "generated_query"
    assert frame.status == "validated"
    assert [group.role for group in frame.groups] == ["technology", "domain"]
    assert all(group.required and group.group_relationship == "AND" for group in frame.groups)
    assert frame.advisory_concepts == ["GLM", "norvax process"]
    corpus = next(item for item in frame.allowed_variants if item.source == "corpus")
    assert corpus.advisory_only and corpus.supporting_paper_ids == ["p1"]


def test_v2_frame_exposes_group_logic_without_flattening_or_terms_into_and_requirements():
    frame = build_screening_rq_frame(
        QUESTION, submitted=_submitted(), frame_version="local-rq-frame-v2"
    )
    payload = frame.compact_prompt_payload()
    assert frame.frame_version == "local-rq-frame-v2"
    assert "required_concepts" not in payload
    assert payload["required_relationship"] == {
        "between_groups": "AND", "within_each_group": "OR",
        "instruction": "Preserve the relationship expressed by the original question.",
    }


def test_stale_or_invalid_generated_frame_falls_back_visibly():
    stale = _submitted("How are other systems used?")
    frame = build_screening_rq_frame(QUESTION, submitted=stale)
    assert frame.source == "parser_fallback"
    assert frame.status == "fallback"
    assert frame.validation_failures
    assert "does not match" in frame.validation_failures[0]


def test_unsupported_corpus_variant_is_rejected_not_silently_accepted():
    submitted = _submitted()
    submitted["term_details"][-1]["supporting_paper_ids"] = []
    frame = build_screening_rq_frame(QUESTION, submitted=submitted)
    assert frame.status == "fallback"
    assert any("supporting paper IDs" in failure for failure in frame.validation_failures)


def test_structured_anchor_preserves_model_semantics_while_baseline_stays_generic():
    raw = _protocol().model_dump(mode="json")
    structured = ThreeLayerLocalOrchestrator._anchor_rq_contract(raw, QUESTION, True)
    baseline = ThreeLayerLocalOrchestrator._anchor_rq_contract(raw, QUESTION, False)
    assert structured["criteria"][0]["description"] == "Glimmers are used in norvax processes."
    assert baseline["criteria"][0]["description"] != structured["criteria"][0]["description"]
    assert structured["criteria"][0]["id"] == "rq_core_relationship"


def test_v41_canonical_criterion_uses_source_linked_rq_not_model_invented_narrowing():
    frame = build_screening_rq_frame(
        QUESTION, submitted=_submitted(), frame_version="local-rq-frame-v2"
    )
    raw = _protocol().model_dump(mode="json")
    raw["criteria"][0]["description"] = "Require real-world optimization and control."
    anchored = ThreeLayerLocalOrchestrator._anchor_rq_contract(
        raw, QUESTION, True, frame, True
    )
    description = anchored["criteria"][0]["description"]
    assert QUESTION in description
    assert "glimmers" in description and "norvax processes" in description
    assert "optimization" not in description and "control" not in description
    assert anchored["semantic_boundaries"] == frame.forbidden_broadening_warnings[:6]


def test_v41_explicit_user_criteria_are_anchored_verbatim():
    frame = build_screening_rq_frame(
        QUESTION, submitted=_submitted(), frame_version="local-rq-frame-v2"
    )
    raw = _protocol().model_dump(mode="json")
    raw["criteria"].append({
        "id": "invented", "kind": "inclusion", "required": True,
        "description": "A distorted paraphrase.", "expected_evidence": "Anything.", "source": "user",
    })
    anchored = ThreeLayerLocalOrchestrator._anchor_rq_contract(
        raw, QUESTION, True, frame, True,
        "Include the stated population", "Exclude editorials",
    )
    user = [criterion for criterion in anchored["criteria"] if criterion["source"] == "user"]
    assert [(criterion["kind"], criterion["description"]) for criterion in user] == [
        ("inclusion", "Include the stated population"),
        ("exclusion", "Exclude editorials"),
    ]


def test_v41_two_unit_evidence_and_required_group_coverage():
    frame = build_screening_rq_frame(
        QUESTION, submitted=_submitted(), frame_version="local-rq-frame-v2"
    )
    parsed = GroundedAssessmentBatch.model_validate({"items": [{
        "p": "p1", "d": "KEEP", "k": "LOW", "r": "Direct relationship.",
        "c": [{"c": "rq_core_relationship", "v": "MET",
               "e": ["abstract_001", "abstract_002"], "r": "Both groups are linked."}],
    }]})
    assert parsed.items[0].c[0].e == ["abstract_001", "abstract_002"]
    assessment = PaperAssessment(
        summary="Direct relationship.", decision="KEEP", confidence=.9,
        reason="Direct relationship.", criteria=[CriterionEvidence(
            criterion_id="rq_core_relationship", verdict="MET", rationale="Linked.",
            evidence=[EvidenceSpan(source="abstract", evidence_id="abstract_001"),
                      EvidenceSpan(source="abstract", evidence_id="abstract_002")],
        )],
    )
    report = validate_assessment(
        assessment, _protocol(), "A study", "Glimmers are evaluated. They are used in norvax processes.", frame
    )
    assert report.valid
    assert all(report.rq_group_coverage.values())
    missing = validate_assessment(
        assessment.model_copy(update={"criteria": [CriterionEvidence(
            criterion_id="rq_core_relationship", verdict="MET", rationale="Linked.",
            evidence=[EvidenceSpan(source="abstract", evidence_id="abstract_001")],
        )]}), _protocol(), "A study", "Glimmers are evaluated. An unrelated setting is discussed.", frame
    )
    assert not missing.valid
    assert any("required RQ groups" in error for error in missing.errors)


def test_structured_critic_is_prediction_blind_and_receives_frame():
    frame = build_screening_rq_frame(QUESTION, submitted=_submitted())
    prior = {"p1": LayerResult({
        "decision": "REJECT", "decision_risk": "HIGH",
        "reason": "SECRET PRIOR DECISION", "validation_errors": ["SECRET ERROR"],
    }, None, None, 0, 0)}
    prompt = critic_batch_prompt(
        _protocol(), [{"id": "p1", "title": "Paper", "abstract": "Abstract."}], prior, frame
    )
    assert frame.frame_id in prompt
    assert "SECRET PRIOR DECISION" not in prompt
    assert "SECRET ERROR" not in prompt


def test_quick_reject_is_never_resumed_as_final(tmp_path):
    checkpoint = tmp_path / "checkpoint.csv"
    pd.DataFrame([{
        "Source_Row_Index": 1, "Protocol_ID": "run", "Validation_Status": "validated",
        "Prompt_Version": "local-structured-rq-v4.0", "Decision": "REJECT",
        "Decision_Risk": "LOW", "Layer_Trace_JSON": '[{"name":"quick_triage"}]',
    }]).to_csv(checkpoint, index=False)
    assert bulk_screen._local_resume_rows(
        str(checkpoint), "run", "local-structured-rq-v4.0"
    ) == {}


def test_v41_quick_keep_is_not_resumed_as_final(tmp_path):
    checkpoint = tmp_path / "checkpoint.csv"
    pd.DataFrame([{
        "Source_Row_Index": 1, "Protocol_ID": "run", "Validation_Status": "validated",
        "Prompt_Version": "local-evidence-grounded-rq-v4.1", "Decision": "KEEP",
        "Decision_Risk": "LOW", "Layer_Trace_JSON": '[{"name":"quick_triage"}]',
    }]).to_csv(checkpoint, index=False)
    assert bulk_screen._local_resume_rows(
        str(checkpoint), "run", "local-evidence-grounded-rq-v4.1"
    ) == {}


def test_missing_experimental_model_fails_without_downgrade():
    hardware = HardwareSnapshot(
        24, 12, 8, "test", "GPU", 6,
        {"qwen2.5:3b": 1, "qwen3:4b-instruct-2507-q4_K_M": 1},
    )
    runtime = resolve_runtime_profile("performance", "balanced", hardware)
    pipeline = ThreeLayerLocalOrchestrator(
        runtime, screening_profile="structured-qwen35-4b"
    )
    with pytest.raises(Exception, match="No automatic downgrade"):
        pipeline.require_profile_models()


def test_local_audit_columns_and_unconfirmed_disagreement_queues():
    row = bulk_screen._row_from_result({}, "T", "A", {
        "decision": "KEEP", "reason": "r", "confidence": .9,
        "rq_frame_id": "frame", "rq_frame_source": "generated_query",
        "rq_frame_version": "local-rq-frame-v2", "rq_frame_status": "validated",
        "local_profile": "structured-current",
        "protocol_model": "p", "deep_model": "d", "edge_model": "e",
    }, 1)
    assert row["RQ_Frame_ID"] == "frame"
    assert row["RQ_Frame_Version"] == "local-rq-frame-v2"
    assert row["Local_Profile"] == "structured-current"
    report = compare_screening_runs([row], [{"Source_Row_Index": 1, "Decision": "REJECT"}])
    assert len(report["suspicious_false_keep_candidates"]) == 1
    assert "not human gold" in report["suspicious_queue_note"]
    summary = summarize_screening_rows([row], expected_ranges={"keep": [0, 0]})
    assert "expected_ranges_passed" not in summary
    assert "descriptive only" in summary["decision_ratio_note"]


def test_profile_registry_keeps_baseline_production_only():
    assert resolve_local_screening_profile("baseline-v3.12").production
    assert not resolve_local_screening_profile("structured-current").production
    grounded = resolve_local_screening_profile("structured-grounded-v4.1")
    assert grounded.evidence_grounded and grounded.require_deep_review


def test_v41_requires_deep_review_even_for_low_risk_triage_keep():
    pipeline = ThreeLayerLocalOrchestrator(screening_profile="structured-grounded-v4.1")
    quick_keep = LayerResult({
        "decision": "KEEP", "decision_risk": "LOW", "validation_status": "validated",
    }, None, None, 0, 0)
    assert not ThreeLayerLocalOrchestrator.needs_deep_review(quick_keep)
    assert pipeline.requires_deep_review(quick_keep)


def test_layer_runtime_is_deduplicated_and_disagreement_export_is_blinded(tmp_path):
    metric = json.dumps([{"layer": "deep_review", "batch_id": "deep-1", "retry": 0,
                          "wall_seconds": 2.5}])
    rows = [{
        "Source_Row_Index": index, "Title": f"Paper {index}", "Abstract": "Text",
        "Decision": decision, "Validation_Status": "validated", "Layer_Metrics_JSON": metric,
    } for index, decision in ((1, "KEEP"), (2, "MAYBE"))]
    summary = summarize_screening_rows(rows)
    assert summary["per_layer_runtime_seconds"]["deep_review"] == 2.5
    assert summary["batch_calls_recorded"] == 1
    output = tmp_path / "labels.csv"
    export = save_blinded_disagreements(
        rows, [{"Source_Row_Index": 1, "Decision": "REJECT"},
               {"Source_Row_Index": 2, "Decision": "MAYBE"}], QUESTION, str(output),
        tmp_path / "private",
    )
    labels = pd.read_csv(output, dtype=str, keep_default_na=False)
    assert len(labels) == 1 and labels.loc[0, "Gold_Decision"] == ""
    assert not ({"Decision", "candidate_decision", "baseline_decision"} & set(labels.columns))
    manifest = json.loads(open(export["private_manifest_path"], encoding="utf-8").read())
    assert manifest["rows"][0]["candidate_decision"] == "KEEP"
