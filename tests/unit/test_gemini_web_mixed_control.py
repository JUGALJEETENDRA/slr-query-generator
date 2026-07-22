import json

import pandas as pd

from evaluation.gemini_web_mixed_control import (
    build_mixed_control,
    compare_screening_runs,
    score_mixed_control,
)


def _source(path, prefix, count):
    pd.DataFrame({
        "Title": [f"{prefix} title {index}" for index in range(count)],
        "Abstract": [f"{prefix} abstract {index}" for index in range(count)],
        "Gold_Decision": ["MUST_NOT_LEAK"] * count,
    }).to_csv(path, index=False)


def _screened_row(index, title, decision, *, valid=True, escalated=False):
    return {
        "Source_Row_Index": index, "Title": title, "Abstract": "Evidence.",
        "Decision": decision, "Reason": "Grounded reason.", "Confidence": .9,
        "Validation_Status": "validated" if valid else "unresolved",
        "Escalated": escalated, "Processing_Seconds": 1.25,
        "Criteria_JSON": "[]", "Evidence_JSON": json.dumps([{
            "criterion_id": "inc1", "source": "abstract",
            "evidence_id": "abstract_001", "quote": "Evidence.",
        }]),
        "Layer_Trace_JSON": "[]", "Uncertainty_JSON": "[]",
        "Contradictions_JSON": "[]", "Validation_Errors": "[]",
        "Failure_Class": "", "Critic_Route": "", "Verification_Status": "not_required",
    }


def test_builder_interleaves_groups_and_blinds_labels(tmp_path):
    positive = tmp_path / "positive.csv"
    negative = tmp_path / "negative.csv"
    papers = tmp_path / "mixed.csv"
    gold = tmp_path / "mixed.gold.csv"
    _source(positive, "positive", 3)
    _source(negative, "negative", 3)

    summary = build_mixed_control(positive, negative, papers, gold, per_group=3)

    screening = pd.read_csv(papers)
    labels = pd.read_csv(gold)
    assert summary["labels_blinded"]
    assert list(screening.columns) == ["Title", "Abstract"]
    assert "MUST_NOT_LEAK" not in papers.read_text(encoding="utf-8")
    assert list(labels["Gold_Decision"]) == ["KEEP", "REJECT"] * 3
    assert list(labels["Source_Row_Index"]) == list(range(6))


def test_scorer_accepts_maybe_for_negative_but_not_keep(tmp_path):
    gold = tmp_path / "gold.csv"
    screened = tmp_path / "screened.csv"
    pd.DataFrame([
        {"Source_Row_Index": 0, "Gold_Decision": "KEEP", "Title": "Relevant"},
        {"Source_Row_Index": 1, "Gold_Decision": "REJECT", "Title": "Negative A"},
        {"Source_Row_Index": 2, "Gold_Decision": "REJECT", "Title": "Negative B"},
    ]).to_csv(gold, index=False)
    pd.DataFrame([
        _screened_row(0, "Relevant", "KEEP"),
        _screened_row(1, "Negative A", "REJECT", escalated=True),
        _screened_row(2, "Negative B", "MAYBE"),
    ]).to_csv(screened, index=False)

    report = score_mixed_control(screened, gold, runtime_seconds=12.5, retry_count=1)

    assert report["passed"]
    assert report["confusion"]["REJECT"] == {"KEEP": 0, "MAYBE": 1, "REJECT": 1}
    assert report["critic_count"] == 1
    assert report["retry_count"] == 1
    assert report["runtime_seconds"] == 12.5


def test_scorer_reports_false_keep_with_evidence_and_invalid_rows(tmp_path):
    gold = tmp_path / "gold.csv"
    screened = tmp_path / "screened.csv"
    pd.DataFrame([
        {"Source_Row_Index": 0, "Gold_Decision": "KEEP", "Title": "Relevant"},
        {"Source_Row_Index": 1, "Gold_Decision": "REJECT", "Title": "Negative"},
    ]).to_csv(gold, index=False)
    pd.DataFrame([
        _screened_row(0, "Relevant", "KEEP"),
        _screened_row(1, "Negative", "KEEP", valid=False),
    ]).to_csv(screened, index=False)

    report = score_mixed_control(screened, gold)

    assert not report["passed"]
    assert report["false_keep_count"] == 1
    assert report["false_keeps"][0]["evidence"][0]["quote"] == "Evidence."
    assert report["invalid_structured_count"] == 1
    assert not report["gates"]["no_clear_negative_kept"]


def test_scorer_rejects_unverified_routed_definitive_decision(tmp_path):
    gold = tmp_path / "gold.csv"
    screened = tmp_path / "screened.csv"
    pd.DataFrame([
        {"Source_Row_Index": 0, "Gold_Decision": "REJECT", "Title": "Negative"},
    ]).to_csv(gold, index=False)
    row = _screened_row(0, "Negative", "REJECT")
    row["Critic_Route"] = "inclusion_only_reject"
    row["Verification_Status"] = "pending"
    pd.DataFrame([row]).to_csv(screened, index=False)

    report = score_mixed_control(screened, gold)

    assert not report["passed"]
    assert report["unverified_definitive_count"] == 1
    assert not report["gates"]["all_definitive_decisions_verified"]


def test_scorer_reads_runtime_and_retry_counts_from_diagnostics_summary(tmp_path):
    gold = tmp_path / "gold.csv"
    screened = tmp_path / "screened.csv"
    diagnostics = tmp_path / "run.summary.json"
    pd.DataFrame([{"Source_Row_Index": 0, "Gold_Decision": "KEEP", "Title": "Relevant"}]).to_csv(gold, index=False)
    pd.DataFrame([_screened_row(0, "Relevant", "KEEP")]).to_csv(screened, index=False)
    diagnostics.write_text(json.dumps({
        "runtime_seconds": 9.25, "retry_count": 2, "timeout_fallback_count": 1,
        "attempt_count": 4, "detector_outcomes": {"completed": 3, "timeout": 1},
        "critic_route_counts": {"inclusion_only_reject": 2},
        "verification_outcomes": {"agreed": 2}, "verified_reject_count": 2,
        "verification_fallback_count": 0, "protocol_cache_version": "gemini-web-protocol-v1",
        "clean_chat_rotations": 1,
    }), encoding="utf-8")

    report = score_mixed_control(screened, gold, diagnostics_path=diagnostics)

    assert report["runtime_seconds"] == 9.25
    assert report["retry_count"] == 2
    assert report["timeout_fallback_count"] == 1
    assert report["attempt_count"] == 4
    assert report["detector_outcomes"] == {"completed": 3, "timeout": 1}
    assert report["critic_route_counts"] == {"inclusion_only_reject": 2}
    assert report["verified_reject_count"] == 2
    assert report["protocol_cache_version"] == "gemini-web-protocol-v1"


def test_scorer_counts_timeout_fallback_rows_separately_from_batches(tmp_path):
    gold = tmp_path / "gold.csv"
    screened = tmp_path / "screened.csv"
    diagnostics = tmp_path / "run.summary.json"
    pd.DataFrame([
        {"Source_Row_Index": index, "Gold_Decision": "MAYBE", "Title": f"Paper {index}"}
        for index in range(5)
    ]).to_csv(gold, index=False)
    rows = [_screened_row(index, f"Paper {index}", "MAYBE") for index in range(5)]
    for row in rows:
        row["Failure_Class"] = "transport_timeout"
    pd.DataFrame(rows).to_csv(screened, index=False)
    diagnostics.write_text(json.dumps({"timeout_fallback_count": 1}))

    report = score_mixed_control(screened, gold, diagnostics_path=diagnostics)

    assert report["timeout_fallback_count"] == 1
    assert report["timeout_fallback_row_count"] == 5


def test_scorer_supports_gold_maybe_and_near_miss_resolution(tmp_path):
    gold = tmp_path / "gold.csv"
    screened = tmp_path / "screened.csv"
    pd.DataFrame([
        {"Source_Row_Index": 0, "Gold_Decision": "MAYBE", "Control_Group": "ambiguous"},
        {"Source_Row_Index": 1, "Gold_Decision": "REJECT", "Control_Group": "hard_near_miss"},
    ]).to_csv(gold, index=False)
    pd.DataFrame([
        _screened_row(0, "Ambiguous", "MAYBE"),
        _screened_row(1, "Near miss", "MAYBE"),
    ]).to_csv(screened, index=False)

    report = score_mixed_control(screened, gold)

    assert report["passed"]
    assert report["confusion"]["MAYBE"]["MAYBE"] == 1
    assert report["unresolved_near_miss_count"] == 1
    assert report["control_groups"]["hard_near_miss"]["predictions"]["MAYBE"] == 1


def test_repeatability_reports_transitions_and_direct_contradictions(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    gold = tmp_path / "gold.csv"
    pd.DataFrame([
        _screened_row(0, "Stable", "MAYBE"),
        _screened_row(1, "Contradiction", "KEEP"),
        _screened_row(2, "Soft disagreement", "KEEP"),
    ]).to_csv(first, index=False)
    pd.DataFrame([
        _screened_row(0, "Stable", "MAYBE"),
        _screened_row(1, "Contradiction", "REJECT"),
        _screened_row(2, "Soft disagreement", "MAYBE"),
    ]).to_csv(second, index=False)
    pd.DataFrame([
        {"Source_Row_Index": 0, "Gold_Decision": "REJECT", "Control_Group": "hard_near_miss"},
        {"Source_Row_Index": 1, "Gold_Decision": "KEEP", "Control_Group": "hard_relevant"},
        {"Source_Row_Index": 2, "Gold_Decision": "KEEP", "Control_Group": "hard_relevant"},
    ]).to_csv(gold, index=False)

    report = compare_screening_runs(first, second, gold_path=gold)

    assert report["exact_agreement_rate"] == 0.3333
    assert report["transition_matrix"]["KEEP"]["REJECT"] == 1
    assert report["keep_reject_contradiction_count"] == 1
    assert report["safety_contradiction_count"] == 1
    assert report["semantic_exact_agreement_rate"] == 0.3333
    assert report["repeated_maybe_count"] == 1
    assert not report["passed"]
