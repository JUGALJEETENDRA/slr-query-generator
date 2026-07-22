import json

import pandas as pd

from evaluation.gemini_web_mixed_control import build_mixed_control, score_mixed_control


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


def test_scorer_reads_runtime_and_retry_counts_from_diagnostics_summary(tmp_path):
    gold = tmp_path / "gold.csv"
    screened = tmp_path / "screened.csv"
    diagnostics = tmp_path / "run.summary.json"
    pd.DataFrame([{"Source_Row_Index": 0, "Gold_Decision": "KEEP", "Title": "Relevant"}]).to_csv(gold, index=False)
    pd.DataFrame([_screened_row(0, "Relevant", "KEEP")]).to_csv(screened, index=False)
    diagnostics.write_text(json.dumps({
        "runtime_seconds": 9.25, "retry_count": 2, "timeout_fallback_count": 1,
        "attempt_count": 4, "detector_outcomes": {"completed": 3, "timeout": 1},
    }), encoding="utf-8")

    report = score_mixed_control(screened, gold, diagnostics_path=diagnostics)

    assert report["runtime_seconds"] == 9.25
    assert report["retry_count"] == 2
    assert report["timeout_fallback_count"] == 1
    assert report["attempt_count"] == 4
    assert report["detector_outcomes"] == {"completed": 3, "timeout": 1}
