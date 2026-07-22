import json
from pathlib import Path

import pandas as pd
import pytest

from evaluation.gemini_web_v2_validation import (
    APPROVED_DIAGNOSTIC_FIELDS,
    prepare_validation_suite,
    run_validation_suite,
    validate_diagnostics,
)


def _screened_rows(papers_path):
    papers = pd.read_csv(papers_path)
    rows = []
    for index, row in papers.iterrows():
        rows.append({
            **row.to_dict(),
            "Source_Row_Index": index,
            "Decision": "MAYBE",
            "Reason": "Insufficient evidence.",
            "Confidence": 0.5,
            "Validation_Status": "validated",
            "Escalated": False,
            "Processing_Seconds": 0.1,
            "Criteria_JSON": "[]",
            "Evidence_JSON": "[]",
            "Layer_Trace_JSON": "[]",
            "Uncertainty_JSON": "[]",
            "Contradictions_JSON": "[]",
            "Validation_Errors": "[]",
        })
    return rows


def test_prepared_fixtures_are_blinded_and_have_separate_gold(tmp_path):
    prepared = prepare_validation_suite(tmp_path / "suite")

    assert prepared["total_papers"] == 220
    for case in prepared["cases"]:
        papers = pd.read_csv(case["papers"])
        gold = pd.read_csv(case["gold"])
        assert list(papers.columns) == ["Title", "Abstract"]
        assert "Gold_Decision" not in papers.columns
        assert len(papers) == len(gold)
    digital_gold = pd.read_csv(prepared["cases"][3]["gold"])
    assert set(digital_gold["Gold_Decision"]) == {"KEEP", "MAYBE", "REJECT"}
    hard_gold = pd.read_csv(prepared["cases"][1]["gold"])
    assert dict(hard_gold["Gold_Decision"].value_counts()) == {
        "KEEP": 20, "REJECT": 18, "MAYBE": 2,
    }


def test_diagnostics_reject_content_fields(tmp_path):
    safe = tmp_path / "safe.jsonl"
    safe.write_text(json.dumps({field: "" for field in APPROVED_DIAGNOSTIC_FIELDS}) + "\n")
    assert validate_diagnostics(safe)["approved_fields_only"]
    unsafe = tmp_path / "unsafe.jsonl"
    unsafe.write_text(json.dumps({"event": "attempt", "prompt": "private"}) + "\n")
    with pytest.raises(ValueError, match="unsafe Gemini Web diagnostic fields"):
        validate_diagnostics(unsafe)


def test_suite_runner_forces_fresh_unique_runs(tmp_path, monkeypatch):
    prepared = prepare_validation_suite(tmp_path / "suite")
    calls = []
    monkeypatch.chdir(tmp_path)

    def fake_screen(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(_screened_rows(kwargs["csv_path"])).to_csv(output, index=False)
        diagnostics = tmp_path / f"{kwargs['progress_job_id']}.jsonl"
        diagnostics.write_text(json.dumps({
            "event": "gemini_web_attempt", "submission_number": 1,
            "stage": "gemini_web_primary", "retry_number": 0,
            "outcome": "completed", "recovery_action": "",
            "attempt_duration_ms": 10, "response_selector": "model-response",
            "response_container_count": 1, "response_state": "json_complete",
            "generation_detected": True, "timeout_stage": "", "fallback_reason": "",
        }) + "\n")
        diagnostics.with_suffix(".summary.json").write_text(json.dumps({
            "runtime_seconds": 1.0, "retry_count": 0,
            "timeout_fallback_count": 0, "attempt_count": 1,
            "detector_outcomes": {"completed": 1},
        }))
        return {
            "resumed_count": 0, "diagnostics_path": str(diagnostics),
            "runtime_seconds": 1.0, "retry_count": 0,
            "timeout_fallback_count": 0,
        }

    report = run_validation_suite(prepared["manifest_path"], screen=fake_screen)

    assert len(calls) == 4
    assert all(call["resume"] is False for call in calls)
    assert len({call["progress_job_id"] for call in calls}) == 4
    assert len({call["output_path"] for call in calls}) == 4
    assert report["repeatability"]["exact_agreement_rate"] == 1.0
    assert report["production_change_supported"] is False


def test_suite_runner_refuses_raw_capture(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": []}))
    monkeypatch.setenv("GEMINI_WEB_CAPTURE_RAW_DEBUG", "true")
    with pytest.raises(RuntimeError, match="RAW_DEBUG"):
        run_validation_suite(manifest, screen=lambda **kwargs: {})
