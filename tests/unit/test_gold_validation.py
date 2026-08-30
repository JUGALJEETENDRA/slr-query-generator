from __future__ import annotations

import json

import pandas as pd
import pytest

from litsync_app.validation.gold import (
    create_blinded_sample,
    evaluate_completed_labels,
    screening_content_fingerprint,
)


JOB_ID = "gold-job-1"
EXPECTED_COLUMNS = [
    "Screening_Job_ID",
    "Validation_Set_ID",
    "Source_Row_Index",
    "Research_Question",
    "Title",
    "Abstract",
    "Year",
    "DOI",
    "Gold_Decision",
    "Reviewer_Notes",
]
UNBOUND_MESSAGE = (
    "This validation CSV is not bound to a screening job. "
    "Regenerate it from the intended completed screening run."
)


def _rows(keep=77, reject=16, maybe=7):
    rows = []
    source = 271
    for decision, count in (("KEEP", keep), ("REJECT", reject), ("MAYBE", maybe)):
        for _ in range(count):
            rows.append({
                "Source_Row_Index": source,
                "Protocol_ID": "protocol-v21",
                "Prompt_Version": "gemini-web-v2.4-assessment-prompt-v5",
                "Decision": decision,
                "Title": f"Blinded paper {source}",
                "Abstract": f"Abstract evidence for paper {source}.",
                "Year": 2026,
                "DOI": f"10.test/{source}",
                "Validation_Status": "unresolved" if decision == "MAYBE" and source == 364 else "validated",
                "Escalated": decision == "MAYBE" and source == 364,
                "Evidence_JSON": json.dumps([
                    {"source": "abstract", "quote": f"Abstract evidence for paper {source}."}
                ]),
            })
            source += 1
    return rows


def test_current_pilot_is_reproducible_and_blinded(tmp_path):
    first = create_blinded_sample(_rows(), "How is a system used?", tmp_path, JOB_ID)
    second = create_blinded_sample(_rows(), "How is a system used?", tmp_path, JOB_ID)
    assert first["validation_set_id"] == second["validation_set_id"]
    assert first["sample_counts"] == {"KEEP": 37, "REJECT": 16, "MAYBE": 7}
    labels = pd.read_csv(first["label_path"], dtype=str, keep_default_na=False)
    assert len(labels) == 60
    assert list(labels.columns) == EXPECTED_COLUMNS
    assert set(labels["Screening_Job_ID"]) == {JOB_ID}
    assert set(labels["Gold_Decision"]) == {""}
    forbidden = {
        "Decision", "Reason", "Confidence", "Evidence_JSON", "Criteria_JSON",
        "Model", "Validation_Status", "Escalated",
    }
    assert not (forbidden & set(labels.columns))
    manifest = json.loads(open(first["manifest_path"], encoding="utf-8").read())
    assert manifest["sample_counts"] == {"KEEP": 37, "REJECT": 16, "MAYBE": 7}
    assert {
        row["immutable_blinded"]["Screening_Job_ID"]
        for row in manifest["rows"]
    } == {JOB_ID}


def test_sample_below_sixty_contains_every_paper(tmp_path):
    result = create_blinded_sample(
        _rows(keep=4, reject=3, maybe=2), "RQ", tmp_path, JOB_ID
    )
    labels = pd.read_csv(result["label_path"])
    assert len(labels) == 9
    assert result["sample_counts"] == {"KEEP": 4, "REJECT": 3, "MAYBE": 2}


def test_sampling_strata_are_diagnostic_configuration_only(tmp_path):
    rows = _rows(keep=10, reject=10, maybe=10)
    original_decisions = [row["Decision"] for row in rows]
    result = create_blinded_sample(
        rows, "RQ", tmp_path, JOB_ID, sample_size=10,
        sampling_strata={"KEEP": 0.2, "REJECT": 0.3, "MAYBE": 0.5},
    )
    manifest = json.loads(open(result["manifest_path"], encoding="utf-8").read())
    assert result["sample_counts"] == {"KEEP": 2, "REJECT": 3, "MAYBE": 5}
    assert manifest["sampling_strata"] == {"KEEP": 0.2, "REJECT": 0.3, "MAYBE": 0.5}
    assert [row["Decision"] for row in rows] == original_decisions


def _completed_sample(tmp_path):
    result = create_blinded_sample(
        _rows(), "How is a system used?", tmp_path, JOB_ID
    )
    labels = pd.read_csv(result["label_path"], dtype=str, keep_default_na=False)
    manifest = json.loads(open(result["manifest_path"], encoding="utf-8").read())
    model_by_id = {row["source_row_index"]: row["model_decision"] for row in manifest["rows"]}
    seen = {"KEEP": 0, "REJECT": 0, "MAYBE": 0}
    for index, row in labels.iterrows():
        model = model_by_id[str(row["Source_Row_Index"])]
        seen[model] += 1
        if model == "KEEP":
            label = "REJECT" if seen[model] <= 4 else "KEEP"
        elif model == "REJECT":
            label = "KEEP" if seen[model] <= 2 else "REJECT"
        else:
            label = "KEEP" if seen[model] <= 3 else "REJECT"
        labels.at[index, "Gold_Decision"] = label
    completed = tmp_path / "completed.csv"
    labels.to_csv(completed, index=False, encoding="utf-8-sig")
    return completed, labels, result


def test_weighted_diagnostic_metrics_and_error_lists(tmp_path):
    completed, _, _ = _completed_sample(tmp_path)
    report = evaluate_completed_labels(completed, tmp_path, JOB_ID, _rows())
    assert report["status"] == "provisional_single_reviewer"
    assert report["resolved_labels"] == 60
    assert report["metrics"]["definitive_keep_precision"] == pytest.approx(33 / 37, abs=0.0001)
    relevant = 33 * (77 / 37) + 2 + 3
    assert report["metrics"]["false_reject_rate"] == pytest.approx(2 / relevant, abs=0.0001)
    assert len(report["false_keeps"]) == 4
    assert len(report["false_rejects"]) == 2
    assert report["full_run_safety"]["exact_evidence_rate"] == 1.0
    assert report["confidence_intervals_95"]["definitive_keep_precision"]["lower"] is not None


def test_unsure_blank_and_missing_rows_are_reported_not_forced(tmp_path):
    completed, labels, _ = _completed_sample(tmp_path)
    labels.at[0, "Gold_Decision"] = "UNSURE"
    labels.at[1, "Gold_Decision"] = ""
    labels = labels.iloc[:-1]
    labels.to_csv(completed, index=False)
    with pytest.raises(ValueError, match="Removed source rows"):
        evaluate_completed_labels(completed, tmp_path, JOB_ID, _rows())


@pytest.mark.parametrize("mutation, message", [
    ("duplicate", "Duplicate"),
    ("altered", "altered"),
    ("invalid_label", "Invalid Gold_Decision"),
    ("unknown_set", "Unknown validation set"),
])
def test_completed_csv_rejects_integrity_errors(tmp_path, mutation, message):
    completed, labels, _ = _completed_sample(tmp_path)
    if mutation == "duplicate":
        labels.at[1, "Source_Row_Index"] = labels.at[0, "Source_Row_Index"]
    elif mutation == "altered":
        labels.at[0, "Title"] = "Changed title"
    elif mutation == "invalid_label":
        labels.at[0, "Gold_Decision"] = "MAYBE"
    else:
        labels["Validation_Set_ID"] = "deadbeefdeadbeef"
    labels.to_csv(completed, index=False)
    with pytest.raises(ValueError, match=message):
        evaluate_completed_labels(completed, tmp_path, JOB_ID, _rows())


def test_phase3a_content_fingerprint_excludes_job_but_tracks_decisions(tmp_path):
    rows = _rows()
    original = screening_content_fingerprint(rows)
    changed = [dict(row) for row in rows]
    changed[0]["Decision"] = "REJECT"
    assert screening_content_fingerprint(changed) != original

    first = create_blinded_sample(rows, "RQ", tmp_path, "job-a")
    second = create_blinded_sample(rows, "RQ", tmp_path, "job-b")
    changed_set = create_blinded_sample(changed, "RQ", tmp_path, "job-a")
    assert first["screening_content_fingerprint"] == second["screening_content_fingerprint"]
    assert first["validation_set_id"] != second["validation_set_id"]
    assert first["validation_set_id"] != changed_set["validation_set_id"]


def test_phase3a_old_validation_file_against_new_job_fails_closed(tmp_path):
    completed, _, result = _completed_sample(tmp_path)
    expected = (
        f"This validation CSV belongs to screening job '{JOB_ID}', "
        "not 'new-job'."
    )
    with pytest.raises(ValueError) as exc_info:
        evaluate_completed_labels(completed, tmp_path, "new-job", _rows())
    assert str(exc_info.value) == expected
    report_path = (
        tmp_path / "gold_validation"
        / f"{result['validation_set_id']}_report.json"
    )
    assert not report_path.exists()


@pytest.mark.parametrize("mutation", [
    "missing_column",
    "blank_job_id",
    "mixed_job_ids",
    "altered_one_row",
])
def test_phase3a_row_level_job_binding_failures_are_unbound(
    tmp_path, mutation,
):
    completed, labels, result = _completed_sample(tmp_path)
    if mutation == "missing_column":
        labels = labels.drop(columns=["Screening_Job_ID"])
    elif mutation == "blank_job_id":
        labels.at[0, "Screening_Job_ID"] = ""
    elif mutation == "mixed_job_ids":
        labels.loc[labels.index[::2], "Screening_Job_ID"] = "other-job"
    else:
        labels.at[0, "Screening_Job_ID"] = "altered-job"
    labels.to_csv(completed, index=False)

    report_path = (
        tmp_path / "gold_validation"
        / f"{result['validation_set_id']}_report.json"
    )
    with pytest.raises(ValueError) as exc_info:
        evaluate_completed_labels(completed, tmp_path, JOB_ID, _rows())
    assert str(exc_info.value) == UNBOUND_MESSAGE
    assert not report_path.exists()


def test_phase3a_binding_failure_does_not_return_or_overwrite_existing_report(
    tmp_path,
):
    completed, labels, result = _completed_sample(tmp_path)
    labels = labels.drop(columns=["Screening_Job_ID"])
    labels.to_csv(completed, index=False)
    report_path = (
        tmp_path / "gold_validation"
        / f"{result['validation_set_id']}_report.json"
    )
    report_path.write_text('{"sentinel": true}', encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        evaluate_completed_labels(completed, tmp_path, JOB_ID, _rows())
    assert str(exc_info.value) == UNBOUND_MESSAGE
    assert report_path.read_text(encoding="utf-8") == '{"sentinel": true}'


def test_phase3a_changed_prompt_or_protocol_fails_closed(tmp_path):
    completed, _, _ = _completed_sample(tmp_path)
    changed = [dict(row) for row in _rows()]
    changed[0]["Prompt_Version"] = "different-prompt"
    with pytest.raises(ValueError, match="protocol or assessment prompt version"):
        evaluate_completed_labels(completed, tmp_path, JOB_ID, changed)

    changed = [dict(row) for row in _rows()]
    changed[0]["Protocol_ID"] = "different-protocol"
    with pytest.raises(ValueError, match="protocol or assessment prompt version"):
        evaluate_completed_labels(completed, tmp_path, JOB_ID, changed)


def test_phase3a_legacy_manifest_requires_regeneration(tmp_path):
    completed, labels, result = _completed_sample(tmp_path)
    manifest_path = tmp_path / "gold_validation" / (
        result["validation_set_id"] + "_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in (
        "binding_version", "job_id", "assessment_prompt_version",
        "screening_content_fingerprint",
    ):
        manifest.pop(key, None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    labels.to_csv(completed, index=False)
    with pytest.raises(ValueError, match="Regenerate it"):
        evaluate_completed_labels(completed, tmp_path, JOB_ID, _rows())
