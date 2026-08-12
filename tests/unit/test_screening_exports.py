from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import litsync_app.app as server
from litsync_app.screening.bulk import SCREENING_SESSION
from litsync_app.screening.exports import (
    ScreeningExportError,
    generate_screening_exports,
)


client = TestClient(server.app)


def _write_job(
    root: Path,
    job_id: str = "export-job",
    *,
    architecture_version: str = "architecture-v1",
    screening_engine: str = "",
) -> list[dict[str, str]]:
    rows = [
        {
            "Title": "Unicode α paper",
            "Abstract": 'Line one, with comma\nLine two "quoted".',
            "Authors": "A. Author",
            "Article_DOI": " https://doi.org/10.1000/ABC ",
            "Record URL": "https://example.test/one",
            "Arbitrary Extra": "kept",
            "Decision": "KEEP",
            "Confidence": "0.91",
            "Reason": "Included reason",
            "Evidence_Quote": "Unicode α paper",
            "Route_Used": "primary_only",
            "Validation_Status": "validated",
            "Failure_Class": "",
            "Primary_Decision": "KEEP",
            "Primary_Confidence": "0.91",
            "Verifier_Decision": "",
            "Verifier_Confidence": "",
            "Agreement_Status": "primary_only",
            "Prompt_Version": "prompt-v1",
            "Architecture_Version": architecture_version,
            "Protocol_ID": "protocol-1",
            "Review_Protocol_ID": "protocol-1",
            "Source_Row_Index": "1",
            "Execution_Origin": "fresh_primary",
        },
        {
            "Title": "Maybe paper",
            "Abstract": "Maybe abstract",
            "Authors": "B. Author",
            "Article_DOI": "doi:",
            "Record URL": "N/A",
            "Arbitrary Extra": "also kept",
            "Decision": "MAYBE",
            "Confidence": "0",
            "Reason": "Independent validation was unavailable.",
            "Evidence_Quote": "",
            "Route_Used": "safe_fallback",
            "Validation_Status": "safe_fallback",
            "Failure_Class": "browser_or_transport_failure",
            "Primary_Decision": "REJECT",
            "Primary_Confidence": "0.8",
            "Verifier_Decision": "",
            "Verifier_Confidence": "",
            "Agreement_Status": "verification_unavailable",
            "Prompt_Version": "prompt-v1",
            "Architecture_Version": architecture_version,
            "Protocol_ID": "protocol-1",
            "Review_Protocol_ID": "protocol-1",
            "Source_Row_Index": "2",
            "Execution_Origin": "technical_fallback",
        },
        {
            "Title": "Rejected paper",
            "Abstract": "Rejected abstract",
            "Authors": "C. Author",
            "Article_DOI": "http://dx.doi.org/10.2000/xyz",
            "Record URL": "",
            "Arbitrary Extra": "third",
            "Decision": "REJECT",
            "Confidence": "0.88",
            "Reason": "Excluded reason",
            "Evidence_Quote": "Rejected abstract",
            "Route_Used": "blind_verification",
            "Validation_Status": "validated",
            "Failure_Class": "",
            "Primary_Decision": "REJECT",
            "Primary_Confidence": "0.7",
            "Verifier_Decision": "REJECT",
            "Verifier_Confidence": "0.88",
            "Agreement_Status": "agree",
            "Prompt_Version": "prompt-v1",
            "Architecture_Version": architecture_version,
            "Protocol_ID": "protocol-1",
            "Review_Protocol_ID": "protocol-1",
            "Source_Row_Index": "3",
            "Execution_Origin": "fresh_verification",
        },
    ]
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        runs / f"screened-{job_id}.csv", index=False, encoding="utf-8-sig"
    )
    prisma = {
        "job_id": job_id,
        "screening_engine": screening_engine,
        "protocol_inputs": {
            "research_question": "Exact question?",
            "research_context": "Exact context",
            "inclusion_criteria": "Include exact criterion",
            "exclusion_criteria": "Exclude exact criterion",
        },
    }
    (root / "prisma").mkdir(exist_ok=True)
    (root / "prisma" / f"{job_id}.json").write_text(
        json.dumps(prisma), encoding="utf-8"
    )
    (root / "latest_screening.json").write_text(json.dumps({
        "job_id": job_id,
        "summary": {
            "runtime_seconds": 12.5,
            "primary_batch_size": 10,
            "primary_batches_submitted": 1,
            "primary_batches_completed": 1,
            "verification_batches_submitted": 1,
            "verification_batches_completed": 1,
            "primary_papers_requested": 3,
            "verification_papers_requested": 2,
            "retry_count": 0,
            "browser_context_started": 1,
            "pages_opened": 3,
            "pages_closed": 3,
            "peak_simultaneous_tabs": 2,
        },
    }), encoding="utf-8")
    return rows


def _read(root: Path, job_id: str, name: str) -> pd.DataFrame:
    return pd.read_csv(
        root / "exports" / job_id / name,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )


def test_export_pack_partitions_and_preserves_original_metadata(tmp_path):
    job_id = "export-job"
    _write_job(tmp_path, job_id)

    result = generate_screening_exports(tmp_path, job_id)

    assert result["counts"] == {
        "all": 3, "keep": 1, "maybe": 1, "reject": 1, "review_queue": 1,
    }
    export_dir = tmp_path / "exports" / job_id
    assert {path.name for path in export_dir.iterdir()} == {
        "screened_all.csv", "screened_keep.csv", "screened_maybe.csv",
        "screened_reject.csv", "screened_maybe_review_queue.csv",
        "screening_summary.json",
    }
    all_rows = _read(tmp_path, job_id, "screened_all.csv")
    assert list(all_rows.columns[:6]) == [
        "Title", "Abstract", "Authors", "Article_DOI", "Record URL", "Arbitrary Extra",
    ]
    assert all_rows.loc[0, "Abstract"] == 'Line one, with comma\nLine two "quoted".'
    assert all_rows["Source_Row_Index"].is_unique
    assert all_rows["Job_ID"].tolist() == [job_id] * 3
    assert set(_read(tmp_path, job_id, "screened_keep.csv")["Decision"]) == {"KEEP"}
    assert set(_read(tmp_path, job_id, "screened_maybe.csv")["Decision"]) == {"MAYBE"}
    assert set(_read(tmp_path, job_id, "screened_reject.csv")["Decision"]) == {"REJECT"}


def test_doi_url_normalization_is_non_destructive_and_does_not_invent(tmp_path):
    _write_job(tmp_path)
    generate_screening_exports(tmp_path, "export-job")
    frame = _read(tmp_path, "export-job", "screened_all.csv")

    assert frame["Article_DOI"].tolist() == [
        " https://doi.org/10.1000/ABC ", "doi:", "http://dx.doi.org/10.2000/xyz",
    ]
    assert frame["Canonical_DOI"].tolist() == ["10.1000/ABC", "", "10.2000/xyz"]
    assert frame["Canonical_Source_URL"].tolist() == [
        "https://example.test/one", "", "https://doi.org/10.2000/xyz",
    ]


def test_review_queue_is_deterministic_and_does_not_change_decisions(tmp_path):
    rows = _write_job(tmp_path)
    extra = dict(rows[1])
    extra.update({
        "Title": "Contradiction", "Source_Row_Index": "4",
        "Failure_Class": "validation_contradiction",
        "Primary_Decision": "KEEP", "Verifier_Decision": "REJECT",
        "Agreement_Status": "disagreement",
    })
    path = tmp_path / "runs" / "screened-export-job.csv"
    pd.DataFrame(rows + [extra]).to_csv(path, index=False, encoding="utf-8-sig")

    generate_screening_exports(tmp_path, "export-job")
    queue = _read(tmp_path, "export-job", "screened_maybe_review_queue.csv")

    assert queue["Source_Row_Index"].tolist() == ["4", "2"]
    assert queue["Review_Priority"].tolist() == ["1", "2"]
    assert queue["Decision"].tolist() == ["MAYBE", "MAYBE"]
    assert queue["Review_Reason"].str.strip().ne("").all()


@pytest.mark.parametrize("decision", ["KEEP", "REJECT"])
@pytest.mark.parametrize(
    ("column", "value"),
    [("Validation_Status", "safe_fallback"), ("Execution_Origin", "technical_fallback")],
)
def test_unsafe_definitive_fallback_blocks_export(tmp_path, decision, column, value):
    rows = _write_job(tmp_path)
    rows[0]["Decision"] = decision
    rows[0][column] = value
    pd.DataFrame(rows).to_csv(
        tmp_path / "runs" / "screened-export-job.csv", index=False, encoding="utf-8-sig"
    )
    with pytest.raises(ScreeningExportError, match="Unsafe definitive fallback"):
        generate_screening_exports(tmp_path, "export-job")


def test_api_exports_exact_persisted_job_after_memory_is_cleared(monkeypatch, tmp_path):
    _write_job(tmp_path, "persisted-job")
    _write_job(tmp_path, "other-job")
    SCREENING_SESSION.begin("cleared-session")
    monkeypatch.setattr(server, "OUTPUT_DIR", str(tmp_path))

    with TestClient(server.app) as restarted_client:
        response = restarted_client.post("/screening-jobs/persisted-job/exports")

        assert response.status_code == 200
        payload = response.json()
        assert payload["job_id"] == "persisted-job"
        assert payload["counts"]["all"] == 3
        assert set(payload["downloads"]) == {
            "all", "keep", "maybe", "reject", "review_queue", "summary",
        }
        downloaded = restarted_client.get(payload["downloads"]["all"])
        assert downloaded.status_code == 200
        assert "persisted-job" in downloaded.text
        assert restarted_client.post("/screening-jobs/missing-job/exports").status_code == 404
        assert restarted_client.get("/screening-jobs/persisted-job/exports/../summary").status_code in {404, 405}
        assert restarted_client.get("/screening-jobs/persisted-job/exports/not-supported").status_code == 404


@pytest.mark.parametrize(
    ("manual_decision", "expected_counts"),
    [
        ("KEEP", {"all": 3, "keep": 2, "maybe": 0, "reject": 1, "review_queue": 0}),
        ("REJECT", {"all": 3, "keep": 1, "maybe": 0, "reject": 2, "review_queue": 0}),
    ],
)
def test_manual_review_regenerates_latest_partitions_and_preserves_history(
    monkeypatch, tmp_path, manual_decision, expected_counts,
):
    rows = _write_job(tmp_path, "manual-job")
    output = tmp_path / "runs" / "screened-manual-job.csv"
    monkeypatch.setattr(server, "OUTPUT_DIR", str(tmp_path))
    SCREENING_SESSION.begin("manual-job", str(output), "architecture-v1")
    SCREENING_SESSION.set_results(
        rows, job_id="manual-job", output_path=str(output), architecture_version="architecture-v1"
    )
    assert client.post("/screening-jobs/manual-job/exports").json()["counts"]["maybe"] == 1

    updated = client.patch(
        "/screening_jobs/manual-job/records/2",
        json={"decision": manual_decision, "exclusion_reason": "Reviewer note"},
    )
    assert updated.status_code == 200
    paper = updated.json()["paper"]
    assert paper["Original_Model_Decision"] == "REJECT"
    assert paper["Manual_Decision"] == manual_decision
    assert paper["Final_Decision_Source"] == "human_review"

    refreshed = client.post("/screening-jobs/manual-job/exports").json()
    assert refreshed["counts"] == expected_counts
    all_rows = _read(tmp_path, "manual-job", "screened_all.csv")
    reviewed = all_rows.loc[all_rows["Source_Row_Index"] == "2"].iloc[0]
    assert reviewed["Final_Decision_Source"] == "human_review"
    assert reviewed["Manual_Review_Status"] == "reviewed"


def test_summary_has_protocol_counts_relative_path_and_no_sensitive_columns(tmp_path):
    rows = _write_job(tmp_path)
    rows[0]["API Key"] = "secret-value"
    pd.DataFrame(rows).to_csv(
        tmp_path / "runs" / "screened-export-job.csv", index=False, encoding="utf-8-sig"
    )
    generate_screening_exports(tmp_path, "export-job")
    summary_text = (tmp_path / "exports" / "export-job" / "screening_summary.json").read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    all_text = (tmp_path / "exports" / "export-job" / "screened_all.csv").read_text(encoding="utf-8-sig")

    assert summary["protocol_id"] == summary["review_protocol_id"] == "protocol-1"
    assert summary["total_count"] == 3
    assert summary["export_files"]["maybe"]["row_count"] == 1
    assert summary["source_output_path"] == "runs/screened-export-job.csv"
    assert summary["generated_from_latest_persisted_decisions"] is True
    assert "secret-value" not in summary_text
    assert "secret-value" not in all_text
    assert "API Key" not in all_text


def test_generation_is_idempotent_and_does_not_mutate_source(tmp_path):
    _write_job(tmp_path)
    source = tmp_path / "runs" / "screened-export-job.csv"
    before = source.read_bytes()
    first = generate_screening_exports(tmp_path, "export-job")
    second = generate_screening_exports(tmp_path, "export-job")
    assert first["counts"] == second["counts"]
    assert source.read_bytes() == before
    assert len(_read(tmp_path, "export-job", "screened_all.csv")) == 3
