from __future__ import annotations

import json
import math

import pandas as pd

from fastapi.testclient import TestClient

import server
from bulk_screen import PROGRESS, SCREENING_SESSION, ScreeningProgress
from local_ai.hardware import HardwareSnapshot, RuntimeProfile


client = TestClient(server.app)


def _profile():
    hardware = HardwareSnapshot(
        total_ram_gb=16.0,
        available_ram_gb=9.0,
        cpu_cores=8,
        platform="Test",
        gpu_name="Test GPU",
        gpu_vram_gb=4.0,
        installed_models={
            "qwen2.5:3b": 1,
            "qwen3:4b-instruct-2507-q4_K_M": 1,
        },
    )
    return RuntimeProfile(
        requested_tier="auto",
        resolved_tier="balanced",
        resource_profile="balanced",
        fast_model="qwen3:8b",
        strong_model="qwen3:8b",
        num_ctx=4096,
        keep_alive="5m",
        concurrency=1,
        memory_reserve_ratio=0.2,
        downgrade_reasons=(),
        hardware=hardware,
        calibration={},
    )


def test_existing_website_is_the_homepage():
    response = client.get("/")
    assert response.status_code == 200
    assert "SLR Query Generator" in response.text
    assert "Query Generator" in response.text
    assert "LitSync" in response.text
    assert "CSV Screener" in response.text
    assert "Automatic local AI screening" in response.text


def test_website_hides_legacy_model_controls_and_uses_relative_local_api():
    html = client.get("/").text
    assert 'id="model"' not in html
    assert 'name="mode"' not in html
    assert "twoStageEnabled" not in html
    assert 'fd.append("model_tier", "auto")' in html
    assert 'fd.append("resource_profile", "balanced")' in html
    assert 'fetch("/screen_csv"' in html
    assert "http://localhost:8000" not in html
    assert "stronger local model" not in html
    assert "Strong-model checks" not in html
    assert "Validation repairs" in html
    assert "function pipelinePercent(progress = {})" in html
    assert "Automatic batch retries" in html
    assert "Layer 3: adversarial 4B edge critic" in html
    assert "independent Phi edge critic" not in html


def test_quick_test_is_checked_and_submits_existing_max_rows_field():
    html = client.get("/").text
    assert 'id="quickTest100" checked' in html
    assert "Quick test: screen first 100 rows only" in html
    assert 'fd.append("max_rows", "100")' in html
    assert 'document.getElementById("quickTest100").disabled = disabled' in html


def test_research_context_is_optional_and_sent_only_to_local_screening():
    html = client.get("/").text
    assert 'id="researchContext"' in html
    assert 'fd.append("research_context", researchContext)' in html
    assert 'document.getElementById("researchContext").disabled = disabled' in html


def test_new_screening_job_cannot_be_repainted_by_restore_race():
    html = client.get("/").text
    assert "let screeningGeneration = 0" in html
    assert "screeningGeneration += 1" in html
    assert "generation !== screeningGeneration" in html
    assert "activeScreeningJobId = started.job_id" in html
    assert "/progress?job_id=${encodeURIComponent(jobId)}" in html
    assert "data.job_id !== jobId" in html


def test_existing_website_contains_minimal_excel_gold_validation_workflow():
    html = client.get("/").text
    assert "Gold Validation" in html
    assert "Download 60-paper validation CSV" in html
    assert 'id="goldLabelFile"' in html
    assert "fetch('/gold_validation/sample'" in html
    assert "fetch('/gold_validation/evaluate'" in html


def test_status_stays_lightweight_without_ui_presets(monkeypatch):
    monkeypatch.setattr(server, "resolve_runtime_profile", _profile)
    payload = client.get("/status").json()
    assert payload["backend_ready"] is True
    assert payload["ollama_ready"] is True
    assert payload["resolved"]["tier"] == "balanced"
    assert payload["resolved"]["triage_model"] == "qwen2.5:3b"
    assert payload["resolved"]["deep_model"] == "qwen3:4b-instruct-2507-q4_K_M"
    assert payload["missing_models"] == []
    assert "presets" not in payload


def test_results_are_json_safe_and_have_browser_download_url():
    SCREENING_SESSION.begin(
        "json-job", "outputs/runs/screened-json-job.csv", "local-semantic-boundary-v3.12"
    )
    SCREENING_SESSION.set_results([{
        "Title": "Paper",
        "Abstract": "Evidence",
        "Decision": "MAYBE",
        "Confidence": float("nan"),
        "Source_Row_Index": 1,
    }], job_id="json-job")
    payload = client.get("/screening_results?job_id=json-job").json()
    assert payload["papers"][0]["Confidence"] is None
    assert payload["download_url"] == "/outputs/runs/screened-json-job.csv"


def test_progress_and_results_reject_mismatched_job_ids():
    job_id = "current-isolated-job"
    assert PROGRESS.start_job(job_id) is True
    PROGRESS.begin_screening(job_id, 3, "local-semantic-boundary-v3.12")
    SCREENING_SESSION.begin(
        job_id, f"outputs/runs/screened-{job_id}.csv", "local-semantic-boundary-v3.12"
    )

    assert client.get("/progress?job_id=old-job").status_code == 404
    assert client.get("/screening_results?job_id=old-job").status_code == 404
    assert client.post(
        "/finalize", json={"job_id": "old-job", "papers": []}
    ).status_code == 404
    running = client.get(f"/screening_results?job_id={job_id}").json()
    assert running["status"] == "running"
    assert running["papers"] == []
    PROGRESS.finish(job_id)


def test_json_safe_removes_non_finite_numbers():
    result = server._json_safe({"nan": math.nan, "inf": math.inf, "nested": [1, -math.inf]})
    assert result == {"nan": None, "inf": None, "nested": [1, None]}


def test_csv_defaults_to_automatic_local_ai(monkeypatch):
    monkeypatch.setattr(server.PROGRESS, "start_job", lambda job_id: True)
    started = []

    class NoopThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            started.append(self.kwargs)
            return None

    monkeypatch.setattr(server, "Thread", NoopThread)
    response = client.post(
        "/screen_csv",
        data={
            "question": "Does this paper answer the question?",
            "research_context": "This explains the intended meaning only.",
        },
        files={"file": ("papers.csv", b"Title,Abstract\nPaper,Text", "text/csv")},
    )
    payload = response.json()
    assert payload["status"] == "started"
    assert payload["model_tier"] == "auto"
    assert payload["resource_profile"] == "balanced"
    assert payload["screening_engine"] == "local"
    assert payload["architecture_version"] == "local-semantic-boundary-v3.12"
    assert started[0]["kwargs"]["output_path"].endswith(f"screened-{payload['job_id']}.csv")
    assert len(started[0]["kwargs"]["input_fingerprint"]) == 64
    assert started[0]["kwargs"]["max_rows"] is None
    assert started[0]["kwargs"]["research_context"] == "This explains the intended meaning only."


def test_csv_accepts_first_100_row_limit(monkeypatch):
    monkeypatch.setattr(server.PROGRESS, "start_job", lambda job_id: True)
    started = []

    class NoopThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            started.append(self.kwargs)

    monkeypatch.setattr(server, "Thread", NoopThread)
    response = client.post(
        "/screen_csv",
        data={"question": "RQ", "max_rows": "100"},
        files={"file": ("papers.csv", b"Title,Abstract\nPaper,Text", "text/csv")},
    )
    assert response.status_code == 200
    assert started[0]["kwargs"]["max_rows"] == 100


def test_csv_rejects_another_active_job_friendly(monkeypatch):
    monkeypatch.setattr(server.PROGRESS, "start_job", lambda job_id: False)
    response = client.post(
        "/screen_csv",
        data={"question": "Does this paper answer the question?"},
        files={"file": ("papers.csv", b"Title,Abstract\nPaper,Text", "text/csv")},
    )
    assert response.status_code == 409
    assert "Another screening job" in response.json()["detail"]


def test_finished_progress_runtime_stops_increasing(monkeypatch):
    clock = iter([10.0, 12.5, 99.0])
    monkeypatch.setattr("bulk_screen.time.perf_counter", lambda: next(clock))
    progress = ScreeningProgress()
    assert progress.start_job("job") is True
    progress.begin_screening("job", 1)
    progress.finish("job")
    finished = progress.snapshot()
    assert finished["runtime_seconds"] == 2.5
    assert progress.snapshot()["runtime_seconds"] == 2.5


def test_invalid_output_progress_uses_repair_language():
    progress = ScreeningProgress()
    assert progress.start_job("job") is True
    progress.begin_screening("job", 2)
    progress.begin_stage2("job", 1)
    state = progress.snapshot()
    assert state["phase"] == "validation_repair"
    assert state["stage2_total"] == 1


def test_gold_sample_and_completed_csv_api_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(server, "PRIVATE_DIR", str(tmp_path / "private"))
    SCREENING_SESSION.set_results([
        {
            "Source_Row_Index": index,
            "Protocol_ID": "p-gold",
            "Prompt_Version": "local-ai-first-v2.1",
            "Title": f"Paper {index}",
            "Abstract": f"Abstract {index}",
            "Decision": decision,
            "Validation_Status": "validated",
            "Escalated": False,
            "Evidence_JSON": "[]",
        }
        for index, decision in enumerate(("KEEP", "REJECT", "MAYBE"), start=1)
    ])
    created = client.post("/gold_validation/sample", json={"question": "Which papers fit?", "sample_size": 60})
    assert created.status_code == 200
    payload = created.json()
    assert payload["sample_size"] == 3
    assert payload["download_url"].startswith("/outputs/gold_validation/")
    assert "manifest_path" not in payload
    assert "label_path" not in payload
    assert list((tmp_path / "private" / "gold_validation").glob("*_manifest.json"))
    assert not list((tmp_path / "gold_validation").glob("*_manifest.json"))

    label_name = payload["download_url"].rsplit("/", 1)[-1]
    labels = pd.read_csv(tmp_path / "gold_validation" / label_name, dtype=str, keep_default_na=False)
    labels["Gold_Decision"] = ["KEEP", "REJECT", "UNSURE"]
    completed = tmp_path / "completed.csv"
    labels.to_csv(completed, index=False)
    with completed.open("rb") as stream:
        evaluated = client.post(
            "/gold_validation/evaluate",
            files={"file": ("completed.csv", stream, "text/csv")},
        )
    assert evaluated.status_code == 200
    report = evaluated.json()
    assert report["label"] == "Provisional—single reviewer"
    assert report["resolved_labels"] == 2
    assert report["unsure_count"] == 1
    assert report["report_download_url"].startswith("/outputs/gold_validation/")


def test_manifest_backed_screening_is_restored_after_server_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "OUTPUT_DIR", str(tmp_path))
    SCREENING_SESSION.begin("cleared")
    output = tmp_path / "runs" / "screened-restored.csv"
    output.parent.mkdir(parents=True)
    pd.DataFrame([{
        "Source_Row_Index": 9,
        "Protocol_ID": "restored",
        "Title": "Restored paper",
        "Abstract": "Restored abstract",
        "Decision": "KEEP",
        "Prompt_Version": "local-semantic-boundary-v3.12",
        "Layer_Trace_JSON": '[{"name":"quick_triage"}]',
    }]).to_csv(output, index=False)
    (tmp_path / "latest_screening.json").write_text(json.dumps({
        "job_id": "restored-job",
        "output_path": str(output),
        "architecture_version": "local-semantic-boundary-v3.12",
    }), encoding="utf-8")
    response = client.get("/screening_results")
    assert response.status_code == 200
    assert response.json()["papers"][0]["Title"] == "Restored paper"
    assert response.json()["job_id"] == "restored-job"
