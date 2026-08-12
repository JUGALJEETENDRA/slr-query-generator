from __future__ import annotations

import json
import math
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from fastapi.testclient import TestClient
from playwright.sync_api import sync_playwright

from litsync_app import app as server
from litsync_app.screening.bulk import PROGRESS, SCREENING_SESSION, ScreeningProgress
from litsync_app.screening.local.hardware import HardwareSnapshot, RuntimeProfile
from litsync_app.prisma import Prisma2020Manifest
from litsync_app.integrations.gemini_web_screening import _review_protocol_id


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
            "qwen3.5:4b": 1,
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
    assert "Local AI" in response.text


def test_website_hides_legacy_model_controls_and_uses_relative_screening_api():
    html = client.get("/").text
    assert 'id="model"' not in html
    assert 'name="mode"' not in html
    assert "twoStageEnabled" not in html
    assert 'fd.append("model_tier"' not in html
    assert 'fd.append("resource_profile"' not in html
    assert 'fetch("/screen_csv"' in html
    assert "http://localhost:8000" not in html
    assert "stronger local model" not in html
    assert "Strong-model checks" not in html
    assert "function pipelinePercent(progress = {})" in html
    assert "Local Fast Binary" not in html
    assert "emergency" not in html.lower()


def test_existing_screener_offers_exactly_local_ai_and_gemini_web():
    html = client.get("/").text
    screening_select = html.split('id="screeningEngine"', 1)[1].split("</select>", 1)[0]
    assert screening_select.count("<option") == 2
    assert 'value="local"' in screening_select
    assert 'value="gemini_web"' in screening_select
    assert 'value="local_v2"' not in screening_select
    assert 'value="gemini_api"' not in screening_select
    assert 'value="gemini_web_v24"' not in screening_select
    assert 'value="gemini_web_fast"' not in screening_select
    assert 'id="geminiApiKey"' not in html
    assert 'fd.append("screening_engine", screeningEngine)' in html
    assert "localStorage" not in html
    assert "sessionStorage" not in html





def test_all_screening_engines_start_with_the_same_prisma_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(server, "PRISMA_STORE", Prisma2020Manifest())
    monkeypatch.setattr(server.PROGRESS, "start_job", lambda job_id: True)

    class NoopThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            return None

    monkeypatch.setattr(server, "Thread", NoopThread)
    for engine in ("local", "gemini_web"):
        data = {"question": "RQ", "screening_engine": engine}
        response = client.post(
            "/screen_csv", data=data,
            files={"file": (f"{engine}.csv", b"Title,Abstract\nPaper,Text", "text/csv")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["prisma"]["standard"] == "PRISMA 2020"
        assert payload["prisma"]["screening_engine"] == engine
        assert payload["prisma"]["workflow_id"] == payload["job_id"]
        assert set(payload["prisma_downloads"]) == {"json", "csv", "svg"}


@pytest.mark.parametrize(
    "obsolete_engine",
    ["gemini_web_fast", "gemini_web_v24", "gemini_api", "local_v2"],
)
def test_obsolete_screening_engines_are_rejected_before_start(monkeypatch, obsolete_engine):
    started = []
    monkeypatch.setattr(
        server.PROGRESS,
        "start_job",
        lambda job_id: started.append(job_id) or True,
    )
    response = client.post(
        "/screen_csv",
        data={"question": "RQ", "screening_engine": obsolete_engine},
        files={"file": ("papers.csv", b"Title,Abstract\nPaper,Text", "text/csv")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Choose Gemini Web or Local AI for screening."
    assert started == []


def test_public_website_has_no_partial_dataset_test_mode():
    html = client.get("/").text
    assert 'id="quickTest100"' not in html
    assert "Limit this run to the first 100 papers" not in html
    assert 'fd.append("max_rows"' not in html


def test_resume_checkbox_is_visible_unchecked_and_posts_its_actual_state():
    html = client.get("/").text
    assert 'id="resumeScreening"' in html
    assert 'id="resumeScreening" checked' not in html
    assert "Resume validated results from an interrupted identical run" in html
    assert (
        'document.getElementById("resumeScreening").checked ? "true" : "false"'
        in html
    )
    assert 'fd.append("resume", "true")' not in html
    assert 'document.getElementById("resumeScreening").disabled = disabled' in html


def test_fast_completed_dashboard_keeps_actual_batches_and_runtime():
    html = client.get("/").text
    assert "Primary batches:" in html
    assert "p.primary_batches_completed" in html
    assert "p.primary_batches_submitted" in html
    assert "Verification batches:" in html
    assert "p.verification_batches_completed" in html
    assert "p.verification_batches_submitted" in html
    assert "renderDashboard(getScreenCounts(), latestProgress || {})" in html


def test_completed_screening_exposes_six_automatic_job_specific_downloads():
    html = client.get("/").text
    for label in (
        "Download All Papers",
        "Download KEEP Papers",
        "Download MAYBE Papers",
        "Download REJECT Papers",
        "Download Human Review Queue",
        "Download Screening Summary",
    ):
        assert label in html
    assert "Bound screening job ID:" in html
    assert "downloadScreeningExport" in html
    assert "/screening-jobs/${encodeURIComponent(requestedJobId)}/exports" in html
    assert "screeningExportBusy" in html
    assert "Finalizing Results" not in html
    assert "Finalize Results" not in html
    assert "CSV downloads are generated only after finalization" not in html
    assert "reviewing MAYBE papers is optional" in html
    assert "available evidence or technical validation was insufficient" in html
    assert "80% accuracy" not in html
    assert "publication-grade accuracy" not in html


def test_download_button_prepares_latest_job_export_and_recovers_after_error():
    html = client.get("/").text
    calls = []

    def handle_route(route):
        parsed = urlparse(route.request.url)
        if parsed.path == "/":
            route.fulfill(status=200, content_type="text/html", body=html)
        elif parsed.path == "/progress":
            route.fulfill(status=404, json={"detail": "none"})
        elif parsed.path == "/status":
            route.fulfill(status=200, json={"ollama_ready": True, "missing_models": []})
        elif parsed.path == "/screening-jobs/ui-job/exports":
            calls.append("prepare")
            route.fulfill(status=200, json={
                "status": "success",
                "job_id": "ui-job",
                "counts": {"all": 3, "keep": 1, "maybe": 1, "reject": 1, "review_queue": 1},
                "downloads": {"all": "/screening-jobs/ui-job/exports/all"},
                "filenames": {"all": "screened_all.csv"},
            })
        elif parsed.path == "/screening-jobs/ui-job/exports/all":
            calls.append("download")
            route.fulfill(
                status=200,
                body="Title,Decision\nPaper,KEEP\n",
                content_type="text/csv",
                headers={"Content-Disposition": 'attachment; filename="screened_all.csv"'},
            )
        elif parsed.path == "/screening-jobs/error-job/exports":
            calls.append("error")
            route.fulfill(status=422, json={"detail": "Persisted output failed validation."})
        else:
            route.abort()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(accept_downloads=True)
        page.route("**/*", handle_route)
        page.goto("http://litsync.test")
        page.evaluate("""() => {
            switchTab('scr');
            activeScreeningJobId = 'ui-job';
            screeningPapers = [
                {Decision: 'KEEP'}, {Decision: 'MAYBE'}, {Decision: 'REJECT'}
            ];
            document.getElementById('scrRes').classList.add('on');
            renderDownloadButtons(getScreenCounts());
        }""")
        assert page.locator("#downloadBoundJob").text_content() == (
            "Bound screening job ID: ui-job"
        )
        with page.expect_download() as pending:
            page.evaluate("document.querySelector('[data-export-name=\"all\"]').click()")
        download = pending.value
        assert download.suggested_filename == "screened_all.csv"
        assert download.url.endswith("/screening-jobs/ui-job/exports/all")
        page.wait_for_function("() => screeningExportBusy === false")
        assert calls[0] == "prepare"
        assert page.locator('[data-export-name="all"]').is_enabled()

        page.evaluate("""() => {
            activeScreeningJobId = 'error-job';
            renderDownloadButtons(getScreenCounts());
        }""")
        page.evaluate("document.querySelector('[data-export-name=\"all\"]').click()")
        page.wait_for_function(
            "() => document.getElementById('scrMsg').textContent.includes('Persisted output failed validation.')"
        )
        assert page.locator('[data-export-name="all"]').is_enabled()
        assert calls[-1] == "error"
        browser.close()


@pytest.mark.parametrize(("posted", "expected"), (("false", False), ("true", True)))
def test_screening_endpoint_receives_the_posted_resume_checkbox_state(
    monkeypatch, posted, expected,
):
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
        data={"question": "RQ", "resume": posted},
        files={"file": ("papers.csv", b"Title,Abstract\nPaper,Text", "text/csv")},
    )
    assert response.status_code == 200
    assert started[0]["kwargs"]["resume"] is expected


def test_research_context_is_optional_and_sent_only_to_local_screening():
    html = client.get("/").text
    assert 'id="researchContext"' in html
    assert 'fd.append("research_context", researchContext)' in html
    assert 'document.getElementById("researchContext").disabled = disabled' in html


def test_authoritative_criteria_fields_are_visible_and_submitted_verbatim():
    html = client.get("/").text
    assert '<label for="inclusionCriteria"' in html
    assert "Inclusion Criteria" in html
    assert '<textarea id="inclusionCriteria"' in html
    assert '<label for="exclusionCriteria"' in html
    assert "Exclusion Criteria" in html
    assert '<textarea id="exclusionCriteria"' in html
    assert html.count(
        "Enter one criterion per line or separate criteria with semicolons."
    ) == 2
    assert (
        'const inclusionCriteriaValue = '
        'document.getElementById("inclusionCriteria").value;'
    ) in html
    assert (
        'const exclusionCriteriaValue = '
        'document.getElementById("exclusionCriteria").value;'
    ) in html
    assert 'fd.append("inclusion_criteria", inclusionCriteriaValue);' in html
    assert 'fd.append("exclusion_criteria", exclusionCriteriaValue);' in html
    assert 'document.getElementById("inclusionCriteria").disabled = disabled' in html
    assert 'document.getElementById("exclusionCriteria").disabled = disabled' in html


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
    assert 'id="goldJobId"' in html
    assert "goldValidatedJobId" in html
    assert "form.append('job_id', jobId)" in html
    assert "document.getElementById('goldDownloadBtn').disabled = !validatedJobId" in html
    assert "document.getElementById('goldEvaluateBtn').disabled = !validatedJobId || !file" in html
    assert "fetch('/gold_validation/sample'" in html
    assert "fetch('/gold_validation/evaluate'" in html


def test_gold_frontend_uses_only_validated_job_state_and_rejects_stale_loads():
    html = client.get("/").text
    base_url = "http://litsync.test"
    restore_mode = {"enabled": False}
    requests_seen = {"sample": None, "evaluation": None}

    def screening_result(job_id):
        return {
            "status": "finished",
            "job_id": job_id,
            "papers": [{
                "Source_Row_Index": 1,
                "Title": f"Paper for {job_id}",
                "Abstract": "Persisted abstract",
                "Decision": "KEEP",
            }],
            "prisma": {},
            "prisma_downloads": {},
        }

    def handle_route(route):
        request = route.request
        parsed = urlparse(request.url)
        if parsed.path == "/":
            route.fulfill(status=200, content_type="text/html", body=html)
        elif parsed.path == "/progress":
            payload = (
                {"status": "finished", "job_id": "restored-job"}
                if restore_mode["enabled"]
                else {"status": "idle"}
            )
            route.fulfill(status=200, json=payload)
        elif parsed.path == "/status":
            route.fulfill(
                status=200,
                json={"ollama_ready": True, "missing_models": []},
            )
        elif parsed.path == "/screening_results":
            job_id = parse_qs(parsed.query).get("job_id", [""])[0]
            payload = (
                screening_result(job_id)
                if job_id
                else {"status": "empty", "papers": []}
            )
            route.fulfill(status=200, json=payload)
        elif parsed.path == "/gold_validation/sample":
            requests_seen["sample"] = json.loads(request.post_data or "{}")
            route.fulfill(
                status=200,
                json={
                    "status": "success",
                    "sample_size": 1,
                    "download_url": "#goldValidationCard",
                },
            )
        elif parsed.path == "/gold_validation/evaluate":
            requests_seen["evaluation"] = request.post_data_buffer.decode(
                "utf-8", errors="replace"
            )
            route.fulfill(
                status=200,
                json={
                    "status": "success",
                    "metrics": {},
                    "full_run_safety": {},
                    "confidence_intervals_95": {},
                    "false_keeps": [],
                    "false_rejects": [],
                    "resolved_labels": 1,
                    "sample_size": 1,
                    "unsure_count": 0,
                    "blank_label_count": 0,
                    "missing_row_count": 0,
                    "report_download_url": "#goldValidationReport",
                },
            )
        else:
            route.abort()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.route("**/*", handle_route)
        page.goto(base_url)
        page.evaluate(
            """() => {
                switchTab('scr');
                document.getElementById('goldValidationCard').style.display = 'block';
                document.getElementById('qi').value = 'Which papers fit?';
            }"""
        )

        candidate = page.locator("#goldJobId")
        def set_candidate(value):
            page.evaluate(
                """value => {
                    const input = document.getElementById('goldJobId');
                    input.value = value;
                    input.dispatchEvent(new Event('input', {bubbles: true}));
                }""",
                value,
            )

        set_candidate("persisted-job")
        assert page.locator("#goldBoundJob").text_content() == (
            "Screening job persisted-job is not validated. "
            "Click Load persisted job."
        )
        assert page.locator("#goldDownloadBtn").is_disabled()
        assert page.locator("#goldEvaluateBtn").is_disabled()
        assert page.evaluate("goldValidatedJobId") == ""

        page.evaluate("() => { selectGoldJob(); }")
        page.wait_for_function(
            "() => document.getElementById('goldValidationMsg').textContent"
            ".includes('Persisted screening job selected.')"
        )
        assert page.evaluate("goldValidatedJobId") == "persisted-job"
        assert page.evaluate("activeScreeningJobId") == "persisted-job"
        assert page.locator("#goldBoundJob").text_content() == (
            "Gold Validation is bound to screening job persisted-job."
        )
        assert page.locator("#goldDownloadBtn").is_enabled()
        assert page.locator("#goldEvaluateBtn").is_disabled()

        page.evaluate("() => { downloadGoldSample(); }")
        page.wait_for_function("() => window.location.hash === '#goldValidationCard'")
        assert requests_seen["sample"]["job_id"] == "persisted-job"

        page.locator("#goldLabelFile").set_input_files({
            "name": "completed.csv",
            "mimeType": "text/csv",
            "buffer": b"Gold_Decision\nKEEP\n",
        })
        assert page.locator("#goldEvaluateBtn").is_enabled()
        page.evaluate("() => { evaluateGoldLabels(); }")
        page.wait_for_function(
            "() => document.getElementById('goldValidationMsg').textContent"
            ".includes('Gold validation report created.')"
        )
        assert 'name="job_id"' in requests_seen["evaluation"]
        assert "persisted-job" in requests_seen["evaluation"]

        set_candidate("edited-job")
        assert page.evaluate("goldValidatedJobId") == ""
        assert page.locator("#goldDownloadBtn").is_disabled()
        assert page.locator("#goldEvaluateBtn").is_disabled()
        assert "is not validated" in page.locator("#goldBoundJob").text_content()

        page.evaluate(
            """() => {
                window.__realFetch = window.fetch;
                window.__goldPending = {};
                window.fetch = (url, options = {}) => {
                    const text = String(url);
                    if (!text.startsWith('/screening_results?job_id=')) {
                        return window.__realFetch(url, options);
                    }
                    const jobId = new URL(text, window.location.href)
                        .searchParams.get('job_id');
                    return new Promise(resolve => {
                        window.__goldPending[jobId] = (result = {}) => resolve({
                            ok: result.ok !== false,
                            json: async () => result.data || {
                                status: 'finished',
                                job_id: jobId,
                                papers: [{Title: jobId, Decision: 'KEEP'}],
                            },
                        });
                    });
                };
            }"""
        )
        set_candidate("older-job")
        page.evaluate("() => { selectGoldJob(); }")
        page.wait_for_function("() => Boolean(window.__goldPending['older-job'])")
        set_candidate("newer-job")
        page.evaluate("() => { selectGoldJob(); }")
        page.wait_for_function("() => Boolean(window.__goldPending['newer-job'])")
        page.evaluate("window.__goldPending['older-job']()")
        page.wait_for_timeout(25)
        assert page.evaluate("goldValidatedJobId") == ""
        page.evaluate("window.__goldPending['newer-job']()")
        page.wait_for_function("() => goldValidatedJobId === 'newer-job'")
        assert page.locator("#goldBoundJob").text_content() == (
            "Gold Validation is bound to screening job newer-job."
        )

        page.evaluate(
            """() => {
                activeScreeningJobId = 'manual-review-job';
                screeningPapers = [{Title: 'manual-review-sentinel'}];
            }"""
        )
        set_candidate("missing-job")
        page.evaluate("() => { selectGoldJob(); }")
        page.wait_for_function("() => Boolean(window.__goldPending['missing-job'])")
        page.evaluate(
            """window.__goldPending['missing-job']({
                ok: false,
                data: {detail: 'Persisted completed screening job was not found.'}
            })"""
        )
        page.wait_for_function(
            "() => document.getElementById('goldValidationMsg').textContent"
            ".includes('Persisted completed screening job was not found.')"
        )
        assert page.evaluate("activeScreeningJobId") == "manual-review-job"
        assert page.evaluate("screeningPapers[0].Title") == (
            "manual-review-sentinel"
        )

        restore_mode["enabled"] = True
        restored_page = browser.new_page()
        restored_page.route("**/*", handle_route)
        restored_page.goto(base_url)
        restored_page.wait_for_function(
            "() => goldValidatedJobId === 'restored-job'"
        )
        assert restored_page.locator("#goldBoundJob").text_content() == (
            "Gold Validation is bound to screening job restored-job."
        )
        assert restored_page.locator("#goldDownloadBtn").is_enabled()
        assert restored_page.evaluate("activeScreeningJobId") == "restored-job"
        browser.close()


def test_query_generator_exposes_degraded_mode_without_fake_queries():
    html = client.get("/").text
    assert "data.concepts?.generation_status" in html
    assert "Reduced-confidence result." in html
    assert "demoResult" not in html
    assert "Backend not running." not in html
    assert "Could not generate queries." in html


def test_prisma_ui_uses_only_the_backend_canonical_manifest():
    html = client.get("/").text
    assert "renderCanonicalPrisma" in html
    assert "prisma.revision" in html
    assert "Download SVG" in html
    assert "Download JSON manifest" in html
    assert "Download CSV manifest" in html
    assert "Could not load PRISMA record" in html
    assert "const PRISMA" not in html
    assert "renderLegacyPrisma" not in html
    assert "renderPrismaFromCounts" not in html
    assert "Reject + Maybe" not in html
    assert "Studies included in final synthesis" not in html
    assert "Reports sought, retrieved" not in html


def test_manual_review_and_prisma_exports_use_server_owned_rows(monkeypatch, tmp_path):
    store = Prisma2020Manifest()
    monkeypatch.setattr(server, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(server, "PRISMA_STORE", store)
    job_id = "prisma-manual-job"
    rows = []
    decisions = ["KEEP"] * 3 + ["MAYBE"] * 2 + ["REJECT"] * 5
    for index, decision in enumerate(decisions):
        rows.append({
            "Source_Row_Index": index,
            "Title": f"Paper {index}",
            "Abstract": f"Abstract {index}",
            "Decision": decision,
            "Original_Decision": decision,
            "Decision_Source": "tool_assisted_screening",
            "Exclusion_Reason": "",
        })
    output = tmp_path / "runs" / f"screened-{job_id}.csv"
    output.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    SCREENING_SESSION.begin(job_id, str(output), "local-semantic-boundary-v3.12")
    SCREENING_SESSION.set_results(rows, job_id=job_id, output_path=str(output))
    assert PROGRESS.start_job(job_id)
    PROGRESS.begin_screening(job_id, 10, "local-semantic-boundary-v3.12")
    PROGRESS.update_counts(job_id, 10, 3, 2, 5)
    PROGRESS.finish(job_id)
    store.begin_screening(
        output_root=tmp_path, job_id=job_id, input_fingerprint="input",
        screening_engine="local",
    )
    store.configure_screening(
        job_id, input_rows=10, missing_abstracts=0,
        records_available=10, records_selected=10,
    )

    first = client.patch(
        f"/screening_jobs/{job_id}/records/3", json={"decision": "KEEP"}
    )
    assert first.status_code == 200
    assert first.json()["prisma"]["screening"]["records_awaiting_manual_review"] == 1
    blocked = client.post("/finalize", json={"job_id": job_id})
    assert blocked.status_code == 400
    assert "Resolve all MAYBE" in blocked.json()["detail"]

    second = client.patch(
        f"/screening_jobs/{job_id}/records/4",
        json={"decision": "REJECT", "exclusion_reason": "Explicit reviewer reason"},
    )
    assert second.status_code == 200
    finalized = client.post("/finalize", json={"job_id": job_id})
    assert finalized.status_code == 200
    payload = finalized.json()
    assert payload["counts"] == {"total": 10, "keep": 4, "maybe": 0, "reject": 6}
    assert payload["prisma"]["screening"]["records_included_after_title_abstract"] == 4
    assert payload["prisma"]["screening"]["records_excluded"] == 6
    assert payload["prisma"]["integrity"]["csv_counts_match"] is True
    for key in ("screened", "included", "excluded", "prisma_svg", "prisma_json", "prisma_csv"):
        assert key in payload["files"]
    assert client.get(f"/prisma/{job_id}").status_code == 200
    assert "PRISMA 2020" in client.get(f"/prisma/{job_id}.svg").text
    assert "screening.records_excluded,6" in client.get(f"/prisma/{job_id}.csv").text


def test_status_stays_lightweight_without_ui_presets(monkeypatch):
    monkeypatch.setattr(server, "resolve_runtime_profile", _profile)
    payload = client.get("/status").json()
    assert payload["backend_ready"] is True
    assert payload["ollama_ready"] is True
    assert payload["resolved"]["tier"] == "balanced"
    assert payload["resolved"]["model"] == server.LOCAL_MODEL
    assert payload["resolved"]["architecture_version"] == server.LOCAL_AI_ARCHITECTURE
    assert payload["missing_models"] == []
    assert "presets" not in payload


def test_results_are_json_safe_and_have_browser_download_url():
    output_path = str(Path(server.OUTPUT_DIR) / "runs" / "screened-json-job.csv")
    SCREENING_SESSION.begin(
        "json-job", output_path, "local-semantic-boundary-v3.12"
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
    assert payload["screening_engine"] == "local"
    assert payload["architecture_version"] == "local-ai-simple-v1"
    assert started[0]["kwargs"]["output_path"].endswith(f"screened-{payload['job_id']}.csv")
    assert len(started[0]["kwargs"]["input_fingerprint"]) == 64
    assert started[0]["kwargs"]["max_rows"] is None
    assert started[0]["kwargs"]["research_context"] == "This explains the intended meaning only."


def test_csv_preserves_authoritative_criteria_and_persists_protocol_audit(
    monkeypatch, tmp_path,
):
    store = Prisma2020Manifest()
    monkeypatch.setattr(server, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(server, "PRISMA_STORE", store)
    monkeypatch.setattr(server.PROGRESS, "start_job", lambda job_id: True)
    started = []

    class NoopThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            started.append(self.kwargs)

    monkeypatch.setattr(server, "Thread", NoopThread)
    inclusion = (
        "Measure learner retention; compare two lesson plans.\n"
        "Report confidence intervals, in the original order!"
    )
    exclusion = (
        "Exclude secondary syntheses; exclude opinion articles.\n"
        "Exclude records without an evaluated intervention?"
    )
    context = "Use the source paper's own reported analysis."
    response = client.post(
        "/screen_csv",
        data={
            "question": "Which teaching interventions improve learner retention?",
            "research_context": context,
            "inclusion_criteria": inclusion,
            "exclusion_criteria": exclusion,
            "screening_engine": "gemini_web",
        },
        files={"file": ("papers.csv", b"Title,Abstract\nPaper,Text", "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert started[0]["kwargs"]["inclusion_criteria"] == inclusion
    assert started[0]["kwargs"]["exclusion_criteria"] == exclusion

    expected_audit = {
        "research_question": "Which teaching interventions improve learner retention?",
        "research_context": context,
        "inclusion_criteria": inclusion,
        "exclusion_criteria": exclusion,
        "parsed_authoritative_inclusion_count": 3,
        "parsed_authoritative_exclusion_count": 3,
    }
    assert payload["prisma"]["protocol_inputs"] == expected_audit
    persisted = json.loads(
        (
            tmp_path / "prisma" / f"{payload['job_id']}.json"
        ).read_text(encoding="utf-8")
    )
    assert persisted["protocol_inputs"] == expected_audit
    assert store.snapshot(payload["job_id"])["protocol_inputs"] == expected_audit


def test_csv_supports_explicitly_empty_authoritative_criteria(monkeypatch, tmp_path):
    store = Prisma2020Manifest()
    monkeypatch.setattr(server, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(server, "PRISMA_STORE", store)
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
        data={
            "question": "Which interventions improve the requested outcome?",
            "inclusion_criteria": "",
            "exclusion_criteria": "",
        },
        files={"file": ("papers.csv", b"Title,Abstract\nPaper,Text", "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert started[0]["kwargs"]["inclusion_criteria"] == ""
    assert started[0]["kwargs"]["exclusion_criteria"] == ""
    audit = payload["prisma"]["protocol_inputs"]
    assert audit["inclusion_criteria"] == ""
    assert audit["exclusion_criteria"] == ""
    assert audit["parsed_authoritative_inclusion_count"] == 0
    assert audit["parsed_authoritative_exclusion_count"] == 0


def test_protocol_cache_identity_changes_with_authoritative_criteria():
    common = {
        "question": "Which interventions improve the requested outcome?",
        "context": "Interpret the requested outcome as directly measured.",
    }
    baseline = _review_protocol_id({
        **common,
        "inclusion": "Must report measured results",
        "exclusion": "Exclude opinion articles",
    })
    changed_inclusion = _review_protocol_id({
        **common,
        "inclusion": "Must compare measured results",
        "exclusion": "Exclude opinion articles",
    })
    changed_exclusion = _review_protocol_id({
        **common,
        "inclusion": "Must report measured results",
        "exclusion": "Exclude secondary syntheses",
    })
    assert baseline != changed_inclusion
    assert baseline != changed_exclusion


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
    monkeypatch.setattr(
        "litsync_app.screening.bulk.time.perf_counter", lambda: next(clock)
    )
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
    job_id = "gold-memory-job"
    SCREENING_SESSION.begin(job_id)
    rows = [
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
    ]
    SCREENING_SESSION.set_results(rows, job_id=job_id)
    output = tmp_path / "runs" / f"screened-{job_id}.csv"
    output.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    created = client.post("/gold_validation/sample", json={
        "job_id": job_id,
        "question": "Which papers fit?",
        "sample_size": 60,
    })
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
    assert list(labels.columns) == [
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
    assert set(labels["Screening_Job_ID"]) == {job_id}
    labels["Gold_Decision"] = ["KEEP", "REJECT", "UNSURE"]
    completed = tmp_path / "completed.csv"
    labels.to_csv(completed, index=False)
    SCREENING_SESSION.begin("cleared-before-gold-evaluation")
    with completed.open("rb") as stream:
        evaluated = client.post(
            "/gold_validation/evaluate",
            data={"job_id": job_id},
            files={"file": ("completed.csv", stream, "text/csv")},
        )
    assert evaluated.status_code == 200
    report = evaluated.json()
    assert report["label"] == "Provisional—single reviewer"
    assert report["resolved_labels"] == 2
    assert report["unsure_count"] == 1
    assert report["report_download_url"].startswith("/outputs/gold_validation/")


def test_manifest_backed_screening_can_create_gold_sample_after_restart(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(server, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(server, "PRIVATE_DIR", str(tmp_path / "private"))
    monkeypatch.setattr(server.PROGRESS, "is_running", lambda: False)
    job_id = "restored-gold-job"
    rows = [
        {
            "Source_Row_Index": index,
            "Protocol_ID": "restored-gold",
            "Prompt_Version": "local-semantic-boundary-v3.12",
            "Title": f"Restored paper {index}",
            "Abstract": f"Restored abstract {index}",
            "Decision": decision,
            "Validation_Status": "validated",
            "Escalated": False,
            "Evidence_JSON": "[]",
        }
        for index, decision in enumerate(("KEEP", "REJECT", "MAYBE"), start=1)
    ]
    output = tmp_path / "runs" / f"screened-{job_id}.csv"
    output.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    (tmp_path / "latest_screening.json").write_text(json.dumps({
        "job_id": job_id,
        "output_path": str(output),
        "architecture_version": "local-semantic-boundary-v3.12",
    }), encoding="utf-8")
    SCREENING_SESSION.begin("cleared")

    response = client.post("/gold_validation/sample", json={
        "job_id": job_id,
        "question": "Which restored papers fit?",
        "sample_size": 60,
    })

    assert response.status_code == 200
    assert response.json()["sample_size"] == 3
    assert SCREENING_SESSION.metadata()["job_id"] == "cleared"


def test_gold_evaluation_rejects_csv_bound_to_another_persisted_job(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(server, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(server, "PRIVATE_DIR", str(tmp_path / "private"))
    rows = [
        {
            "Source_Row_Index": index,
            "Protocol_ID": "p-gold-wrong-job",
            "Prompt_Version": "gemini-web-screening-prompt-v6",
            "Title": f"Paper {index}",
            "Abstract": f"Abstract {index}",
            "Decision": decision,
            "Validation_Status": "validated",
            "Escalated": False,
            "Evidence_JSON": "[]",
        }
        for index, decision in enumerate(("KEEP", "REJECT", "MAYBE"), start=1)
    ]
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    for job_id in ("job-x", "job-y"):
        pd.DataFrame(rows).to_csv(
            runs_dir / f"screened-{job_id}.csv", index=False
        )
    created = client.post("/gold_validation/sample", json={
        "job_id": "job-x",
        "question": "Which papers fit?",
        "sample_size": 60,
    })
    label_name = created.json()["download_url"].rsplit("/", 1)[-1]
    label_path = tmp_path / "gold_validation" / label_name

    with label_path.open("rb") as stream:
        evaluated = client.post(
            "/gold_validation/evaluate",
            data={"job_id": "job-y"},
            files={"file": ("completed.csv", stream, "text/csv")},
        )

    assert evaluated.status_code == 400
    assert evaluated.json()["detail"] == (
        "This validation CSV belongs to screening job 'job-x', not 'job-y'."
    )
    assert not list((tmp_path / "gold_validation").glob("*_report.json"))


def test_gold_sample_rejects_unknown_job_id(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(server, "PRIVATE_DIR", str(tmp_path / "private"))
    SCREENING_SESSION.begin("known-gold-job")
    SCREENING_SESSION.set_results([{
        "Source_Row_Index": 1,
        "Protocol_ID": "known-gold",
        "Title": "Known paper",
        "Abstract": "Known abstract",
        "Decision": "KEEP",
    }], job_id="known-gold-job")

    response = client.post("/gold_validation/sample", json={
        "job_id": "unknown-gold-job",
        "question": "Which papers fit?",
    })

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "Persisted screening output for job 'unknown-gold-job' was not found."
    )


def test_gold_evaluation_requires_explicit_job_id(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(server, "PRIVATE_DIR", str(tmp_path / "private"))
    response = client.post(
        "/gold_validation/evaluate",
        files={"file": ("completed.csv", b"Validation_Set_ID\n", "text/csv")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Select a completed screening job before evaluating Gold Validation."
    )


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
            "Prompt_Version": "local-ai-screening-v2",
    }]).to_csv(output, index=False)
    (tmp_path / "latest_screening.json").write_text(json.dumps({
        "job_id": "restored-job",
        "output_path": str(output),
            "architecture_version": "local-ai-simple-v1",
    }), encoding="utf-8")
    response = client.get("/screening_results")
    assert response.status_code == 200
    assert response.json()["papers"][0]["Title"] == "Restored paper"
    assert response.json()["job_id"] == "restored-job"
