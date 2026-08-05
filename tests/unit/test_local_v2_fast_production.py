from __future__ import annotations

from fastapi.testclient import TestClient

from litsync_app import app as server
from litsync_app.prisma import Prisma2020Manifest
from litsync_app.screening.engines import (
    LOCAL_V2_ENGINE,
    LOCAL_V2_FAST_ENGINE,
    normalize_processing_engine,
)
from litsync_app.screening.local_v2.fast import FAST_RUNNER_VERSION


client = TestClient(server.app)


def test_fast_mode_is_explicit_and_separate():
    assert normalize_processing_engine("local-v2-fast") == LOCAL_V2_FAST_ENGINE
    assert normalize_processing_engine("local_ai_v2_fast") == LOCAL_V2_FAST_ENGINE
    assert normalize_processing_engine("local-v2") == LOCAL_V2_ENGINE
    html = client.get("/").text
    assert '<option value="local_v2_fast">' in html
    assert "Local AI v2 Fast" in html
    assert "engine === 'local_v2_fast'" in html
    assert '<option value="local_v2">' in html


def test_fast_endpoint_reports_separate_architecture(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(server, "PRISMA_STORE", Prisma2020Manifest())
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
            "question": "Which studies evaluate the intervention?",
            "screening_engine": "local_v2_fast",
        },
        files={
            "file": (
                "papers.csv",
                b"Title,Abstract\nPaper,Text",
                "text/csv",
            )
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["screening_engine"] == LOCAL_V2_FAST_ENGINE
    assert payload["architecture_version"] == FAST_RUNNER_VERSION
    assert payload["local_profile"] == "local-v2-fast"
    assert payload["prisma"]["screening_engine"] == LOCAL_V2_FAST_ENGINE
    assert started[0]["kwargs"]["screening_engine"] == LOCAL_V2_FAST_ENGINE
