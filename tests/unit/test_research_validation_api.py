from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from litsync_app import app as server


def test_research_validation_routes_and_panel_are_available():
    routes = {route.path for route in server.app.routes}
    assert {
        "/research_validation/init", "/research_validation/run",
        "/research_validation/{study_id}/status",
        "/research_validation/{study_id}/review-packs",
        "/research_validation/{study_id}/reviews/{reviewer_id}",
        "/research_validation/{study_id}/adjudication-pack",
        "/research_validation/{study_id}/adjudication",
        "/research_validation/{study_id}/report",
        "/research_validation/{study_id}/root-causes",
    } <= routes
    html = server.HTML_FILE.read_text(encoding="utf-8")
    assert 'id="researchValidationCard"' in html
    assert "fetch('/research_validation/init'" in html
    assert "runResearchValidation" in html
    assert "uploadResearchReviews" in html


def test_api_initializes_private_preregistered_study(tmp_path, monkeypatch):
    uploads, outputs, private = tmp_path / "uploads", tmp_path / "outputs", tmp_path / "private"
    for path in (uploads, outputs, private):
        path.mkdir()
    monkeypatch.setattr(server, "UPLOAD_DIR", str(uploads))
    monkeypatch.setattr(server, "OUTPUT_DIR", str(outputs))
    monkeypatch.setattr(server, "PRIVATE_DIR", str(private))
    corpus = tmp_path / "corpus.csv"
    pd.DataFrame({
        "Document Title": [f"Paper {index}" for index in range(70)],
        "Abstract": [f"Abstract {index}" for index in range(70)],
    }).to_csv(corpus, index=False)
    with corpus.open("rb") as handle:
        response = TestClient(server.app).post(
            "/research_validation/init",
            files={"file": ("corpus.csv", handle, "text/csv")},
            data={
                "question": "Which studies address the review objective?",
                "title_column": "Document Title", "abstract_column": "Abstract",
                "reviewer_a": "reviewer-a", "reviewer_b": "reviewer-b",
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "INITIALIZED"
    manifest = private / "research_validation" / payload["study_id"] / "study.json"
    assert manifest.exists()
    assert not list(outputs.rglob("*gold*"))
