from __future__ import annotations

import pytest

from litsync_app import app as app_module
from litsync_app.paper_collection.orchestrator import AgenticWorkflowManager
from litsync_app.paper_collection.store import AgenticRunStore


@pytest.fixture(autouse=True)
def isolate_runtime_files(tmp_path, monkeypatch):
    """Keep every test-created runtime artifact in pytest's temporary tree."""
    uploads = tmp_path / "uploads"
    outputs = tmp_path / "outputs"
    private = tmp_path / "private"
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(uploads))
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(outputs))
    monkeypatch.setattr(app_module, "PRIVATE_DIR", str(private))
    monkeypatch.setenv("LOCAL_AI_CACHE_PATH", str(tmp_path / "cache" / "local_ai"))
    monkeypatch.setenv("GEMINI_WEB_PROFILE_DIR", str(tmp_path / "browser_profile"))
    monkeypatch.setattr(
        app_module,
        "AGENTIC_WORKFLOWS",
        AgenticWorkflowManager(
            store=AgenticRunStore(private / "agentic_runs.sqlite3"),
            output_root=outputs,
            private_root=private / "agentic_runs",
        ),
    )
