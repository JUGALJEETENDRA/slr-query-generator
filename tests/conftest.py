from __future__ import annotations

import pytest

from litsync_app import app as app_module
from litsync_app.experimental_collection.service import ExperimentalCollectionService
from litsync_app.experimental_collection.store import CollectionStore


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
        "EXPERIMENTAL_COLLECTION",
        ExperimentalCollectionService(
            store=CollectionStore(private / "experimental_collection.sqlite3"),
            root=outputs / "experimental-collection",
            private_root=private / "experimental_collection",
        ),
    )
