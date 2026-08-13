from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from litsync_app import app as server
from litsync_app.experimental_collection.importers import normalize_export, read_export
from litsync_app.experimental_collection.service import ExperimentalCollectionService
from litsync_app.experimental_collection.store import CollectionStore


QUERIES = {
    "google_scholar": '"adaptive learning" AND students',
    "scopus": 'TITLE-ABS-KEY("adaptive learning" AND students)',
    "web_of_science": 'TS=("adaptive learning" AND students)',
    "ieee_xplore": '("All Metadata":"adaptive learning")',
    "pubmed": '("adaptive learning"[tiab]) AND students[tiab]',
}


def service(tmp_path):
    return ExperimentalCollectionService(
        store=CollectionStore(tmp_path / "private" / "runs.sqlite3"),
        root=tmp_path / "outputs" / "experimental-collection",
        private_root=tmp_path / "private",
    )


def write_scopus(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_create_checkpoints_exact_queries_and_modes(tmp_path):
    manager = service(tmp_path)
    run = manager.create("How does adaptive learning affect students?", QUERIES, 100)
    restored = manager.get(run.run_id)
    assert {key: state.query for key, state in restored.sources.items()} == QUERIES
    assert restored.sources["pubmed"].mode == "automated"
    assert restored.sources["scopus"].mode == "assisted"
    assert restored.sources["google_scholar"].mode == "assisted"
    checkpoint = tmp_path / "private" / "runs" / run.run_id / "requested_queries.json"
    assert checkpoint.is_file()
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["queries"]["scopus"] == QUERIES["scopus"]


def test_native_exports_are_preserved_validated_and_deduplicated(tmp_path):
    manager = service(tmp_path)
    run = manager.create("How does adaptive learning affect students?", QUERIES, 100)
    first = tmp_path / "scopus.csv"
    second = tmp_path / "pubmed.csv"
    write_scopus(first, [
        {"Authors": "A", "Title": "Shared Paper", "Year": 2024, "Source title": "J", "DOI": "10.1/shared", "Abstract": "A"},
        {"Authors": "B", "Title": "Scopus Paper", "Year": 2023, "Source title": "J", "DOI": "10.1/scopus", "Abstract": "B"},
    ])
    write_scopus(second, [
        {"Authors": "A", "Title": "Shared Paper", "Year": 2024, "Source title": "J", "DOI": "10.1/shared", "Abstract": "A"},
        {"Authors": "C", "Title": "PubMed Paper", "Year": 2022, "Source title": "J", "DOI": "10.1/pubmed", "Abstract": "C"},
    ])
    manager.import_path(run.run_id, "scopus", first)
    manager.import_path(run.run_id, "pubmed", second)
    finished = manager.finalize(run.run_id)
    assert finished.status == "completed"
    assert finished.counts == {"identified": 4, "deduplicated": 3, "duplicates_removed": 1}
    raw = next((tmp_path / "private" / "runs" / run.run_id / "scopus").glob("raw-*-scopus.csv"))
    assert raw.read_bytes() == first.read_bytes()
    combined = pd.read_csv(tmp_path / "outputs" / "experimental-collection" / run.run_id / "combined_with_provenance.csv")
    assert set(combined["Collection Source"]) == {"scopus", "pubmed"}
    clean = pd.read_csv(tmp_path / "outputs" / "experimental-collection" / run.run_id / "clean_dataset.csv")
    shared = clean.loc[clean["Title"] == "Shared Paper"].iloc[0]
    assert set(shared["Collection Sources"].split("; ")) == {"scopus", "pubmed"}
    assert json.loads(shared["Collection Queries JSON"])["scopus"] == QUERIES["scopus"]
    assert finished.sources["scopus"].raw_sha256


def test_ris_import_and_invalid_empty_export(tmp_path):
    ris = tmp_path / "records.ris"
    ris.write_text(
        "TY  - JOUR\nTI  - A useful study\nAU  - One, A\nAU  - Two, B\nPY  - 2024\nDO  - 10.1/example\nAB  - Evidence.\nER  -\n",
        encoding="utf-8",
    )
    mapped = normalize_export(ris, "web_of_science")
    assert mapped.iloc[0]["Title"] == "A useful study"
    assert "One, A" in mapped.iloc[0]["Authors"]
    empty = tmp_path / "empty.csv"
    empty.write_text("Title,Abstract\n", encoding="utf-8")
    try:
        normalize_export(empty, "scopus")
    except ValueError as exc:
        assert "no records" in str(exc).lower()
    else:
        raise AssertionError("empty export was accepted")


def test_api_upload_resume_finalize_and_ui_contract(monkeypatch, tmp_path):
    manager = service(tmp_path)
    monkeypatch.setattr(server, "EXPERIMENTAL_COLLECTION", manager)
    client = TestClient(server.app)
    created = client.post("/experimental/collection-runs", json={
        "research_question": "How does adaptive learning affect students?",
        "queries": QUERIES, "limit": 100,
    })
    assert created.status_code == 201
    run_id = created.json()["run_id"]
    payload = b"Authors,Title,Year,Source title,DOI,Abstract\nA,Paper,2024,J,10.1/a,Useful\n"
    uploaded = client.post(
        f"/experimental/collection-runs/{run_id}/sources/scopus/upload",
        files={"file": ("scopus.csv", payload, "text/csv")},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["sources"]["scopus"]["records"] == 1
    assert client.get(f"/experimental/collection-runs/{run_id}").status_code == 200
    finalized = client.post(f"/experimental/collection-runs/{run_id}/finalize")
    assert finalized.status_code == 200
    assert finalized.json()["artifacts"]["clean_dataset"].endswith("clean_dataset.csv")
    html = Path("web/slr_query_generator.html").read_text(encoding="utf-8")
    assert "Experimental paper collection" in html
    assert "collection_run" in html
    assert "/experimental/collection-runs" in html
