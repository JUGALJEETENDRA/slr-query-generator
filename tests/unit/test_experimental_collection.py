from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from litsync_app import app as server
from litsync_app.experimental_collection.browser_collectors import (
    CollectionNeedsAttention, source_launch_url,
)
from litsync_app.experimental_collection.importers import (
    detect_export_format, normalize_export, read_export, source_warnings,
)
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


def test_pubmed_tagged_text_preserves_multiple_records_and_multiline_abstracts(tmp_path):
    export = tmp_path / "pubmed.txt"
    export.write_text(
        "PMID- 1\nTI  - First paper\nAB  - First abstract line\n      continued evidence.\nDP  - 2024\nLID - 10.1/first [doi]\n\n"
        "PMID- 2\nTI  - Second paper\nAB  - Second abstract.\nDP  - 2023\nAID - 10.1/second [doi]\n",
        encoding="utf-8",
    )
    frame = read_export(export)
    assert len(frame) == 2
    assert frame.iloc[0]["Abstract"] == "First abstract line continued evidence."
    assert frame.iloc[0]["DOI"] == "10.1/first"
    assert detect_export_format(export, frame) == "pubmed_text"
    assert not source_warnings("pubmed", "pubmed_text", frame)


def test_assisted_launch_urls_preserve_exact_queries():
    scholar = source_launch_url("google_scholar", QUERIES["google_scholar"])
    ieee = source_launch_url("ieee_xplore", QUERIES["ieee_xplore"])
    assert "scholar?q=" in scholar and "%22adaptive+learning%22" in scholar
    assert "queryText=" in ieee and "%22All+Metadata%22" in ieee


def test_native_like_exports_for_each_assisted_source(tmp_path):
    manager = service(tmp_path)
    run = manager.create("How does adaptive learning affect students?", QUERIES, 100)
    fixtures = {
        "scopus": pd.DataFrame([{
            "Authors": "A", "Title": "Scopus result", "Year": 2024,
            "Source title": "Journal", "EID": "2-s2.0-1", "DOI": "10.1/scopus",
            "Abstract": "Scopus evidence",
        }]),
        "web_of_science": pd.DataFrame([{
            "Authors": "B", "Article Title": "WoS result", "Publication Year": 2023,
            "Source Title": "Journal", "UT (Unique WOS ID)": "WOS:1",
            "DOI": "10.1/wos", "Abstract": "WoS evidence",
        }]),
        "ieee_xplore": pd.DataFrame([{
            "Authors": "C", "Document Title": "IEEE result", "Publication Year": 2022,
            "Publication Title": "Conference", "Article Citation Count": 2,
            "DOI": "10.1/ieee", "Abstract": "IEEE evidence",
        }]),
    }
    for source, frame in fixtures.items():
        path = tmp_path / f"{source}.csv"
        frame.to_csv(path, index=False)
        imported = manager.import_path(run.run_id, source, path)
        assert imported.sources[source].records == 1
        assert imported.sources[source].detected_format != "unknown"
        assert not imported.sources[source].warnings

    scholar = tmp_path / "scholar.ris"
    scholar.write_text(
        "TY  - JOUR\nTI  - Scholar result\nAU  - D\nPY  - 2021\nDO  - 10.1/scholar\nAB  - Scholar evidence\nER  -\n",
        encoding="utf-8",
    )
    imported = manager.import_path(run.run_id, "google_scholar", scholar)
    assert imported.sources["google_scholar"].detected_format == "ris"
    assert not imported.sources["google_scholar"].warnings


def test_assisted_launch_pauses_without_faking_collection(tmp_path):
    manager = service(tmp_path)
    run = manager.create("How does adaptive learning affect students?", QUERIES, 100)
    paused = manager.launch(run.run_id, "scopus")
    assert paused.sources["scopus"].status == "needs_attention"
    assert paused.sources["scopus"].records == 0
    public = manager.public(run.run_id)
    assert public["sources"]["scopus"]["export_steps"]
    assert public["sources"]["scopus"]["recommended_format"]


def test_automated_source_failure_is_retryable_and_preserves_search_url(tmp_path):
    export = tmp_path / "pubmed.csv"
    pd.DataFrame([{
        "PMID": "1", "Title": "Recovered paper", "Authors": "A",
        "Journal/Book": "Journal", "Publication Year": 2024, "DOI": "10.1/recovered",
    }]).to_csv(export, index=False)

    class Browser:
        def __init__(self):
            self.calls = 0

        async def collect(self, source, query, limit, artifact_dir):
            self.calls += 1
            if self.calls == 1:
                raise CollectionNeedsAttention("temporary page change")
            return export, "https://pubmed.ncbi.nlm.nih.gov/?term=recovered"

    manager = ExperimentalCollectionService(
        store=CollectionStore(tmp_path / "private" / "runs.sqlite3"),
        root=tmp_path / "outputs" / "experimental-collection",
        private_root=tmp_path / "private",
        browser=Browser(),
    )
    run = manager.create("How does adaptive learning affect students?", {"pubmed": QUERIES["pubmed"]}, 100)
    manager.launch(run.run_id, "pubmed")
    deadline = time.monotonic() + 3
    while manager.get(run.run_id).sources["pubmed"].status in {"ready", "running"} and time.monotonic() < deadline:
        time.sleep(0.01)
    failed = manager.get(run.run_id).sources["pubmed"]
    assert failed.status == "needs_attention"
    assert failed.records == 0
    assert failed.completed_at

    manager.launch(run.run_id, "pubmed")
    deadline = time.monotonic() + 3
    while manager.get(run.run_id).sources["pubmed"].status != "imported" and time.monotonic() < deadline:
        time.sleep(0.01)
    recovered = manager.get(run.run_id).sources["pubmed"]
    assert recovered.status == "imported"
    assert recovered.attempts == 2
    assert recovered.records == 1
    assert recovered.search_url.endswith("term=recovered")


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
    assert "Retrieve papers from database searches" in html
    assert "collection_run" in html
    assert "/experimental/collection-runs" in html
    assert html.index('id="panel-ls"') < html.index('id="experimentalCollector"') < html.index('id="panel-scr"')
    assert "Collect &amp; Deduplicate" in html
    assert "isolated from Gemini Web and Local AI screening" in html
    assert "switchTab('ls');\n            pollExperimentalCollection();" in html
    assert 'id="agenticWorkspace"' not in html
    assert "startAgenticRun()" not in html
    assert "/agentic-runs" not in html
