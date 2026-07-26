from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from litsync_app.screening.bulk import PROGRESS
from litsync_app.paper_collection.collectors import (
    CollectionNeedsAttention,
    DatabaseCollector,
    ExtractedPaper,
    ExtractedPaperBatch,
)
from litsync_app.paper_collection.models import (
    AgenticRun,
    CollectedPaper,
    DATABASES,
    RQCandidate,
    SourceCollection,
)
from litsync_app.paper_collection.orchestrator import AgenticWorkflowManager, deduplicate_collected
from litsync_app.paper_collection.rq_agents import RQAgentService
from litsync_app.paper_collection.store import AgenticRunStore
from litsync_app.prisma import PRISMA_STORE
from litsync_app import app as server


class FakeRQService:
    def generate_and_select(self, topic):
        candidates = [
            RQCandidate(
                question=f"How does {topic} affect evidence outcome {index}?",
                rationale=f"Candidate {index}",
                specificity=4,
                answerability=4,
                searchability=4,
                cross_database_suitability=4,
                evidence_availability=index,
                evidence_record_count=index,
                criticism="Searchable and answerable.",
            )
            for index in range(1, 6)
        ]
        return candidates, candidates[-1]


class FakeQueryBundle:
    def to_api_response(self):
        return {
            "status": "success",
            **{database: f"query for {database}" for database in DATABASES},
        }


class FakeCollector:
    def __init__(self):
        self.active = 0
        self.maximum_active = 0

    async def collect(self, database, query, private_root):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await asyncio.sleep(0.01)
        rank_one = CollectedPaper(
            title="Shared systematic review paper",
            authors=["A. Author"],
            year=2025,
            venue="Journal",
            abstract="Shared abstract",
            doi="https://doi.org/10.1000/shared",
            url="https://example.test/shared",
            cited_by=3,
            database=database,
            source_rank=1,
            query=query,
            provenance=[{"database": database, "source_rank": 1, "query": query}],
        )
        rank_two = CollectedPaper(
            title=f"Unique {database} paper",
            authors=["B. Author"],
            year=2024,
            venue="Proceedings",
            abstract="" if database == "pubmed" else "Usable abstract",
            doi=f"10.1000/{database}",
            url=f"https://example.test/{database}",
            database=database,
            source_rank=2,
            query=query,
            provenance=[{"database": database, "source_rank": 2, "query": query}],
        )
        self.active -= 1
        return SourceCollection(
            database=database,
            status="partial",
            records=[rank_one, rank_two],
            attempted_candidates=2,
            attempts=1,
        )


def fake_screener(**kwargs):
    frame = pd.read_csv(kwargs["csv_path"])
    job_id = kwargs["progress_job_id"]
    assert PROGRESS.start_job(job_id)
    PRISMA_STORE.configure_screening(
        job_id,
        input_rows=len(frame),
        missing_abstracts=0,
        records_available=len(frame),
        records_selected=len(frame),
    )
    PROGRESS.begin_screening(job_id, len(frame), "fake-agentic-screening")
    rows = []
    for index, row in frame.fillna("").iterrows():
        result = row.to_dict()
        result.update({
            "Source_Row_Index": index,
            "Decision": "REJECT",
            "Reason": "Fake decision",
            "Exclusion Reason": "Fake exclusion",
        })
        rows.append(result)
    pd.DataFrame(rows).to_csv(kwargs["output_path"], index=False)
    PROGRESS.update_counts(job_id, len(rows), 0, 0, len(rows))
    PROGRESS.finish(job_id)
    return {
        "architecture_version": "fake-agentic-screening",
        "output_file": kwargs["output_path"],
    }


def test_agentic_end_to_end_with_fake_collectors(tmp_path):
    store = AgenticRunStore(tmp_path / "private" / "runs.sqlite3")
    collector = FakeCollector()
    manager = AgenticWorkflowManager(
        store=store,
        rq_service=FakeRQService(),
        query_generator=lambda question: FakeQueryBundle(),
        collector=collector,
        output_root=tmp_path / "outputs",
        private_root=tmp_path / "private" / "artifacts",
        screener=fake_screener,
    )
    run = AgenticRun.create("agentic-test-run", "autonomous research agents")
    store.save(run)
    manager._execute(run.run_id)

    finished = manager.get(run.run_id)
    assert finished.status == "completed_partial"
    assert len(finished.rq_candidates) == 5
    assert finished.selected_rq.endswith("outcome 5?")
    assert set(finished.queries) == set(DATABASES)
    assert collector.maximum_active == 2
    assert finished.counts["identified"] == 10
    assert finished.counts["deduplicated"] == 6
    assert finished.counts["duplicates_removed"] == 4
    assert finished.counts["maybe"] == 1
    assert finished.counts["reject"] == 5
    clean = pd.read_csv(tmp_path / "outputs" / "agentic-runs" / run.run_id / "clean_library.csv")
    shared = clean.loc[clean["DOI"] == "https://doi.org/10.1000/shared"].iloc[0]
    assert all(database in shared["Source Databases"] for database in DATABASES)
    public = manager.public(run.run_id)
    assert all("records" not in source for source in public["sources"].values())
    assert public["sources"]["scopus"]["record_count"] == 2


def test_rq_agents_score_all_candidates_and_break_ties_by_original_order():
    class Result:
        def __init__(self, value):
            self.value = value

    class Engine:
        def __init__(self):
            self.calls = 0

        def generate(self, model, prompt, schema):
            self.calls += 1
            if self.calls == 1:
                return Result({"candidates": [
                    {
                        "question": f"How does agent workflow {index} affect review quality?",
                        "rationale": f"Reason {index}",
                    }
                    for index in range(1, 6)
                ]})
            return Result({"evaluations": [
                {
                    "candidate_index": index,
                    "specificity": 5 if index in {2, 3} else 3,
                    "answerability": 5 if index in {2, 3} else 3,
                    "searchability": 5 if index in {2, 3} else 3,
                    "cross_database_suitability": 5 if index in {2, 3} else 3,
                    "valid": True,
                    "criticism": "Valid.",
                }
                for index in range(1, 6)
            ]})

    class Grounder:
        def search(self, question):
            return [object(), object()]

    service = RQAgentService(engine=Engine(), model="fake", grounder=Grounder())
    candidates, selected = service.generate_and_select("agentic reviews")
    assert len(candidates) == 5
    assert all(candidate.evidence_availability == 2 for candidate in candidates)
    assert selected.question == "How does agent workflow 2 affect review quality?"


def test_deduplication_uses_doi_then_exact_title_with_compatible_year():
    def paper(database, rank, *, title, doi="", year=2025):
        return CollectedPaper(
            title=title,
            year=year,
            database=database,
            source_rank=rank,
            query="q",
            doi=doi,
            provenance=[{"database": database, "source_rank": rank}],
        )

    records = [
        paper("scopus", 1, title="Paper A", doi="10.1/A"),
        paper("pubmed", 1, title="Different title", doi="https://doi.org/10.1/a"),
        paper("ieee_xplore", 2, title="Exact: Paper B", year=2024),
        paper("web_of_science", 3, title="Exact Paper B", year=2024),
        paper("google_scholar", 4, title="Exact Paper B", year=2023),
    ]
    result = deduplicate_collected(records)
    assert len(result) == 3
    assert len(result[0].provenance) == 2
    assert len(result[1].provenance) == 2


class FakeCloudClient:
    def __init__(self, *, blocked=False):
        self.blocked = blocked
        self.calls = 0

    async def collect(self, database, query, artifact_dir):
        self.calls += 1
        if self.blocked:
            raise CollectionNeedsAttention(
                "MFA required", live_url="https://live.example/session", run_id="pbs_1"
            )
        papers = [
            ExtractedPaper(title=f"Paper {index}", source_rank=index)
            for index in range(1, 13)
        ]
        return ExtractedPaperBatch(papers=papers, attempted_candidates=20), {
            "skyvern_run_id": "pbs_2", "live_url": ""
        }


def test_collector_caps_at_ten_and_marks_blockers(tmp_path):
    completed = asyncio.run(DatabaseCollector(FakeCloudClient()).collect(
        "scopus", "TITLE-ABS-KEY(test)", tmp_path
    ))
    assert completed.status == "completed"
    assert len(completed.records) == 10
    assert completed.attempted_candidates == 20

    blocked = asyncio.run(DatabaseCollector(FakeCloudClient(blocked=True)).collect(
        "scopus", "TITLE-ABS-KEY(test)", tmp_path
    ))
    assert blocked.status == "needs_attention"
    assert blocked.skyvern_run_id == "pbs_1"
    assert blocked.live_url == "https://live.example/session"


def test_public_store_redacts_skyvern_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("SKYVERN_API_KEY", "skyvern-private-value")
    store = AgenticRunStore(tmp_path / "runs.sqlite3")
    run = AgenticRun.create("redaction", "a topic")
    run.error = "request failed with skyvern-private-value"
    run.sources["scopus"].blocker = "credential skyvern-private-value failed"
    store.save(run)
    payload = store.public_payload(run.run_id)
    assert "skyvern-private-value" not in str(payload)
    assert "[REDACTED]" in str(payload)


def test_agentic_dashboard_is_default_and_manual_workflow_remains_available():
    html = Path("web/slr_query_generator.html").read_text(encoding="utf-8")
    assert 'id="agenticWorkspace"' in html
    assert "Skyvern Evidence Gathering" in html
    assert "Advanced / Manual Mode" in html
    assert 'id="manualWorkspace"' in html
    assert "fetch('/agentic-runs'" in html
    assert "LitSync Manual" in html


def test_agentic_api_contract(monkeypatch):
    run = AgenticRun.create("api-run", "autonomous evidence agents")

    class FakeManager:
        def create(self, topic):
            assert topic == "autonomous evidence agents"
            return run

        def public(self, run_id):
            assert run_id == run.run_id
            return run.model_dump()

        def resume(self, run_id):
            return run

        def skip_source(self, run_id, database):
            assert database == "scopus"
            return run

        def cancel(self, run_id):
            run.status = "cancelled"
            return run

    monkeypatch.setattr(server, "AGENTIC_WORKFLOWS", FakeManager())
    client = TestClient(server.app)
    created = client.post("/agentic-runs", json={"topic": "autonomous evidence agents"})
    assert created.status_code == 202
    assert created.json()["run_id"] == "api-run"
    assert client.get("/agentic-runs/api-run").status_code == 200
    assert client.post("/agentic-runs/api-run/resume").status_code == 202
    assert client.post("/agentic-runs/api-run/sources/scopus/skip").status_code == 202
    cancelled = client.post("/agentic-runs/api-run/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
