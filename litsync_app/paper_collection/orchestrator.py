from __future__ import annotations

import asyncio
import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from threading import Lock, Thread, current_thread
from typing import Any, Callable

import pandas as pd

from litsync_app.screening.bulk import PROGRESS, SCREENING_SESSION, screen_csv
from litsync_app.query.generator import generate_query_bundle
from litsync_app.prisma import PRISMA_STORE

from .collectors import DatabaseCollector
from .models import AgenticRun, CollectedPaper, DATABASES, utc_now
from .rq_agents import RQAgentService
from .store import AgenticRunStore


STAGES = (
    "topic_intake", "rq_proposal", "rq_critique", "query_generation",
    "skyvern_collection", "normalization", "deduplication", "screening", "publishing",
)
TERMINAL_SOURCE_STATES = {"completed", "partial", "failed", "skipped"}


def _normalize_doi(value: str) -> str:
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", str(value or "").strip(), flags=re.I)
    return value.lower().strip()


def _normalize_title(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _years_compatible(left: int | None, right: int | None) -> bool:
    return left is None or right is None or left == right


def deduplicate_collected(records: list[CollectedPaper]) -> list[CollectedPaper]:
    deduped: list[CollectedPaper] = []
    doi_index: dict[str, int] = {}
    title_index: dict[str, list[int]] = {}
    for paper in records:
        doi = _normalize_doi(paper.doi)
        title = _normalize_title(paper.title)
        match = doi_index.get(doi) if doi else None
        if match is None and title:
            for candidate in title_index.get(title, []):
                if _years_compatible(deduped[candidate].year, paper.year):
                    match = candidate
                    break
        if match is None:
            clone = paper.model_copy(deep=True)
            clone.provenance = list(clone.provenance or [{
                "database": clone.database, "source_rank": clone.source_rank,
                "query": clone.query, "retrieved_at": clone.retrieved_at,
            }])
            deduped.append(clone)
            index = len(deduped) - 1
            if doi:
                doi_index[doi] = index
            if title:
                title_index.setdefault(title, []).append(index)
            continue
        current = deduped[match]
        current.provenance.extend(paper.provenance or [{
            "database": paper.database, "source_rank": paper.source_rank,
            "query": paper.query, "retrieved_at": paper.retrieved_at,
        }])
        if not current.abstract and paper.abstract:
            current.abstract = paper.abstract
        if not current.doi and paper.doi:
            current.doi = paper.doi
            doi_index[_normalize_doi(paper.doi)] = match
        if not current.url and paper.url:
            current.url = paper.url
    return deduped


class AgenticWorkflowManager:
    def __init__(
        self,
        *,
        store: AgenticRunStore | None = None,
        rq_service=None,
        query_generator: Callable[[str], Any] = generate_query_bundle,
        collector=None,
        output_root: str | Path = "outputs",
        private_root: str | Path = "private/agentic_runs",
        screener: Callable[..., dict[str, Any]] = screen_csv,
    ):
        self.store = store or AgenticRunStore()
        self.rq_service = rq_service
        self.query_generator = query_generator
        self.collector = collector or DatabaseCollector()
        self.output_root = Path(output_root)
        self.private_root = Path(private_root)
        self.screener = screener
        self._threads: dict[str, Thread] = {}
        self._lock = Lock()

    def create(self, topic: str) -> AgenticRun:
        if self.store.has_active():
            raise RuntimeError(
                "Another agentic run is active. Complete, skip, or cancel it before starting a new run."
            )
        run = AgenticRun.create(str(uuid.uuid4()), topic)
        self.store.save(run)
        self.start(run.run_id)
        return run

    def start(self, run_id: str) -> None:
        with self._lock:
            active = self._threads.get(run_id)
            if active and active.is_alive():
                return
            thread = Thread(target=self._execute_safe, args=(run_id,), daemon=True)
            self._threads[run_id] = thread
            thread.start()

    def recover(self) -> None:
        for run in self.store.recoverable():
            self.start(run.run_id)

    def resume(self, run_id: str) -> AgenticRun:
        run = self._require(run_id)
        if run.cancelled:
            raise ValueError("cancelled runs cannot be resumed")
        for source in run.sources.values():
            if source.status in {"needs_attention", "failed", "running"}:
                source.status = "pending"
                source.blocker = ""
                source.error = ""
        run.status = "queued"
        run.error = ""
        run.stage = "skyvern_collection" if run.selected_rq else "topic_intake"
        self.store.save(run)
        self.start(run_id)
        return run

    def skip_source(self, run_id: str, database: str) -> AgenticRun:
        run = self._require(run_id)
        if database not in DATABASES:
            raise ValueError(f"unsupported database: {database}")
        source = run.sources[database]
        if source.status not in {"needs_attention", "failed", "pending"}:
            raise ValueError("only blocked, failed, or pending sources can be skipped")
        source.status = "skipped"
        source.completed_at = utc_now()
        source.blocker = ""
        run.status = "queued"
        self.store.save(run)
        self.start(run_id)
        return run

    def cancel(self, run_id: str) -> AgenticRun:
        run = self._require(run_id)
        run.cancelled = True
        run.status = "cancelled"
        self.store.save(run)
        return run

    def get(self, run_id: str) -> AgenticRun:
        return self._require(run_id)

    def has_active(self) -> bool:
        return self.store.has_active()

    def public(self, run_id: str) -> dict[str, Any]:
        payload = self.store.public_payload(run_id)
        screening_job = payload.get("screening", {}).get("job_id")
        if screening_job:
            progress = PROGRESS.snapshot(screening_job)
            if progress:
                payload["screening"]["progress"] = progress
        return payload

    def _require(self, run_id: str) -> AgenticRun:
        run = self.store.get(run_id)
        if run is None:
            raise KeyError(run_id)
        return run

    def _execute_safe(self, run_id: str) -> None:
        try:
            self._execute(run_id)
        except Exception as exc:
            run = self._require(run_id)
            if not run.cancelled:
                run.status = "failed"
                run.error = str(exc)
                self.store.save(run)
        finally:
            restart = False
            with self._lock:
                if self._threads.get(run_id) is current_thread():
                    self._threads.pop(run_id, None)
                latest = self.store.get(run_id)
                restart = bool(
                    latest and latest.status == "queued" and not latest.cancelled
                )
            if restart:
                self.start(run_id)

    def _execute(self, run_id: str) -> None:
        run = self._require(run_id)
        if run.cancelled:
            return
        run.status = "running"
        self.store.save(run)
        if not run.selected_rq:
            run.stage = "rq_proposal"
            self.store.save(run)
            service = self.rq_service or RQAgentService()
            candidates, selected = service.generate_and_select(run.topic)
            run.rq_candidates = candidates
            run.stage = "rq_critique"
            run.selected_rq = selected.question
            run.selection_rationale = (
                f"Selected the highest valid score ({selected.total_score}/25). "
                f"{selected.rationale} {selected.criticism}"
            ).strip()
            self.store.save(run)
        if not run.queries:
            run.stage = "query_generation"
            self.store.save(run)
            bundle = self.query_generator(run.selected_rq).to_api_response()
            queries = {database: str(bundle.get(database) or "").strip() for database in DATABASES}
            missing = [database for database, query in queries.items() if not query]
            if missing or str(bundle.get("status", "success")).lower() != "success":
                raise ValueError("validated queries unavailable for: " + ", ".join(missing))
            run.queries = queries
            self.store.save(run)
        run.stage = "skyvern_collection"
        self.store.save(run)
        self._collect(run)
        run = self._require(run_id)
        if run.cancelled:
            return
        blocked = [
            source for source in run.sources.values()
            if source.status == "needs_attention"
        ]
        pending = [
            source for source in run.sources.values()
            if source.status not in TERMINAL_SOURCE_STATES | {"needs_attention"}
        ]
        if pending:
            raise RuntimeError("collection ended with unfinished source checkpoints")
        if blocked:
            run.status = "needs_attention"
            self.store.save(run)
            return
        records = [
            paper
            for database in DATABASES
            for paper in run.sources[database].records
        ]
        if not records:
            raise ValueError("no papers were collected from the selected databases")
        run.stage = "normalization"
        self.store.save(run)
        deduped = deduplicate_collected(records)
        run.stage = "deduplication"
        run.counts = {
            "identified": len(records),
            "deduplicated": len(deduped),
            "duplicates_removed": len(records) - len(deduped),
        }
        self.store.save(run)
        paths = self._write_library(run, deduped)
        run.artifacts.update(paths)
        self.store.save(run)
        self._screen(run, deduped)

    def _collect(self, run: AgenticRun) -> None:
        databases = [
            database for database in DATABASES
            if run.sources[database].status not in TERMINAL_SOURCE_STATES
        ]

        async def execute_all():
            semaphore = asyncio.Semaphore(2)

            async def execute_one(database: str):
                async with semaphore:
                    current = self._require(run.run_id)
                    if current.cancelled:
                        return database, current.sources[database]
                    private_dir = self.private_root / run.run_id
                    result = await self.collector.collect(
                        database, run.queries[database], private_dir
                    )
                    checkpoint = self._require(run.run_id)
                    checkpoint.sources[database] = result
                    self.store.save(checkpoint)
                    return database, result

            return await asyncio.gather(*(execute_one(database) for database in databases))

        results = asyncio.run(execute_all()) if databases else []
        # Each source is saved as soon as it completes inside execute_one.

    def _write_library(
        self, run: AgenticRun, deduped: list[CollectedPaper]
    ) -> dict[str, str]:
        directory = self.output_root / "agentic-runs" / run.run_id
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for paper in deduped:
            sources = [item["database"] for item in paper.provenance]
            ranks = {
                item["database"]: item["source_rank"] for item in paper.provenance
            }
            rows.append({
                "Authors": "; ".join(paper.authors),
                "Title": paper.title,
                "Year": paper.year or "",
                "Source title": paper.venue,
                "Cited by": paper.cited_by if paper.cited_by is not None else "",
                "DOI": paper.doi,
                "Link": paper.url,
                "Abstract": paper.abstract or "[Abstract unavailable from database record]",
                "Abstract Missing": not bool(paper.abstract.strip()),
                "Source Databases": "; ".join(dict.fromkeys(sources)),
                "Source Ranks JSON": json.dumps(ranks, sort_keys=True),
                "Provenance JSON": json.dumps(paper.provenance, ensure_ascii=False),
            })
        clean_path = directory / "clean_library.csv"
        pd.DataFrame(rows).to_csv(clean_path, index=False)
        source_path = directory / "source_summary.json"
        source_path.write_text(json.dumps({
            database: {
                "status": source.status,
                "records": len(source.records),
                "attempted_candidates": source.attempted_candidates,
                "attempts": source.attempts,
            }
            for database, source in run.sources.items()
        }, indent=2), encoding="utf-8")
        import_id = f"{run.run_id}-import"
        PRISMA_STORE.create_import(
            output_root=self.output_root,
            import_id=import_id,
            records_identified=run.counts["identified"],
            duplicate_records_removed=run.counts["duplicates_removed"],
            source_files=[
                {"name": database, "records": len(source.records)}
                for database, source in run.sources.items()
            ],
            clean_fingerprint=sha256(clean_path.read_bytes()).hexdigest(),
            clean_path=str(clean_path),
        )
        return {
            "clean_library": f"/outputs/agentic-runs/{run.run_id}/clean_library.csv",
            "source_summary": f"/outputs/agentic-runs/{run.run_id}/source_summary.json",
            "import_id": import_id,
        }

    def _screen(self, run: AgenticRun, deduped: list[CollectedPaper]) -> None:
        run.stage = "screening"
        output_path = self.output_root / "agentic-runs" / run.run_id / "screened.csv"
        clean_path = self.output_root / "agentic-runs" / run.run_id / "clean_library.csv"
        fingerprint = sha256(clean_path.read_bytes()).hexdigest()
        SCREENING_SESSION.begin(run.run_id, str(output_path), "agentic-local")
        PRISMA_STORE.begin_screening(
            output_root=self.output_root,
            job_id=run.run_id,
            input_fingerprint=fingerprint,
            screening_engine="local",
            import_id=run.artifacts["import_id"],
        )
        run.screening = {"job_id": run.run_id, "status": "running"}
        self.store.save(run)
        summary = self.screener(
            csv_path=str(clean_path),
            research_question=run.selected_rq,
            output_path=str(output_path),
            progress_job_id=run.run_id,
            screening_engine="local",
            max_rows=None,
            resume=True,
            input_fingerprint=fingerprint,
        )
        rows = pd.read_csv(output_path).fillna("").to_dict(orient="records")
        for row in rows:
            missing = str(row.get("Abstract Missing", "")).strip().lower()
            if missing in {"true", "1", "yes"}:
                row["Original_Decision"] = str(row.get("Decision") or "MAYBE").upper()
                row["Decision"] = "MAYBE"
                row["Reason"] = "Abstract unavailable; title alone is insufficient for a final decision."
                row["Exclusion_Reason"] = ""
                row["Exclusion Reason"] = ""
                row["Decision_Source"] = "agentic_missing_abstract_policy"
        pd.DataFrame(rows).to_csv(output_path, index=False)
        SCREENING_SESSION.set_results(
            rows, job_id=run.run_id, output_path=str(output_path),
            architecture_version=str(summary.get("architecture_version") or "agentic-local"),
        )
        counts = SCREENING_SESSION.counts(rows)
        PROGRESS.update_counts(
            run.run_id, len(rows), counts["keep"], counts["maybe"], counts["reject"]
        )
        run = self._require(run.run_id)
        run.stage = "publishing"
        run.screening = {
            "job_id": run.run_id,
            "status": "finished",
            "counts": counts,
            "architecture_version": summary.get("architecture_version"),
        }
        run.counts.update({
            "screened": len(rows),
            "keep": counts["keep"],
            "maybe": counts["maybe"],
            "reject": counts["reject"],
        })
        run.artifacts.update({
            "screened": f"/outputs/agentic-runs/{run.run_id}/screened.csv",
            "prisma_json": f"/prisma/{run.run_id}",
            "prisma_csv": f"/prisma/{run.run_id}.csv",
            "prisma_svg": f"/prisma/{run.run_id}.svg",
        })
        partial = any(
            source.status in {"partial", "failed", "skipped"}
            for source in run.sources.values()
        )
        run.status = "completed_partial" if partial else "completed"
        self.store.save(run)
