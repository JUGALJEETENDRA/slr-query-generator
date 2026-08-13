from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from hashlib import sha256
from pathlib import Path
from threading import Lock, Thread

import pandas as pd

from litsync_app.deduplication import deduplicate
from litsync_app.prisma import PRISMA_STORE

from .browser_collectors import (
    CollectionNeedsAttention, NativeExportBrowser, SOURCE_INFO, source_launch_url,
)
from .importers import (
    SUPPORTED_SUFFIXES, detect_export_format, normalize_export, read_export, source_warnings,
)
from .models import CollectionRun, SOURCES, SourceState, utc_now
from .store import CollectionStore


class ExperimentalCollectionService:
    def __init__(self, *, store=None, root="outputs/experimental-collection", private_root="private/experimental_collection", browser=None):
        self.store = store or CollectionStore(Path(private_root) / "runs.sqlite3")
        self.root = Path(root)
        self.private_root = Path(private_root)
        self.browser = browser or NativeExportBrowser(Path(private_root) / "browser-profile")
        self._threads: dict[str, Thread] = {}
        self._lock = Lock()

    def create(self, research_question: str, queries: dict[str, str], limit: int = 100) -> CollectionRun:
        question = research_question.strip()
        if len(question) < 3:
            raise ValueError("Enter the research question used to create these queries")
        clean = {key: str(value or "").strip() for key, value in queries.items() if key in SOURCES}
        if not clean:
            raise ValueError("No supported database queries were supplied")
        run_id = str(uuid.uuid4())
        sources = {
            key: SourceState(source=key, query=value, mode=SOURCE_INFO[key]["mode"])
            for key, value in clean.items() if value
        }
        run = CollectionRun(run_id=run_id, research_question=question, limit=limit, sources=sources)
        directory = self.private_root / "runs" / run_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "requested_queries.json").write_text(json.dumps({
            "research_question": question, "limit": limit, "queries": clean,
            "created_at": run.created_at,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.store.save(run)

    def get(self, run_id: str) -> CollectionRun:
        run = self.store.get(run_id)
        if run is None:
            raise KeyError(run_id)
        return run

    def public(self, run_id: str) -> dict:
        run = self.get(run_id)
        payload = run.model_dump()
        for key, state in payload["sources"].items():
            state.update(SOURCE_INFO[key])
            state["url"] = source_launch_url(key, state["query"])
        return payload

    def launch(self, run_id: str, source: str) -> CollectionRun:
        run = self.get(run_id)
        if source not in run.sources:
            raise ValueError("Source is not part of this run")
        if run.sources[source].mode == "assisted":
            run.sources[source].status = "needs_attention"
            run.sources[source].message = SOURCE_INFO[source]["reason"]
            return self.store.save(run)
        key = f"{run_id}:{source}"
        with self._lock:
            active = self._threads.get(key)
            if active and active.is_alive():
                return run
            thread = Thread(target=self._collect_safe, args=(run_id, source), daemon=True)
            self._threads[key] = thread
            thread.start()
        return run

    def _collect_safe(self, run_id: str, source: str) -> None:
        run = self.get(run_id)
        state = run.sources[source]
        state.status = "running"
        state.attempts += 1
        state.started_at = utc_now()
        run.status = "collecting"
        self.store.save(run)
        try:
            artifact_dir = self.private_root / "runs" / run_id / source
            path, search_url = asyncio.run(self.browser.collect(source, state.query, run.limit, artifact_dir))
            checkpoint = self.get(run_id)
            checkpoint.sources[source].search_url = search_url
            self.store.save(checkpoint)
            self.import_path(run_id, source, path, already_private=True)
        except CollectionNeedsAttention as exc:
            run = self.get(run_id)
            run.sources[source].status = "needs_attention"
            run.sources[source].message = str(exc)
            run.sources[source].completed_at = utc_now()
            run.status = "needs_attention"
            self.store.save(run)
        except Exception as exc:
            run = self.get(run_id)
            run.sources[source].status = "failed"
            run.sources[source].message = str(exc)
            run.sources[source].completed_at = utc_now()
            run.status = "needs_attention"
            self.store.save(run)

    def import_path(self, run_id: str, source: str, path: Path, *, already_private=False) -> CollectionRun:
        run = self.get(run_id)
        if source not in run.sources:
            raise ValueError("Source is not part of this run")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError("Upload CSV, Excel, RIS, NBIB, or PubMed text")
        directory = self.private_root / "runs" / run_id / source
        directory.mkdir(parents=True, exist_ok=True)
        safe_name = Path(path.name).name
        content_digest = sha256(path.read_bytes()).hexdigest()
        raw = path if already_private else directory / f"raw-{content_digest[:12]}-{safe_name}"
        if not already_private:
            shutil.copy2(path, raw)
        digest = sha256(raw.read_bytes()).hexdigest()
        frame = read_export(raw)
        detected = detect_export_format(raw, frame)
        warnings = source_warnings(source, detected, frame)
        normalized = normalize_export(raw, source).head(run.limit)
        normalized_path = directory / "normalized.csv"
        normalized.to_csv(normalized_path, index=False)
        state = run.sources[source]
        state.status = "imported"
        state.records = len(normalized)
        state.raw_filename = safe_name
        state.raw_sha256 = digest
        state.detected_format = detected
        state.warnings = warnings
        state.completed_at = utc_now()
        state.message = f"Validated {len(normalized)} records" + (
            f" with {len(warnings)} warning(s)" if warnings else ""
        )
        run.status = "ready"
        return self.store.save(run)

    def skip(self, run_id: str, source: str) -> CollectionRun:
        run = self.get(run_id)
        if source not in run.sources:
            raise ValueError("Source is not part of this run")
        run.sources[source].status = "skipped"
        run.sources[source].completed_at = utc_now()
        return self.store.save(run)

    def finalize(self, run_id: str) -> CollectionRun:
        run = self.get(run_id)
        imported = [key for key, state in run.sources.items() if state.status == "imported"]
        if not imported:
            raise ValueError("Import at least one source export before finalizing")
        frames = []
        for source in imported:
            path = self.private_root / "runs" / run_id / source / "normalized.csv"
            frame = pd.read_csv(path).fillna("")
            frame["Collection Query"] = run.sources[source].query
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        mapped_columns = [column for column in combined.columns if column not in {"Collection Source", "Collection Query"}]
        clean, removed = deduplicate(combined[mapped_columns])
        provenance_sources = []
        provenance_queries = []
        combined_doi = combined["DOI"].fillna("").astype(str).str.strip().str.lower()
        combined_title = combined["Title"].fillna("").astype(str).str.strip().str.lower()
        for _, paper in clean.iterrows():
            doi = str(paper.get("DOI") or "").strip().lower()
            title = str(paper.get("Title") or "").strip().lower()
            matches = combined.loc[combined_doi.eq(doi)] if doi else combined.loc[combined_title.eq(title)]
            sources = list(dict.fromkeys(matches["Collection Source"].astype(str)))
            query_map = {
                str(row["Collection Source"]): str(row["Collection Query"])
                for _, row in matches.iterrows()
            }
            provenance_sources.append("; ".join(sources))
            provenance_queries.append(json.dumps(query_map, ensure_ascii=False, sort_keys=True))
        clean["Collection Sources"] = provenance_sources
        clean["Collection Queries JSON"] = provenance_queries
        output = self.root / run_id
        output.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output / "combined_with_provenance.csv", index=False)
        clean_path = output / "clean_dataset.csv"
        clean.to_csv(clean_path, index=False)
        run.counts = {"identified": len(combined), "deduplicated": len(clean), "duplicates_removed": removed}
        manifest = {
            "run_id": run.run_id, "research_question": run.research_question,
            "limit_per_source": run.limit, "created_at": run.created_at,
            "completed_at": utc_now(), "counts": run.counts,
            "sources": {key: state.model_dump() for key, state in run.sources.items()},
        }
        (output / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        import_id = f"experimental-{run_id}"
        PRISMA_STORE.create_import(
            output_root=self.root.parent,
            import_id=import_id,
            records_identified=len(combined),
            duplicate_records_removed=removed,
            source_files=[
                {"name": run.sources[key].raw_filename or key, "records": run.sources[key].records}
                for key in imported
            ],
            clean_fingerprint=sha256(clean_path.read_bytes()).hexdigest(),
            clean_path=str(clean_path),
        )
        run.artifacts = {
            "clean_dataset": f"/outputs/experimental-collection/{run_id}/clean_dataset.csv",
            "combined_with_provenance": f"/outputs/experimental-collection/{run_id}/combined_with_provenance.csv",
            "audit_manifest": f"/outputs/experimental-collection/{run_id}/audit_manifest.json",
            "prisma_json": f"/prisma/{import_id}",
            "prisma_csv": f"/prisma/{import_id}.csv",
            "prisma_svg": f"/prisma/{import_id}.svg",
        }
        run.status = "completed"
        return self.store.save(run)
