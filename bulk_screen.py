from __future__ import annotations

import json
import os
import time
import uuid
from hashlib import sha256
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd

from config import DEV_SCREENING_ROW_LIMIT, LOCAL_CHECKPOINT_INTERVAL
from local_ai.contracts import SCHEMA_VERSION
from local_ai.hardware import resolve_runtime_profile
from local_ai.three_layer import (
    DEEP_MODEL,
    DEEP_BATCH_SIZE,
    EDGE_MODEL,
    EDGE_BATCH_SIZE,
    THREE_LAYER_PROMPT_VERSION,
    TRIAGE_BATCH_SIZE,
    TRIAGE_MODEL,
    ThreeLayerLocalOrchestrator,
)
from processing_engines import LOCAL_ENGINE, normalize_processing_engine, resolve_processing_engine


class ScreeningProgress:
    def __init__(self):
        self._lock = Lock()
        self._state = self._idle_state()
        self._started_at: float | None = None

    @staticmethod
    def _idle_state():
        return {
            "status": "idle", "phase": "idle", "job_id": None,
            "current": 0, "total": 0, "stage2_current": 0, "stage2_total": 0,
            "keep": 0, "maybe": 0, "reject": 0, "error": None,
            "resumed_count": 0, "architecture_version": None,
            "runtime_seconds": None, "remaining": 0,
            "estimated_remaining_seconds": None,
            "batch_current": 0, "batch_total": 0, "batch_size": 0,
            "retry_count": 0,
        }

    def start_job(self, job_id):
        with self._lock:
            if self._state["status"] in {"starting", "running"}:
                return self._state["job_id"] == job_id
            self._state = self._idle_state()
            self._started_at = None
            self._state.update(status="starting", job_id=job_id)
            return True

    def begin_screening(self, job_id, total, architecture_version=None):
        with self._lock:
            self._assert_job(job_id)
            self._state.update(
                status="running", phase="fast_assessment", current=0, total=int(total),
                stage2_current=0, stage2_total=0, keep=0, maybe=0, reject=0, error=None,
                resumed_count=0, architecture_version=architecture_version,
            )
            self._started_at = time.perf_counter()

    def set_resumed_count(self, job_id, count):
        with self._lock:
            self._assert_job(job_id)
            self._state["resumed_count"] = int(count)

    def update_counts(self, job_id, current, keep, maybe, reject):
        with self._lock:
            self._assert_job(job_id)
            self._state.update(
                current=int(current), keep=int(keep), maybe=int(maybe), reject=int(reject)
            )
            self._update_timing()

    def begin_stage2(self, job_id, total):
        with self._lock:
            self._assert_job(job_id)
            self._state.update(phase="validation_repair", stage2_current=0, stage2_total=int(total))

    def begin_secondary(self, job_id, phase, total):
        with self._lock:
            self._assert_job(job_id)
            self._state.update(phase=str(phase), stage2_current=0, stage2_total=int(total))

    def begin_batches(self, job_id, phase, papers, batches, batch_size):
        with self._lock:
            self._assert_job(job_id)
            self._state.update(
                phase=str(phase), stage2_current=0, stage2_total=int(papers),
                batch_current=0, batch_total=int(batches), batch_size=int(batch_size),
                retry_count=0,
            )

    def update_batch(self, job_id, batch_current, paper_current=None):
        with self._lock:
            self._assert_job(job_id)
            self._state["batch_current"] = int(batch_current)
            self._state["batch_total"] = max(
                int(self._state.get("batch_total") or 0), int(batch_current)
            )
            if paper_current is not None:
                self._state["stage2_current"] = int(paper_current)
                if self._state.get("phase") == "batched_triage":
                    self._state["current"] = int(self._state.get("resumed_count") or 0) + int(paper_current)
            self._update_timing()

    def record_retry(self, job_id):
        with self._lock:
            self._assert_job(job_id)
            self._state["retry_count"] = int(self._state.get("retry_count") or 0) + 1
            self._update_timing()

    def update_stage2(self, job_id, current):
        with self._lock:
            self._assert_job(job_id)
            self._state["stage2_current"] = int(current)
            self._update_timing()

    def update_secondary(self, job_id, current):
        self.update_stage2(job_id, current)

    def finish(self, job_id):
        with self._lock:
            self._assert_job(job_id)
            self._state.update(status="finished", phase="finished", current=self._state["total"])
            self._update_timing()
            self._state.update(remaining=0, estimated_remaining_seconds=0.0)
            self._started_at = None

    def fail(self, job_id, message):
        with self._lock:
            if self._state.get("job_id") != job_id:
                return
            self._state.update(status="error", phase="error", error=str(message))
            self._update_timing()
            self._started_at = None

    def snapshot(self, job_id=None):
        with self._lock:
            self._update_timing()
            if job_id is not None and self._state.get("job_id") != job_id:
                return None
            return dict(self._state)

    def is_running(self):
        with self._lock:
            return self._state["status"] in {"starting", "running"}

    def _assert_job(self, job_id):
        if self._state.get("job_id") != job_id:
            raise RuntimeError(f"inactive screening job: {job_id}")

    def _update_timing(self):
        if self._started_at is None:
            return
        elapsed = time.perf_counter() - self._started_at
        if str(self._state.get("phase", "")).startswith("batched_"):
            current = int(self._state.get("stage2_current") or 0)
            total = int(self._state.get("stage2_total") or 0)
        else:
            current = int(self._state.get("current") or 0)
            total = int(self._state.get("total") or 0)
        remaining = max(0, total - current)
        self._state["runtime_seconds"] = round(elapsed, 2)
        self._state["remaining"] = remaining
        self._state["estimated_remaining_seconds"] = (
            round(elapsed / current * remaining, 2) if current else None
        )


class ScreeningSession:
    def __init__(self):
        self._lock = Lock()
        self._results: list[dict[str, Any]] = []
        self._job_id: str | None = None
        self._output_path: str | None = None
        self._architecture_version: str | None = None

    def begin(self, job_id, output_path=None, architecture_version=None):
        with self._lock:
            self._results = []
            self._job_id = str(job_id)
            self._output_path = output_path
            self._architecture_version = architecture_version

    def set_results(self, results, job_id=None, output_path=None, architecture_version=None):
        with self._lock:
            if job_id is not None and self._job_id not in {None, str(job_id)}:
                raise RuntimeError(f"inactive screening session: {job_id}")
            self._results = [dict(row) for row in results]
            if job_id is not None:
                self._job_id = str(job_id)
            if output_path is not None:
                self._output_path = str(output_path)
            if architecture_version is not None:
                self._architecture_version = str(architecture_version)

    def snapshot(self, job_id=None):
        with self._lock:
            if job_id is not None and self._job_id != str(job_id):
                return []
            return [dict(row) for row in self._results]

    def metadata(self):
        with self._lock:
            return {
                "job_id": self._job_id,
                "output_path": self._output_path,
                "architecture_version": self._architecture_version,
            }

    def counts(self, results=None):
        rows = self.snapshot() if results is None else results
        return {
            "total": len(rows),
            "keep": sum(row.get("Decision") == "KEEP" for row in rows),
            "maybe": sum(row.get("Decision") == "MAYBE" for row in rows),
            "reject": sum(row.get("Decision") == "REJECT" for row in rows),
        }

    def finalize(self, edited_results, output_dir="outputs"):
        rows = [dict(row) for row in edited_results if row.get("Title")]
        if not rows:
            raise RuntimeError("No edited screening results were provided.")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.set_results(rows)
        frame = pd.DataFrame(rows)
        paths = {
            "screened": os.path.join(output_dir, "screened.csv"),
            "included": os.path.join(output_dir, "included_studies.csv"),
            "maybe": os.path.join(output_dir, "maybe_studies.csv"),
            "excluded": os.path.join(output_dir, "excluded_studies.csv"),
        }
        frame.to_csv(paths["screened"], index=False)
        frame[frame["Decision"] == "KEEP"].to_csv(paths["included"], index=False)
        frame[frame["Decision"] == "MAYBE"].to_csv(paths["maybe"], index=False)
        frame[frame["Decision"] == "REJECT"].to_csv(paths["excluded"], index=False)
        return {"counts": self.counts(rows), "files": paths}


PROGRESS = ScreeningProgress()
SCREENING_SESSION = ScreeningSession()


def _find_col(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    return next((lower[name.lower()] for name in candidates if name.lower() in lower), None)


def _row_from_result(source: dict[str, Any], title: str, abstract: str, result: dict[str, Any], source_index: Any):
    row = dict(source)
    row.update({
        "Title": title,
        "Abstract": abstract,
        "Decision": result["decision"],
        "Reason": result["reason"],
        "Confidence": result["confidence"],
        "Protocol_ID": result.get("protocol_id", ""),
        "Evidence_JSON": json.dumps(result.get("evidence", []), ensure_ascii=False),
        "Criteria_JSON": json.dumps(result.get("criteria", []), ensure_ascii=False),
        "Uncertainty_JSON": json.dumps(result.get("uncertainty", []), ensure_ascii=False),
        "Escalated": bool(result.get("escalated")),
        "Validation_Status": result.get("validation_status", "unresolved"),
        "Validation_Errors": json.dumps(result.get("validation_errors", []), ensure_ascii=False),
        "Schema_Version": result.get("schema_version", SCHEMA_VERSION),
        "Model_Tier": result.get("model_tier", ""),
        "Resource_Profile": result.get("resource_profile", ""),
        "Model": result.get("model", ""),
        "Prompt_Version": result.get("prompt_version", ""),
        "Processing_Seconds": result.get("processing_seconds", 0.0),
        "Original_Processing_Seconds": result.get("original_processing_seconds", 0.0),
        "Cache_Hit": bool(result.get("cache_hit")),
        "Runtime_Downgrades": json.dumps(result.get("runtime_downgrades", []), ensure_ascii=False),
        "Layer_Trace_JSON": json.dumps(result.get("layer_trace", []), ensure_ascii=False),
        "Layer_Metrics_JSON": json.dumps(result.get("layer_metrics", []), ensure_ascii=False),
        "Decision_Risk": result.get("decision_risk", ""),
        "Triage_Basis": result.get("triage_basis", ""),
        "Source_Row_Index": source_index,
    })
    return row


def _envelope_result(
    envelope: Any, protocol, resource_profile: str, title: str, abstract: str
):
    result = envelope.to_public_result(protocol, title, abstract)
    result["resource_profile"] = resource_profile
    return result


def _counts(rows):
    return SCREENING_SESSION.counts(rows)


def _checkpoint(rows: list[dict[str, Any]], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _local_checkpoint_key(input_fingerprint: str, run_id: str) -> str:
    payload = json.dumps({
        "input_fingerprint": input_fingerprint,
        "run_id": run_id,
        "architecture_version": THREE_LAYER_PROMPT_VERSION,
        "triage_model": TRIAGE_MODEL,
        "deep_model": DEEP_MODEL,
        "edge_model": EDGE_MODEL,
        "triage_batch_size": TRIAGE_BATCH_SIZE,
        "deep_batch_size": DEEP_BATCH_SIZE,
        "edge_batch_size": EDGE_BATCH_SIZE,
    }, sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def _resume_rows(output_path: str, protocol_id: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(output_path):
        return {}
    try:
        frame = pd.read_csv(output_path)
        if "Protocol_ID" not in frame or "Source_Row_Index" not in frame:
            return {}
        frame = frame[frame["Protocol_ID"].astype(str) == str(protocol_id)]
        if "Validation_Status" in frame:
            frame = frame[frame["Validation_Status"].astype(str) == "validated"]
        resumed: dict[str, dict[str, Any]] = {}
        for _, source_row in frame.iterrows():
            row = source_row.to_dict()
            historical = row.get("Original_Processing_Seconds")
            if pd.isna(historical) or historical in {None, ""}:
                historical = row.get("Processing_Seconds", 0.0)
            row["Original_Processing_Seconds"] = historical
            row["Processing_Seconds"] = 0.0
            row["Cache_Hit"] = True
            resumed[str(row["Source_Row_Index"])] = row
        return resumed
    except (OSError, ValueError, KeyError):
        return {}


def _local_resume_rows(output_path: str, protocol_id: str) -> dict[str, dict[str, Any]]:
    rows = _resume_rows(output_path, protocol_id)
    final: dict[str, dict[str, Any]] = {}
    for source_id, row in rows.items():
        if str(row.get("Prompt_Version", "")) != THREE_LAYER_PROMPT_VERSION:
            continue
        try:
            trace = json.loads(str(row.get("Layer_Trace_JSON") or "[]"))
        except json.JSONDecodeError:
            trace = []
        last_name = str(trace[-1].get("name", "")) if trace else ""
        decision = str(row.get("Decision", ""))
        risk = str(row.get("Decision_Risk", ""))
        if last_name == "edge_critic":
            final[source_id] = row
        elif last_name == "deep_review" and decision in {"KEEP", "REJECT"} and risk == "LOW":
            final[source_id] = row
        elif last_name == "quick_triage" and decision in {"KEEP", "REJECT"} and risk == "LOW":
            final[source_id] = row
    return final


def _screen_csv_local_three_layer(
    *, frame, valid, title_col, abstract_col, research_question, inclusion_criteria,
    exclusion_criteria, research_context, output_path, checkpoint_path, job_id, profile, resume, limit,
):
    orchestrator = ThreeLayerLocalOrchestrator(profile=profile)
    run_id = orchestrator.run_protocol_id(
        research_question, inclusion_criteria, exclusion_criteria, research_context
    )
    resumed = _local_resume_rows(checkpoint_path, run_id) if resume else {}
    rows_by_source: dict[str, dict[str, Any]] = dict(resumed)
    paper_by_id: dict[str, dict[str, Any]] = {}
    for source_index, source_row in valid.iterrows():
        source_key = str(source_index)
        source = source_row.to_dict()
        paper_by_id[source_key] = {
            "id": source_key, "source_index": source_index, "source": source,
            "title": "" if pd.isna(source_row[title_col]) else str(source_row[title_col]),
            "abstract": "" if pd.isna(source_row[abstract_col]) else str(source_row[abstract_col]),
        }

    PROGRESS.set_resumed_count(job_id, len(resumed))
    if resumed:
        resumed_counts = _counts(list(resumed.values()))
        PROGRESS.update_counts(
            job_id, len(resumed), resumed_counts["keep"],
            resumed_counts["maybe"], resumed_counts["reject"],
        )
    pending = [paper_by_id[str(index)] for index in valid.index if str(index) not in rows_by_source]
    protocol = None
    if pending:
        PROGRESS.begin_batches(job_id, "protocol_setup", 1, 1, 1)
        protocol = orchestrator.compile_protocol(
            research_question, inclusion_criteria, exclusion_criteria, research_context
        )
        PROGRESS.update_batch(job_id, 1, 1)
        orchestrator.unload_deep()
    triage_results = {}
    triage_batches = (len(pending) + TRIAGE_BATCH_SIZE - 1) // TRIAGE_BATCH_SIZE
    PROGRESS.begin_batches(job_id, "batched_triage", len(pending), triage_batches, TRIAGE_BATCH_SIZE)
    triage_seen = 0
    triage_batch_current = 0
    live_counts = dict(resumed_counts) if resumed else {"keep": 0, "maybe": 0, "reject": 0}

    def triage_progress(metric):
        nonlocal triage_seen, triage_batch_current
        if int(metric.get("invalid_papers") or 0) > 0:
            PROGRESS.record_retry(job_id)
        completed = int(metric.get("completed_papers") or 0)
        if not completed:
            return
        triage_seen += completed
        triage_batch_current += 1
        decisions = metric.get("decision_counts") or {}
        for decision, key in (("KEEP", "keep"), ("MAYBE", "maybe"), ("REJECT", "reject")):
            live_counts[key] += int(decisions.get(decision) or 0)
        PROGRESS.update_batch(job_id, triage_batch_current, min(len(pending), triage_seen))
        PROGRESS.update_counts(
            job_id, len(resumed) + min(len(pending), triage_seen),
            live_counts["keep"], live_counts["maybe"], live_counts["reject"],
        )

    if pending:
        triage_results, _ = orchestrator.triage_batch(
            research_question, pending, inclusion_criteria, exclusion_criteria,
            research_context, protocol,
            on_batch=triage_progress,
        )
        for paper in pending:
            layer = triage_results[str(paper["id"])]
            rows_by_source[str(paper["id"])] = _row_from_result(
                paper["source"], paper["title"], paper["abstract"], layer.result, paper["source_index"]
            )
        ordered = [rows_by_source[str(index)] for index in valid.index]
        _checkpoint(ordered, checkpoint_path)
        counts = _counts(ordered)
        PROGRESS.update_counts(job_id, len(ordered), counts["keep"], counts["maybe"], counts["reject"])

    deep_papers = [
        paper for paper in pending
        if orchestrator.needs_deep_review(triage_results[str(paper["id"])])
    ]
    edge_papers: list[dict[str, Any]] = []
    deep_results = {}
    if deep_papers:
        orchestrator.unload_triage()
        deep_batches = (len(deep_papers) + DEEP_BATCH_SIZE - 1) // DEEP_BATCH_SIZE
        PROGRESS.begin_batches(job_id, "batched_deep_review", len(deep_papers), deep_batches, DEEP_BATCH_SIZE)
        if protocol is not None:
            deep_seen = 0
            deep_batch_current = 0

            def deep_progress(metric):
                nonlocal deep_seen, deep_batch_current
                if int(metric.get("invalid_papers") or 0) > 0:
                    PROGRESS.record_retry(job_id)
                completed = int(metric.get("completed_papers") or 0)
                if not completed:
                    return
                deep_seen += completed
                deep_batch_current += 1
                PROGRESS.update_batch(job_id, min(deep_batches, deep_batch_current), min(len(deep_papers), deep_seen))

            deep_results, _ = orchestrator.deep_review_batch(
                protocol, run_id, deep_papers, triage_results, on_batch=deep_progress
            )
            for paper in deep_papers:
                source_key = str(paper["id"])
                deep = deep_results[source_key]
                rows_by_source[source_key] = _row_from_result(
                    paper["source"], paper["title"], paper["abstract"], deep.result, paper["source_index"]
                )
                if orchestrator.needs_edge_critic(deep):
                    edge_papers.append(paper)
            _checkpoint([rows_by_source[str(index)] for index in valid.index], checkpoint_path)
            counts = _counts(list(rows_by_source.values()))
            PROGRESS.update_counts(job_id, len(valid), counts["keep"], counts["maybe"], counts["reject"])

    if edge_papers and protocol is not None:
        orchestrator.prepare_edge_critic()
        edge_batches = (len(edge_papers) + EDGE_BATCH_SIZE - 1) // EDGE_BATCH_SIZE
        PROGRESS.begin_batches(job_id, "batched_edge_critic", len(edge_papers), edge_batches, EDGE_BATCH_SIZE)
        edge_seen = 0
        edge_batch_current = 0

        def edge_progress(metric):
            nonlocal edge_seen, edge_batch_current
            if int(metric.get("invalid_papers") or 0) > 0:
                PROGRESS.record_retry(job_id)
            completed = int(metric.get("completed_papers") or 0)
            if not completed:
                return
            edge_seen += completed
            edge_batch_current += 1
            PROGRESS.update_batch(job_id, min(edge_batches, edge_batch_current), min(len(edge_papers), edge_seen))

        edge_results, _ = orchestrator.edge_critic_batch(
            protocol, run_id, edge_papers, deep_results, on_batch=edge_progress
        )
        for paper in edge_papers:
            source_key = str(paper["id"])
            rows_by_source[source_key] = _row_from_result(
                paper["source"], paper["title"], paper["abstract"],
                edge_results[source_key].result, paper["source_index"],
            )
        _checkpoint([rows_by_source[str(index)] for index in valid.index], checkpoint_path)
        counts = _counts(list(rows_by_source.values()))
        PROGRESS.update_counts(job_id, len(valid), counts["keep"], counts["maybe"], counts["reject"])

    results = [rows_by_source[str(index)] for index in valid.index]
    _checkpoint(results, checkpoint_path)
    _checkpoint(results, output_path)
    SCREENING_SESSION.set_results(
        results, job_id=job_id, output_path=output_path,
        architecture_version=THREE_LAYER_PROMPT_VERSION,
    )
    final_counts = _counts(results)
    PROGRESS.update_counts(
        job_id, len(results), final_counts["keep"], final_counts["maybe"], final_counts["reject"]
    )
    PROGRESS.finish(job_id)
    return {
        **final_counts, "parse_error": 0, "output_file": output_path,
        "total_papers": len(results), "input_total_rows": len(frame),
        "screened_total_rows": len(results), "row_limit_applied": bool(limit),
        "row_limit_value": limit or "", "screening_engine": LOCAL_ENGINE,
        "schema_version": SCHEMA_VERSION, "protocol_id": run_id,
        "model_tier": "resident_three_layer_local", "resource_profile": profile.resource_profile,
        "architecture_version": THREE_LAYER_PROMPT_VERSION,
        "resumed_count": len(resumed),
        "runtime_seconds": PROGRESS.snapshot(job_id).get("runtime_seconds"),
        "fast_model": orchestrator.triage_profile.fast_model,
        "strong_model": orchestrator.deep_profile.fast_model,
        "edge_model": EDGE_MODEL,
        "escalated_count": sum(bool(row.get("Escalated")) for row in results),
        "hardware": profile.as_dict(),
    }


def screen_csv(
    csv_path,
    research_question,
    output_path="outputs/screened.csv",
    mode="local",
    model=None,
    progress_job_id=None,
    max_rows=None,
    screening_engine=None,
    inclusion_criteria="",
    exclusion_criteria="",
    research_context="",
    model_tier=None,
    resource_profile=None,
    resume=True,
    input_fingerprint=None,
    checkpoint_path=None,
    gemini_api_key=None,
    **legacy_options,
):
    job_id = progress_job_id or f"direct-{uuid.uuid4()}"
    if not PROGRESS.start_job(job_id):
        raise RuntimeError("Another screening job is already running.")
    selected_engine = normalize_processing_engine(screening_engine or mode)
    frame = pd.read_csv(csv_path)
    title_col = _find_col(frame, ["Title", "TI", "Article Title", "Document Title", "paper_title", "Name"])
    abstract_col = _find_col(frame, ["Abstract", "AB", "Abstracts", "Summary", "Author Abstract", "Description"])
    if title_col is None or abstract_col is None:
        PROGRESS.fail(job_id, "Input requires title and abstract columns")
        raise KeyError(f"No usable title/abstract columns found. Columns: {list(frame.columns)}")
    valid = frame[frame[abstract_col].notna()]
    limit = int(max_rows or DEV_SCREENING_ROW_LIMIT or 0)
    if limit > 0:
        valid = valid.head(limit)
    architecture_version = (
        THREE_LAYER_PROMPT_VERSION if selected_engine == LOCAL_ENGINE else "external-structured-v2.1"
    )
    SCREENING_SESSION.begin(job_id, output_path, architecture_version)
    PROGRESS.begin_screening(job_id, len(valid), architecture_version)
    profile = resolve_runtime_profile(model_tier, resource_profile)

    if selected_engine == LOCAL_ENGINE:
        fingerprint = str(input_fingerprint or sha256(Path(csv_path).read_bytes()).hexdigest())
        run_id = ThreeLayerLocalOrchestrator(profile=profile).run_protocol_id(
            research_question, inclusion_criteria, exclusion_criteria, research_context
        )
        output_parent = Path(output_path).parent
        checkpoint_root = (
            output_parent.parent / "cache" / "checkpoints"
            if output_parent.name == "runs"
            else output_parent / ".checkpoints"
        )
        resolved_checkpoint = checkpoint_path or str(
            checkpoint_root / f"{_local_checkpoint_key(fingerprint, run_id)}.csv"
        )
        try:
            return _screen_csv_local_three_layer(
                frame=frame, valid=valid, title_col=title_col, abstract_col=abstract_col,
                research_question=research_question,
                inclusion_criteria=inclusion_criteria,
                exclusion_criteria=exclusion_criteria,
                research_context=research_context,
                output_path=output_path, checkpoint_path=resolved_checkpoint,
                job_id=job_id, profile=profile,
                resume=resume, limit=limit,
            )
        except Exception as exc:
            PROGRESS.fail(job_id, exc)
            raise

    from external_ai.orchestrator import ExternalAIScreeningOrchestrator

    engine_context = (
        resolve_processing_engine(selected_engine, gemini_api_key=gemini_api_key)
        if selected_engine != LOCAL_ENGINE else None
    )
    context = engine_context if engine_context is not None else _NullContext()
    try:
        with context as external_engine:
            orchestrator = ExternalAIScreeningOrchestrator(
                profile=profile,
                inference_engine=external_engine if selected_engine != LOCAL_ENGINE else None,
            )
            worker_concurrency = profile.concurrency
            if (
                selected_engine == LOCAL_ENGINE
                and profile.resource_profile == "maximum"
                and hasattr(orchestrator.engine, "calibrate")
            ):
                calibration = orchestrator.engine.calibrate()
                worker_concurrency = max(
                    1, int(calibration.get("recommended_concurrency") or worker_concurrency)
                )
            protocol = orchestrator.compile_protocol(
                research_question, inclusion_criteria, exclusion_criteria
            )
            resumed = _resume_rows(output_path, protocol.protocol_id) if resume else {}
            rows_by_source: dict[str, dict[str, Any]] = {}
            pending: list[tuple[str, str, str, dict[str, Any], Any]] = []

            work = []
            for source_index, source_row in valid.iterrows():
                source_key = str(source_index)
                if source_key in resumed:
                    rows_by_source[source_key] = resumed[source_key]
                else:
                    work.append((source_key, source_index, source_row.to_dict(), str(source_row[title_col] or ""), str(source_row[abstract_col] or "")))

            completed = len(rows_by_source)
            counts = _counts(list(rows_by_source.values()))
            PROGRESS.update_counts(job_id, completed, counts["keep"], counts["maybe"], counts["reject"])

            def record_fast(item, envelope):
                nonlocal completed
                source_key, source_index, source, title, abstract = item
                result = _envelope_result(
                    envelope, protocol, profile.resource_profile, title, abstract
                )
                rows_by_source[source_key] = _row_from_result(source, title, abstract, result, source_index)
                if envelope.needs_escalation():
                    pending.append((source_key, title, abstract, source, envelope))
                completed += 1
                ordered = [rows_by_source[str(index)] for index in valid.index if str(index) in rows_by_source]
                counts = _counts(ordered)
                PROGRESS.update_counts(job_id, completed, counts["keep"], counts["maybe"], counts["reject"])
                if LOCAL_CHECKPOINT_INTERVAL and completed % LOCAL_CHECKPOINT_INTERVAL == 0:
                    _checkpoint(ordered, output_path)

            if worker_concurrency > 1 and len(work) > 1:
                with ThreadPoolExecutor(max_workers=worker_concurrency) as pool:
                    futures = {
                        pool.submit(orchestrator.assess_fast, protocol, item[3], item[4]): item
                        for item in work
                    }
                    for future in as_completed(futures):
                        record_fast(futures[future], future.result())
            else:
                for item in work:
                    record_fast(item, orchestrator.assess_fast(protocol, item[3], item[4]))

            if pending:
                PROGRESS.begin_stage2(job_id, len(pending))
                orchestrator.prepare_strong_pass()
                for number, (source_key, title, abstract, source, envelope) in enumerate(pending, start=1):
                    final = orchestrator.escalate(protocol, title, abstract, envelope)
                    result = _envelope_result(
                        final, protocol, profile.resource_profile, title, abstract
                    )
                    rows_by_source[source_key] = _row_from_result(
                        source, title, abstract, result, source_key
                    )
                    PROGRESS.update_stage2(job_id, number)
                    if LOCAL_CHECKPOINT_INTERVAL and number % LOCAL_CHECKPOINT_INTERVAL == 0:
                        ordered = [rows_by_source[str(index)] for index in valid.index]
                        _checkpoint(ordered, output_path)

            results = [rows_by_source[str(index)] for index in valid.index]
            _checkpoint(results, output_path)
            SCREENING_SESSION.set_results(
                results, job_id=job_id, output_path=output_path,
                architecture_version=architecture_version,
            )
            final_counts = _counts(results)
            PROGRESS.update_counts(
                job_id, len(results), final_counts["keep"], final_counts["maybe"], final_counts["reject"]
            )
            PROGRESS.finish(job_id)
            return {
                **final_counts,
                "parse_error": 0,
                "output_file": output_path,
                "total_papers": len(results),
                "input_total_rows": len(frame),
                "screened_total_rows": len(results),
                "row_limit_applied": bool(limit),
                "row_limit_value": limit or "",
                "screening_engine": selected_engine,
                "architecture_version": architecture_version,
                "resumed_count": len(resumed),
                "schema_version": SCHEMA_VERSION,
                "protocol_id": protocol.protocol_id,
                "model_tier": orchestrator.profile.resolved_tier,
                "resource_profile": orchestrator.profile.resource_profile,
                "fast_model": orchestrator.profile.fast_model,
                "strong_model": orchestrator.profile.strong_model,
                "escalated_count": sum(bool(row.get("Escalated")) for row in results),
                "hardware": orchestrator.hardware_diagnostics(),
            }
    except Exception as exc:
        PROGRESS.fail(job_id, exc)
        raise


class _NullContext:
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return None
