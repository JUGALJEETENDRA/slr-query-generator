from __future__ import annotations

import asyncio
import os
import re
import math
import json
import traceback
import uuid
from contextlib import asynccontextmanager
from hashlib import sha256
from datetime import date, datetime
from pathlib import Path
from threading import Thread
from typing import Any, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
from dotenv import load_dotenv

from litsync_app.screening.bulk import PROGRESS, SCREENING_SESSION, screen_csv
from litsync_app.screening.exports import (
    EXPORT_FILENAMES,
    ScreeningExportError,
    export_file_path,
    generate_screening_exports,
    validate_job_id,
)
from litsync_app.query.generator import generate_query_bundle
from litsync_app.validation.gold import create_blinded_sample, evaluate_completed_labels
from litsync_app.deduplication import deduplicate, parse_upload_files
from litsync_app.screening.local.hardware import resolve_runtime_profile
from litsync_app.screening.local_ai import (
    ARCHITECTURE_VERSION as LOCAL_AI_ARCHITECTURE,
    LOCAL_MODEL,
)
from litsync_app.screening.engines import (
    GEMINI_WEB_ENGINE, LOCAL_ENGINE,
)
from litsync_app.prisma import PRISMA_STORE, manifest_csv, manifest_svg
from litsync_app.paper_collection import AgenticWorkflowManager
from litsync_app.integrations.gemini_web_screening_prompt import criterion_entries


load_dotenv()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
PRIVATE_DIR = "private"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HTML_FILE = PROJECT_ROOT / "web" / "slr_query_generator.html"
TEAM_HTML_FILE = PROJECT_ROOT / "web" / "team.html"
AGENTIC_WORKFLOWS = AgenticWorkflowManager()


def _ensure_runtime_directories() -> None:
    for directory in (UPLOAD_DIR, OUTPUT_DIR, PRIVATE_DIR):
        Path(directory).mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def app_lifespan(_app: FastAPI):
    _ensure_runtime_directories()
    AGENTIC_WORKFLOWS.recover()
    yield


app = FastAPI(
    title="LitSync Local-AI Systematic Review API",
    version="2.0",
    lifespan=app_lifespan,
)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR, check_dir=False), name="outputs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"^http://(?:localhost|127\.0\.0\.1)(?::\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)


def _json_safe(value: Any):
    """Convert pandas/numpy values and non-finite floats into browser-safe JSON."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if value != value:
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _output_url(path: str) -> str:
    target = Path(path).resolve()
    root = Path(OUTPUT_DIR).resolve()
    try:
        relative = target.relative_to(root)
    except ValueError:
        relative = Path(target.name)
    return "/outputs/" + relative.as_posix()


def _persisted_screening_rows(job_id: str) -> tuple[list[dict[str, Any]], Path]:
    selected = str(job_id or "").strip()
    if not selected:
        raise ValueError(
            "Select a completed screening job before evaluating Gold Validation."
        )
    if not re.fullmatch(r"[A-Za-z0-9._-]+", selected):
        raise ValueError("Invalid screening job ID.")
    runs_root = (Path(OUTPUT_DIR) / "runs").resolve()
    output_path = (runs_root / f"screened-{selected}.csv").resolve()
    try:
        output_path.relative_to(runs_root)
    except ValueError as exc:
        raise ValueError("Invalid screening job ID.") from exc
    if not output_path.is_file():
        raise FileNotFoundError(
            f"Persisted screening output for job '{selected}' was not found."
        )
    frame = pd.read_csv(
        output_path, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    rows = frame.to_dict(orient="records")
    if not rows:
        raise ValueError(f"Persisted screening output for job '{selected}' is empty.")
    return rows, output_path


def _current_screening_rows(job_id: str | None = None) -> list[dict[str, Any]]:
    rows = SCREENING_SESSION.snapshot(job_id)
    if rows:
        return rows
    if job_id is not None:
        try:
            rows, output_path = _persisted_screening_rows(job_id)
        except (OSError, ValueError):
            return []
        architecture = str(rows[0].get("Prompt_Version") or "")
        SCREENING_SESSION.begin(str(job_id), str(output_path), architecture)
        SCREENING_SESSION.set_results(
            rows, job_id=str(job_id), output_path=str(output_path),
            architecture_version=architecture,
        )
        return rows
    if PROGRESS.is_running():
        return []
    manifest_path = Path(OUTPUT_DIR) / "latest_screening.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        output_path = Path(str(manifest["output_path"]))
        architecture = str(manifest.get("architecture_version") or "")
        if architecture not in {
            LOCAL_AI_ARCHITECTURE,
            "gemini-web-screening-v1",
        }:
            return []
        rows = pd.read_csv(output_path).to_dict(orient="records")
        SCREENING_SESSION.begin(
            str(manifest["job_id"]), str(output_path), architecture
        )
        SCREENING_SESSION.set_results(
            rows, job_id=str(manifest["job_id"]), output_path=str(output_path),
            architecture_version=architecture,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return []
    return rows


def _save_latest_screening(job_id: str, summary: dict[str, Any]) -> None:
    output_path = str(summary["output_file"])
    manifest = {
        "job_id": job_id,
        "output_path": output_path,
        "architecture_version": summary.get("architecture_version"),
        "prisma_workflow_id": job_id,
        "prisma_path": str(Path(OUTPUT_DIR) / "prisma" / f"{job_id}.json"),
        "summary": {
            key: summary.get(key) for key in (
                "keep", "maybe", "reject", "total_papers", "runtime_seconds",
                "screening_engine", "primary_batch_size", "primary_batches_submitted",
                "primary_batches_completed", "verification_batches_submitted",
                "verification_batches_completed", "primary_papers_requested",
                "verification_papers_requested", "missing_abstract_count",
                "safe_fallback_count", "retry_count", "stopped_by_time_budget",
                "resumed_count", "fresh_primary_count", "browser_context_started",
                "peak_simultaneous_tabs", "pages_opened", "pages_closed",
                "transport_failure_count",
                "protocol_id", "transport_diagnostics",
            )
        },
    }
    target = Path(OUTPUT_DIR) / "latest_screening.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(target)


@app.get("/", include_in_schema=False)
async def homepage():
    return FileResponse(HTML_FILE)


@app.get("/team.html", include_in_schema=False)
async def team_page():
    return FileResponse(TEAM_HTML_FILE)


@app.get("/status")
async def local_ai_status():
    """Report readiness without loading an Ollama model."""
    profile = resolve_runtime_profile()
    hardware = profile.hardware
    installed = sorted(hardware.installed_models)
    required = [LOCAL_MODEL]
    missing = [model for model in required if model not in hardware.installed_models]
    ollama_ready = bool(installed)
    # Local screening uses the fixed throughput stack above; legacy tier-model
    # downgrade warnings do not describe this path and would be misleading.
    warnings = []
    if not ollama_ready:
        warnings.append("Ollama is not running, or no local models are installed.")
    elif missing:
        warnings.append("Missing selected model(s): " + ", ".join(missing))
    return _json_safe({
        "status": "ready" if ollama_ready and not missing else "needs_attention",
        "backend_ready": True,
        "ollama_ready": ollama_ready,
        "hardware": {
            "total_ram_gb": hardware.total_ram_gb,
            "available_ram_gb": hardware.available_ram_gb,
            "cpu_cores": hardware.cpu_cores,
            "gpu_name": hardware.gpu_name,
            "gpu_vram_gb": hardware.gpu_vram_gb,
        },
        "resolved": {
            "tier": profile.resolved_tier,
            "resource_profile": profile.resource_profile,
            "model": LOCAL_MODEL,
            "architecture_version": LOCAL_AI_ARCHITECTURE,
        },
        "installed_models": installed,
        "required_models": required,
        "missing_models": missing,
        "calibrated": bool(profile.calibration),
        "skyvern": {
            "configured": bool(os.getenv("SKYVERN_API_KEY", "").strip()),
            "credential_sources_configured": sorted(
                source for source, variable in {
                    "google_scholar": "SKYVERN_CREDENTIAL_GOOGLE_SCHOLAR",
                    "scopus": "SKYVERN_CREDENTIAL_SCOPUS",
                    "web_of_science": "SKYVERN_CREDENTIAL_WEB_OF_SCIENCE",
                    "ieee_xplore": "SKYVERN_CREDENTIAL_IEEE_XPLORE",
                    "pubmed": "SKYVERN_CREDENTIAL_PUBMED",
                }.items()
                if os.getenv(variable, "").strip()
            ),
        },
        "warnings": warnings,
    })


class QuestionRequest(BaseModel):
    question: str
    processing_engine: str = LOCAL_ENGINE


class AgenticRunRequest(BaseModel):
    topic: str


class FinalizeRequest(BaseModel):
    titles: List[str] = []
    papers: List[dict] = []
    job_id: str | None = None


class GoldSampleRequest(BaseModel):
    question: str
    sample_size: int = 60
    job_id: str = ""
    sampling_strata: dict[str, float] | None = None


class ManualDecisionRequest(BaseModel):
    decision: str
    exclusion_reason: str = ""


def _prisma_snapshot(job_id: str, progress: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = SCREENING_SESSION.snapshot(job_id)
    try:
        return PRISMA_STORE.snapshot(
            job_id, output_root=OUTPUT_DIR, progress=progress or {}, rows=rows,
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        metadata = SCREENING_SESSION.metadata()
        output_path = Path(str(metadata.get("output_path") or ""))
        fingerprint = sha256(output_path.read_bytes()).hexdigest() if output_path.is_file() else "restored"
        PRISMA_STORE.begin_screening(
            output_root=OUTPUT_DIR, job_id=job_id, input_fingerprint=fingerprint,
            screening_engine="restored", import_id=None,
        )
        total = len(rows) or int((progress or {}).get("total") or 0)
        PRISMA_STORE.configure_screening(
            job_id, input_rows=total, missing_abstracts=0,
            records_available=total, records_selected=total,
        )
        return PRISMA_STORE.snapshot(job_id, progress=progress or {}, rows=rows)


def _prisma_urls(workflow_id: str) -> dict[str, str]:
    quoted = str(workflow_id)
    return {
        "json": f"/prisma/{quoted}",
        "csv": f"/prisma/{quoted}.csv",
        "svg": f"/prisma/{quoted}.svg",
    }


def _workflow_manifest(workflow_id: str) -> dict[str, Any]:
    progress = PROGRESS.snapshot(workflow_id)
    rows = SCREENING_SESSION.snapshot(workflow_id)
    return PRISMA_STORE.snapshot(
        workflow_id, output_root=OUTPUT_DIR, progress=progress or {}, rows=rows,
    )


def _write_prisma_exports(workflow_id: str, manifest: dict[str, Any], directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "prisma_json": directory / "prisma_manifest.json",
        "prisma_csv": directory / "prisma_manifest.csv",
        "prisma_svg": directory / "prisma_flow.svg",
    }
    paths["prisma_json"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["prisma_csv"].write_text(manifest_csv(manifest), encoding="utf-8-sig")
    paths["prisma_svg"].write_text(manifest_svg(manifest), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


@app.get("/debug/hardware-profile")
async def hardware_profile(calibrate: bool = False):
    profile = resolve_runtime_profile()
    return {
        **profile.as_dict(),
        "model": LOCAL_MODEL,
        "architecture_version": LOCAL_AI_ARCHITECTURE,
        "calibration_disabled": True,
        "legacy_hardware_tier_models_ignored": True,
    }


@app.get("/debug/model-judge-config")
async def compatibility_model_config():
    profile = resolve_runtime_profile()
    return {
        "deprecated": True,
        "message": "Legacy model judges were removed. This endpoint now reports the local-AI runtime.",
        **profile.as_dict(),
    }


@app.post("/generate")
async def generate(req: QuestionRequest):
    question = req.question.strip()
    selected_engine = req.processing_engine.strip().lower()
    if not question:
        return {"status": "error", "message": "Enter a research question."}
    if selected_engine not in {LOCAL_ENGINE, GEMINI_WEB_ENGINE}:
        return {
            "status": "error",
            "message": "Choose Local Ollama or Gemini Web Automation for query generation.",
        }
    try:
        bundle = await asyncio.to_thread(
            generate_query_bundle,
            question,
            processing_engine=selected_engine,
        )
        response = bundle.to_api_response()
        response["concepts"] = {
            **dict(response.get("concepts") or {}),
            "question": question,
            "question_fingerprint": sha256(question.encode("utf-8")).hexdigest(),
        }
        return response
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.post("/agentic-runs", status_code=202)
async def create_agentic_run(req: AgenticRunRequest):
    topic = req.topic.strip()
    if len(topic) < 3:
        raise HTTPException(status_code=400, detail="Enter a research topic.")
    if PROGRESS.is_running():
        raise HTTPException(
            status_code=409,
            detail="A screening job is already running. Wait for it to finish before starting an agentic run.",
        )
    try:
        run = AGENTIC_WORKFLOWS.create(topic)
        return _json_safe({
            "status": run.status,
            "run_id": run.run_id,
            "poll_url": f"/agentic-runs/{run.run_id}",
        })
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/agentic-runs/{run_id}")
async def get_agentic_run(run_id: str):
    try:
        return _json_safe(AGENTIC_WORKFLOWS.public(run_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agentic run not found.") from exc


@app.post("/agentic-runs/{run_id}/resume", status_code=202)
async def resume_agentic_run(run_id: str):
    try:
        run = AGENTIC_WORKFLOWS.resume(run_id)
        return _json_safe({"status": run.status, "run_id": run.run_id})
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agentic run not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/agentic-runs/{run_id}/sources/{database}/skip", status_code=202)
async def skip_agentic_source(run_id: str, database: str):
    try:
        run = AGENTIC_WORKFLOWS.skip_source(run_id, database)
        return _json_safe({"status": run.status, "run_id": run.run_id, "database": database})
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agentic run not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/agentic-runs/{run_id}/cancel")
async def cancel_agentic_run(run_id: str):
    try:
        run = AGENTIC_WORKFLOWS.cancel(run_id)
        return _json_safe({"status": run.status, "run_id": run.run_id})
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agentic run not found.") from exc


@app.post("/litsync")
async def litsync_endpoint(files: List[UploadFile] = File(...)):
    _ensure_runtime_directories()
    saved_paths = []
    source_files: list[dict[str, Any]] = []
    try:
        for uploaded in files:
            if not uploaded.filename:
                continue
            filename = os.path.basename(uploaded.filename)
            if Path(filename).suffix.lower() not in {".csv", ".xls", ".xlsx"}:
                continue
            destination = os.path.join(UPLOAD_DIR, filename)
            Path(destination).write_bytes(await uploaded.read())
            saved_paths.append(destination)
            try:
                if Path(destination).suffix.lower() == ".csv":
                    source_count = len(pd.read_csv(destination))
                else:
                    source_count = len(pd.read_excel(destination, sheet_name=0))
            except (OSError, ValueError):
                source_count = 0
            source_files.append({"name": filename, "records": source_count})
        combined, total_initial = parse_upload_files(saved_paths)
        deduped, removed = deduplicate(combined)
        import_id = str(uuid.uuid4())
        import_dir = Path(OUTPUT_DIR) / "imports" / import_id
        import_dir.mkdir(parents=True, exist_ok=True)
        output_name = "clean_dataset.csv"
        output_path = import_dir / output_name
        deduped.to_csv(output_path, index=False)
        clean_fingerprint = sha256(output_path.read_bytes()).hexdigest()
        prisma = PRISMA_STORE.create_import(
            output_root=OUTPUT_DIR, import_id=import_id,
            records_identified=int(total_initial), duplicate_records_removed=int(removed),
            source_files=source_files, clean_fingerprint=clean_fingerprint,
            clean_path=str(output_path),
        )
        return {
            "status": "success",
            "import_id": import_id,
            "counts": {"initial": int(total_initial), "deduped": len(deduped), "duplicates_removed": int(removed)},
            "prisma": prisma,
            "prisma_downloads": _prisma_urls(import_id),
            "download_url": _output_url(str(output_path)),
            "output_filename": output_name,
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def run_screening(job_id: str, csv_path: str, question: str, **options):
    try:
        summary = screen_csv(
            csv_path=csv_path, research_question=question, progress_job_id=job_id, **options
        )
        _save_latest_screening(job_id, summary)
    except Exception as exc:
        traceback.print_exc()
        PROGRESS.fail(job_id, exc)


@app.post("/screen_csv")
async def screen_csv_endpoint(
    question: str = Form(...),
    file: UploadFile = File(...),
    inclusion_criteria: str = Form(""),
    exclusion_criteria: str = Form(""),
    research_context: str = Form(""),
    screening_engine: str = Form(LOCAL_ENGINE),
    max_rows: int | None = Form(None),
    resume: bool = Form(True),
    import_id: str = Form(""),
):
    _ensure_runtime_directories()
    if AGENTIC_WORKFLOWS.has_active():
        raise HTTPException(
            status_code=409,
            detail="An agentic workflow is active. Complete or cancel it before starting a manual screening job.",
        )
    job_id = str(uuid.uuid4())
    requested_engine = str(screening_engine or "").strip().lower().replace("-", "_")
    if requested_engine not in {LOCAL_ENGINE, GEMINI_WEB_ENGINE}:
        raise HTTPException(status_code=400, detail="Choose Gemini Web or Local AI for screening.")
    selected_engine = requested_engine
    if not PROGRESS.start_job(job_id):
        raise HTTPException(status_code=409, detail="Another screening job is already running.")
    filename = os.path.basename(file.filename or "screening.csv")
    csv_path = os.path.join(UPLOAD_DIR, f"{job_id}-{filename}")
    try:
        Path(csv_path).write_bytes(await file.read())
    except Exception as exc:
        PROGRESS.fail(job_id, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    output_path = os.path.join(OUTPUT_DIR, "runs", f"screened-{job_id}.csv")
    architecture_version = (
        LOCAL_AI_ARCHITECTURE if selected_engine == LOCAL_ENGINE else "gemini-web-screening-v1"
    )
    SCREENING_SESSION.begin(job_id, output_path, architecture_version)
    input_fingerprint = sha256(Path(csv_path).read_bytes()).hexdigest()
    protocol_inputs = {
        "research_question": question,
        "research_context": research_context,
        "inclusion_criteria": inclusion_criteria,
        "exclusion_criteria": exclusion_criteria,
        "parsed_authoritative_inclusion_count": len(
            criterion_entries(inclusion_criteria)
        ),
        "parsed_authoritative_exclusion_count": len(
            criterion_entries(exclusion_criteria)
        ),
    }
    prisma = PRISMA_STORE.begin_screening(
        output_root=OUTPUT_DIR, job_id=job_id, input_fingerprint=input_fingerprint,
        screening_engine=selected_engine, import_id=import_id or None,
        protocol_inputs=protocol_inputs,
    )
    thread = Thread(
        target=run_screening,
        kwargs={
            "job_id": job_id, "csv_path": csv_path, "question": question,
            "output_path": output_path,
            "inclusion_criteria": inclusion_criteria,
            "exclusion_criteria": exclusion_criteria,
            "research_context": research_context,
            "screening_engine": selected_engine,
            "max_rows": max_rows,
            "resume": resume,
            "input_fingerprint": input_fingerprint,
        },
        daemon=True,
    )
    thread.start()
    return {
        "status": "started", "job_id": job_id,
        "architecture_version": architecture_version,
        "screening_engine": selected_engine,
        "prisma": prisma,
        "prisma_downloads": _prisma_urls(job_id),
    }


@app.get("/maybe_papers")
async def get_maybe_papers(job_id: str | None = None):
    if job_id and SCREENING_SESSION.metadata().get("job_id") != job_id:
        raise HTTPException(status_code=404, detail="Screening job not found.")
    rows = [
        row for row in SCREENING_SESSION.snapshot(job_id)
        if row.get("Decision") == "MAYBE"
    ]
    return _json_safe({"count": len(rows), "papers": rows})


@app.get("/screening_results")
async def get_screening_results(job_id: str | None = None):
    if job_id:
        progress = PROGRESS.snapshot(job_id)
        metadata = SCREENING_SESSION.metadata()
        if progress is None and metadata.get("job_id") != job_id:
            try:
                _persisted_screening_rows(job_id)
            except (OSError, ValueError):
                raise HTTPException(status_code=404, detail="Screening job not found.")
        if progress and progress.get("status") in {"starting", "running"}:
            return _json_safe({
                "status": "running", "job_id": job_id, "papers": [],
                "counts": {"total": 0, "keep": 0, "maybe": 0, "reject": 0},
                "prisma": _prisma_snapshot(job_id, progress),
                "prisma_downloads": _prisma_urls(job_id),
                "download_url": None,
            })
    papers = _current_screening_rows(job_id)
    metadata = SCREENING_SESSION.metadata()
    workflow_id = metadata.get("job_id")
    prisma = (
        _prisma_snapshot(str(workflow_id), PROGRESS.snapshot(workflow_id) or {})
        if workflow_id else None
    )
    return _json_safe({
        "status": "finished" if papers else "empty",
        "job_id": metadata.get("job_id"),
        "architecture_version": metadata.get("architecture_version"),
        "counts": SCREENING_SESSION.counts(papers),
        "prisma": prisma,
        "prisma_downloads": _prisma_urls(str(workflow_id)) if workflow_id else {},
        "papers": papers,
        "download_url": _output_url(metadata["output_path"])
        if papers and metadata.get("output_path") else None,
    })


@app.post("/gold_validation/sample")
async def create_gold_validation_sample(req: GoldSampleRequest):
    _ensure_runtime_directories()
    try:
        if not req.job_id.strip():
            raise ValueError(
                "Select a completed screening job before creating Gold Validation."
            )
        rows, _ = _persisted_screening_rows(req.job_id)
        result = create_blinded_sample(
            rows, req.question,
            OUTPUT_DIR, req.job_id, req.sample_size, PRIVATE_DIR, req.sampling_strata
        )
        filename = Path(result["label_path"]).name
        public_result = {
            key: value for key, value in result.items()
            if key not in {"label_path", "manifest_path"}
        }
        return _json_safe({
            "status": "success",
            **public_result,
            "download_url": f"/outputs/gold_validation/{filename}",
        })
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/gold_validation/evaluate")
async def evaluate_gold_validation(
    job_id: str = Form(""),
    file: UploadFile = File(...),
):
    _ensure_runtime_directories()
    filename = os.path.basename(file.filename or "completed_gold_validation.csv")
    if Path(filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Upload the completed validation CSV file.")
    upload_path = Path(UPLOAD_DIR) / f"gold-{uuid.uuid4()}-{filename}"
    try:
        rows, _ = _persisted_screening_rows(job_id)
        upload_path.write_bytes(await file.read())
        result = evaluate_completed_labels(
            upload_path, PRIVATE_DIR, job_id, rows, OUTPUT_DIR
        )
        report_name = Path(result.pop("report_path")).name
        return _json_safe({
            "status": "success",
            **result,
            "report_download_url": f"/outputs/gold_validation/{report_name}",
        })
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        try:
            upload_path.unlink(missing_ok=True)
        except OSError:
            pass




@app.get("/prisma/{workflow_id}.csv")
async def prisma_csv_export(workflow_id: str):
    try:
        content = manifest_csv(_workflow_manifest(workflow_id))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=404, detail="PRISMA workflow not found.") from exc
    return Response(
        content=content, media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="prisma-{workflow_id}.csv"'},
    )


@app.get("/prisma/{workflow_id}.svg")
async def prisma_svg_export(workflow_id: str):
    try:
        content = manifest_svg(_workflow_manifest(workflow_id))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=404, detail="PRISMA workflow not found.") from exc
    return Response(
        content=content, media_type="image/svg+xml",
        headers={"Content-Disposition": f'inline; filename="prisma-{workflow_id}.svg"'},
    )


@app.get("/prisma/{workflow_id}")
async def prisma_json_export(workflow_id: str):
    try:
        manifest = _workflow_manifest(workflow_id)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=404, detail="PRISMA workflow not found.") from exc
    return _json_safe({**manifest, "downloads": _prisma_urls(workflow_id)})


@app.patch("/screening_jobs/{job_id}/records/{source_row_index}")
async def update_manual_screening_decision(
    job_id: str, source_row_index: str, req: ManualDecisionRequest
):
    try:
        paper = SCREENING_SESSION.update_decision(
            job_id, source_row_index, req.decision, req.exclusion_reason
        )
        rows = SCREENING_SESSION.snapshot(job_id)
        output_value = str(SCREENING_SESSION.metadata().get("output_path") or "").strip()
        if output_value:
            output_path = Path(output_value)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_suffix(".manual.tmp")
            pd.DataFrame(rows).to_csv(temporary, index=False)
            temporary.replace(output_path)
        prisma = _prisma_snapshot(job_id, PROGRESS.snapshot(job_id) or {})
        return _json_safe({
            "status": "success", "paper": paper,
            "counts": SCREENING_SESSION.counts(rows), "prisma": prisma,
            "prisma_downloads": _prisma_urls(job_id),
        })
    except (KeyError, RuntimeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/screening-jobs/{job_id}/exports")
async def create_screening_exports(job_id: str):
    _ensure_runtime_directories()
    try:
        selected = validate_job_id(job_id)
        result = generate_screening_exports(OUTPUT_DIR, selected)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ScreeningExportError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Screening exports could not be written.",
        ) from exc
    export_names = result["export_names"]
    try:
        PRISMA_STORE.snapshot(selected, output_root=OUTPUT_DIR)
        PRISMA_STORE.mark_csv_counts_verified(selected, csv_counts_match=True)
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        # Compatibility exports can predate PRISMA manifests. Their downloads
        # remain valid without manufacturing lineage data that never existed.
        pass
    downloads = {
        name: f"/screening-jobs/{selected}/exports/{name}"
        for name in export_names
    }
    return _json_safe({
        "status": "success",
        "job_id": selected,
        "generated": result["generated"],
        "counts": result["counts"],
        "downloads": downloads,
        "filenames": {
            name: EXPORT_FILENAMES[name]
            for name in export_names
        },
    })


@app.get("/screening-jobs/{job_id}/exports/{export_name}")
async def download_screening_export(job_id: str, export_name: str):
    try:
        selected = validate_job_id(job_id)
        target = export_file_path(OUTPUT_DIR, selected, export_name)
    except ScreeningExportError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not target.is_file():
        raise HTTPException(
            status_code=404,
            detail="Screening export has not been generated or is unavailable.",
        )
    media_type = "application/json" if export_name == "summary" else "text/csv"
    return FileResponse(
        target,
        media_type=media_type,
        filename=EXPORT_FILENAMES[export_name],
    )


@app.post("/finalize")
async def finalize_endpoint(req: FinalizeRequest):
    if req.job_id and SCREENING_SESSION.metadata().get("job_id") != req.job_id:
        raise HTTPException(status_code=404, detail="Screening job not found.")
    current_job = req.job_id or SCREENING_SESSION.metadata().get("job_id") or "manual"
    papers = SCREENING_SESSION.snapshot(str(current_job))
    if not papers and req.papers:
        papers = [dict(row) for row in req.papers]
        SCREENING_SESSION.set_results(papers, job_id=str(current_job))
    try:
        unresolved = sum(str(row.get("Decision", "")).upper() == "MAYBE" for row in papers)
        if unresolved:
            raise ValueError(
                f"Resolve all MAYBE records before finalization ({unresolved} remaining)."
            )
        finalize_dir = os.path.join(OUTPUT_DIR, "runs", str(current_job), "finalized")
        result = SCREENING_SESSION.finalize(papers, finalize_dir)
        csv_counts = {
            "total": len(pd.read_csv(result["files"]["screened"])),
            "keep": len(pd.read_csv(result["files"]["included"])),
            "maybe": len(pd.read_csv(result["files"]["maybe"])),
            "reject": len(pd.read_csv(result["files"]["excluded"])),
        }
        expected = SCREENING_SESSION.counts(papers)
        consistent = csv_counts == expected
        if not consistent:
            raise RuntimeError("Final CSV counts do not match the canonical screening results.")
        try:
            PRISMA_STORE.snapshot(str(current_job), output_root=OUTPUT_DIR)
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            PRISMA_STORE.begin_screening(
                output_root=OUTPUT_DIR, job_id=str(current_job),
                input_fingerprint="compatibility-finalization",
                screening_engine="compatibility", import_id=None,
            )
            PRISMA_STORE.configure_screening(
                str(current_job), input_rows=len(papers), missing_abstracts=0,
                records_available=len(papers), records_selected=len(papers),
            )
        PRISMA_STORE.mark_finalized(str(current_job), csv_counts_match=True)
        prisma = _prisma_snapshot(str(current_job), PROGRESS.snapshot(str(current_job)) or {})
        result["files"].update(
            _write_prisma_exports(str(current_job), prisma, Path(finalize_dir))
        )
        result["files"] = {name: _output_url(path) for name, path in result["files"].items()}
        result["download_url"] = result["files"]["screened"]
        return _json_safe({
            "status": "success", **result, "prisma": prisma,
            "prisma_downloads": _prisma_urls(str(current_job)),
        })
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/progress")
async def get_progress(job_id: str | None = None):
    progress = PROGRESS.snapshot(job_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Screening job not found.")
    workflow_id = str(progress.get("job_id"))
    return _json_safe({
        **progress,
        "prisma": _prisma_snapshot(workflow_id, progress),
        "prisma_downloads": _prisma_urls(workflow_id),
    })
