from __future__ import annotations

import os
import re
import math
import json
import traceback
import uuid
from hashlib import sha256
from datetime import date, datetime
from pathlib import Path
from threading import Thread
from typing import Any, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd

from bulk_screen import PROGRESS, SCREENING_SESSION, screen_csv
from direct_ai_generator import generate_query
from gold_validation import create_blinded_sample, evaluate_completed_labels
from litsync import deduplicate, parse_upload_files
from local_ai.hardware import resolve_runtime_profile
from local_ai.three_layer import (
    DEEP_MODEL, EDGE_MODEL, THREE_LAYER_PROMPT_VERSION, TRIAGE_MODEL,
    ThreeLayerLocalOrchestrator,
)
from processing_engines import LOCAL_ENGINE, normalize_processing_engine, resolve_processing_engine
from screening_strategies import DEFAULT_SCREENING_STRATEGY, screen_candidate


UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
PRIVATE_DIR = "private"
HTML_FILE = Path(__file__).resolve().parent / "archive" / "slr_query_generator.html"
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(PRIVATE_DIR).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LitSync Local-AI Systematic Review API", version="2.0")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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


def _current_screening_rows(job_id: str | None = None) -> list[dict[str, Any]]:
    rows = SCREENING_SESSION.snapshot(job_id)
    if rows or job_id is not None or PROGRESS.is_running():
        return rows
    manifest_path = Path(OUTPUT_DIR) / "latest_screening.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        output_path = Path(str(manifest["output_path"]))
        architecture = str(manifest.get("architecture_version") or "")
        if architecture not in {THREE_LAYER_PROMPT_VERSION, "external-structured-v2.1"}:
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
    }
    target = Path(OUTPUT_DIR) / "latest_screening.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(target)


@app.get("/", include_in_schema=False)
async def homepage():
    return FileResponse(HTML_FILE)


@app.get("/status")
async def local_ai_status():
    """Report readiness without loading an Ollama model."""
    profile = resolve_runtime_profile()
    hardware = profile.hardware
    installed = sorted(hardware.installed_models)
    # The website's local path always uses the fixed three-layer stack.  Do not
    # report the legacy hardware-tier pair here: both triage and deep models are
    # required even though only one is resident at a time.
    required = sorted({TRIAGE_MODEL, DEEP_MODEL, EDGE_MODEL})
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
            "triage_model": TRIAGE_MODEL,
            "deep_model": DEEP_MODEL,
            "edge_model": EDGE_MODEL,
        },
        "installed_models": installed,
        "required_models": required,
        "missing_models": missing,
        "calibrated": bool(profile.calibration),
        "warnings": warnings,
    })


class QuestionRequest(BaseModel):
    question: str


class ScreenRequest(BaseModel):
    question: str
    title: str
    abstract: str
    inclusion_criteria: str = ""
    exclusion_criteria: str = ""
    research_context: str = ""
    model_tier: str = "auto"
    resource_profile: str = "balanced"
    processing_engine: str = LOCAL_ENGINE
    semantic_strategy: str = DEFAULT_SCREENING_STRATEGY
    gemini_api_key: str = ""


class FinalizeRequest(BaseModel):
    titles: List[str] = []
    papers: List[dict] = []
    job_id: str | None = None


class GoldSampleRequest(BaseModel):
    question: str
    sample_size: int = 60
    job_id: str | None = None


@app.get("/debug/hardware-profile")
async def hardware_profile(calibrate: bool = False):
    profile = resolve_runtime_profile()
    return {
        **profile.as_dict(),
        "fast_model": TRIAGE_MODEL,
        "strong_model": DEEP_MODEL,
        "architecture_version": THREE_LAYER_PROMPT_VERSION,
        "triage_model": TRIAGE_MODEL,
        "deep_model": DEEP_MODEL,
        "edge_model": EDGE_MODEL,
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
    try:
        base_query = generate_query(req.question).replace("\n", " ")
        return {
            "status": "success",
            "google_scholar": base_query,
            "scopus": f"TITLE-ABS-KEY({base_query})",
            "web_of_science": f"TS=({base_query})",
            "ieee_xplore": base_query,
            "pubmed": re.sub(r'"([^"]+)"', r'"\1"[tiab]', base_query),
            "concepts": {},
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.post("/screen")
async def screen(req: ScreenRequest):
    selected = normalize_processing_engine(req.processing_engine)
    try:
        if selected == LOCAL_ENGINE:
            result = screen_candidate(
                title=req.title, abstract=req.abstract, research_question=req.question,
                inclusion_criteria=req.inclusion_criteria,
                exclusion_criteria=req.exclusion_criteria,
                research_context=req.research_context,
                model_tier=req.model_tier, resource_profile=req.resource_profile,
            )
        else:
            with resolve_processing_engine(selected, gemini_api_key=req.gemini_api_key or None) as engine:
                result = screen_candidate(
                    title=req.title, abstract=req.abstract, research_question=req.question,
                    inclusion_criteria=req.inclusion_criteria,
                    exclusion_criteria=req.exclusion_criteria,
                    model_tier=req.model_tier, resource_profile=req.resource_profile,
                    inference_engine=engine,
                )
        return {"status": "success", **result}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.post("/litsync")
async def litsync_endpoint(files: List[UploadFile] = File(...)):
    saved_paths = []
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
        combined, total_initial = parse_upload_files(saved_paths)
        deduped, removed = deduplicate(combined)
        output_name = "LitSync_Clean_Dataset.csv"
        deduped.to_csv(os.path.join(OUTPUT_DIR, output_name), index=False)
        return {
            "status": "success",
            "counts": {"initial": int(total_initial), "deduped": len(deduped), "duplicates_removed": int(removed)},
            "download_url": f"/outputs/{output_name}",
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
    model_tier: str = Form("auto"),
    resource_profile: str = Form("balanced"),
    screening_engine: str = Form(LOCAL_ENGINE),
    max_rows: int | None = Form(None),
    resume: bool = Form(True),
    gemini_api_key: str = Form(""),
    mode: str = Form("local"),
    model: str = Form(""),
    semantic_strategy: str = Form(DEFAULT_SCREENING_STRATEGY),
    two_stage_enabled: bool = Form(True),
    first_stage_model: str = Form(""),
    second_stage_model: str = Form(""),
):
    job_id = str(uuid.uuid4())
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
    selected_engine = normalize_processing_engine(screening_engine)
    architecture_version = (
        THREE_LAYER_PROMPT_VERSION
        if selected_engine == LOCAL_ENGINE else "external-structured-v2.1"
    )
    SCREENING_SESSION.begin(job_id, output_path, architecture_version)
    input_fingerprint = sha256(Path(csv_path).read_bytes()).hexdigest()
    thread = Thread(
        target=run_screening,
        kwargs={
            "job_id": job_id, "csv_path": csv_path, "question": question,
            "output_path": output_path,
            "inclusion_criteria": inclusion_criteria,
            "exclusion_criteria": exclusion_criteria,
            "research_context": research_context,
            "model_tier": model_tier,
            "resource_profile": resource_profile,
            "screening_engine": screening_engine,
            "gemini_api_key": gemini_api_key or None,
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
        "model_tier": model_tier, "resource_profile": resource_profile,
        "screening_engine": selected_engine,
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
            raise HTTPException(status_code=404, detail="Screening job not found.")
        if progress and progress.get("status") in {"starting", "running"}:
            return _json_safe({
                "status": "running", "job_id": job_id, "papers": [],
                "counts": {"total": 0, "keep": 0, "maybe": 0, "reject": 0},
                "download_url": None,
            })
    papers = _current_screening_rows(job_id)
    metadata = SCREENING_SESSION.metadata()
    return _json_safe({
        "status": "finished" if papers else "empty",
        "job_id": metadata.get("job_id"),
        "architecture_version": metadata.get("architecture_version"),
        "counts": SCREENING_SESSION.counts(papers),
        "papers": papers,
        "download_url": _output_url(metadata["output_path"])
        if papers and metadata.get("output_path") else None,
    })


@app.post("/gold_validation/sample")
async def create_gold_validation_sample(req: GoldSampleRequest):
    if req.job_id and SCREENING_SESSION.metadata().get("job_id") != req.job_id:
        raise HTTPException(status_code=404, detail="Screening job not found.")
    try:
        result = create_blinded_sample(
            _current_screening_rows(req.job_id), req.question,
            OUTPUT_DIR, req.sample_size, PRIVATE_DIR
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
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/gold_validation/evaluate")
async def evaluate_gold_validation(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or "completed_gold_validation.csv")
    if Path(filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Upload the completed validation CSV file.")
    upload_path = Path(UPLOAD_DIR) / f"gold-{uuid.uuid4()}-{filename}"
    try:
        upload_path.write_bytes(await file.read())
        result = evaluate_completed_labels(upload_path, PRIVATE_DIR, OUTPUT_DIR)
        report_name = Path(result.pop("report_path")).name
        return _json_safe({
            "status": "success",
            **result,
            "report_download_url": f"/outputs/gold_validation/{report_name}",
        })
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        try:
            upload_path.unlink(missing_ok=True)
        except OSError:
            pass


@app.post("/finalize")
async def finalize_endpoint(req: FinalizeRequest):
    if req.job_id and SCREENING_SESSION.metadata().get("job_id") != req.job_id:
        raise HTTPException(status_code=404, detail="Screening job not found.")
    papers = req.papers
    if not papers and req.titles:
        selected = set(req.titles)
        papers = []
        for row in SCREENING_SESSION.snapshot():
            edited = dict(row)
            if edited.get("Decision") == "MAYBE" and edited.get("Title") in selected:
                edited["Decision"] = "KEEP"
            papers.append(edited)
    try:
        current_job = req.job_id or SCREENING_SESSION.metadata().get("job_id") or "manual"
        finalize_dir = os.path.join(OUTPUT_DIR, "runs", str(current_job), "finalized")
        result = SCREENING_SESSION.finalize(papers, finalize_dir)
        result["files"] = {name: _output_url(path) for name, path in result["files"].items()}
        result["download_url"] = result["files"]["screened"]
        return _json_safe({"status": "success", **result})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/progress")
async def get_progress(job_id: str | None = None):
    progress = PROGRESS.snapshot(job_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Screening job not found.")
    return _json_safe(progress)
