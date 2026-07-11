from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import re

# Keep your imports—these are the "engine"
from direct_ai_generator import generate_query
from screening_strategies import DEFAULT_SCREENING_STRATEGY, screen_candidate
# CHANGED: import PROGRESS from bulk_screen
from bulk_screen import screen_csv, PROGRESS, SCREENING_SESSION
from litsync import parse_upload_files, deduplicate
import os
from runtime_config import get_model_judge_config, print_model_judge_config

# ===== NEW IMPORTS FOR ASYNC SCREENING =====
from threading import Thread
import uuid

# ===== IMPORT CONFIG DEFAULTS =====
from config import (
    TWO_STAGE_SCREENING_ENABLED,
    FIRST_STAGE_MODEL,
    SECOND_STAGE_MODEL,
    GEMINI_WEB_BATCH_SIZE,
    GEMINI_WEB_PROFILE_DIR,
)
from processing_engines import (
    DEFAULT_PROCESSING_ENGINE,
    normalize_processing_engine,
    resolve_processing_engine,
)

# ===== DIRECTORIES – MUST EXIST BEFORE MOUNTING =====
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== FASTAPI APP =====
app = FastAPI(title="SLR Query Generator API")


@app.on_event("startup")
async def _print_runtime_model_config():
    print_model_judge_config()

# Mount static files for outputs directory
app.mount(
    "/outputs",
    StaticFiles(directory="outputs"),
    name="outputs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOCAL_MODEL = "qwen2.5:3b"
DEFAULT_MODEL = LOCAL_MODEL  # default model for screen_csv endpoint


def _normalize_row_limit(value):
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return None
    return limit if limit > 0 else None


class QuestionRequest(BaseModel):
    question: str

class ScreenRequest(BaseModel):
    question: str
    title: str
    abstract: str
    semantic_strategy: str = DEFAULT_SCREENING_STRATEGY
    processing_engine: str = DEFAULT_PROCESSING_ENGINE
    model: str = DEFAULT_MODEL
    gemini_api_key: str = ""

class FinalizeRequest(BaseModel):
    titles: List[str] = []
    papers: List[dict] = []


@app.get("/debug/model-judge-config")
async def debug_model_judge_config():
    cfg = get_model_judge_config()
    return {
        "process_id": os.getpid(),
        "enable_model_judges": cfg["enable_model_judges"],
        "model_judge_mode": cfg["model_judge_mode"],
        "model_judge_profile": cfg["model_judge_profile"],
        "enable_hf_model_loading": cfg["enable_hf_model_loading"],
        "enable_hf_model_download": cfg["enable_hf_model_download"],
        "enable_reranker_judge": cfg["enable_reranker_judge"],
        "enable_nli_judge": cfg["enable_nli_judge"],
        "enable_zero_shot_judge": cfg["enable_zero_shot_judge"],
        "enable_llm_judge": cfg["enable_llm_judge"],
        "source": cfg["source"],
        "raw_env": cfg["raw_env"],
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
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/screen")
async def screen(req: ScreenRequest):
    try:
        selected_engine = normalize_processing_engine(req.processing_engine)
        with resolve_processing_engine(
            selected_engine,
            gemini_api_key=req.gemini_api_key or None,
        ) as inference_engine:
            result = screen_candidate(
                title=req.title,
                abstract=req.abstract,
                research_question=req.question,
                strategy=req.semantic_strategy,
                mode=selected_engine,
                model=req.model,
                inference_engine=inference_engine,
            )

        return {
            "status": "success",
            "decision": result["decision"],
            "reason": result["reason"],
            "confidence": result.get("confidence", ""),
            "litsync_decision": result.get("litsync_decision", ""),
            "litsync_reason": result.get("litsync_reason", ""),
            "litsync_confidence": result.get("litsync_confidence", ""),
            "direct_ai_decision": result.get("direct_ai_decision", ""),
            "direct_ai_reason": result.get("direct_ai_reason", ""),
            "direct_ai_confidence": result.get("direct_ai_confidence", ""),
            "final_fused_decision": result.get("decision", ""),
            "final_fused_reason": result.get("reason", ""),
            "agreement": result.get("agreement", ""),
            "fusion_policy": result.get("fusion_policy", ""),
            "metadata": result.get("metadata", {}),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/litsync")
async def litsync_endpoint(files: List[UploadFile] = File(...)):
    try:
        saved_paths = []
        for f in files:
            if not f.filename:
                continue
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in ['.csv', '.xls', '.xlsx']:
                continue
            out_path = os.path.join(UPLOAD_DIR, f.filename)
            with open(out_path, "wb") as buf:
                buf.write(await f.read())
            saved_paths.append(out_path)

        combined_mapped, total_initial = parse_upload_files(saved_paths)
        deduped_df, removed = deduplicate(combined_mapped)
        deduped_count = int(len(deduped_df.index))

        from datetime import datetime
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
        out_name = f"LitSync_Clean_Dataset_{date_str}.csv"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        deduped_df.to_csv(out_path, index=False)

        return {
            "status": "success",
            "counts": {
                "initial": int(total_initial),
                "deduped": deduped_count,
                "duplicates_removed": int(removed)
            },
            "download_url": f"http://localhost:8000/outputs/{out_name}",
            "output_filename": out_name
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ===== NEW HELPER FUNCTION FOR BACKGROUND SCREENING =====
def run_screening(
    job_id,
    csv_path,
    question,
    mode,
    model,
    two_stage_enabled=False,
    first_stage_model=None,
    second_stage_model=None,
    max_rows=None,
    semantic_strategy=DEFAULT_SCREENING_STRATEGY,
    screening_engine=None,
    gemini_web_batch_size=GEMINI_WEB_BATCH_SIZE,
    gemini_web_profile_dir=GEMINI_WEB_PROFILE_DIR,
    gemini_api_key="",
    inclusion_criteria="",
    exclusion_criteria="",
):
    import traceback

    try:
        selected_engine = normalize_processing_engine(screening_engine or mode)
        normalized_max_rows = _normalize_row_limit(max_rows)
        screen_csv(
            csv_path=csv_path,
            research_question=question,
            mode=selected_engine,
            model=model,
            progress_job_id=job_id,
            two_stage_enabled=two_stage_enabled,
            first_stage_model=first_stage_model,
            second_stage_model=second_stage_model,
            max_rows=normalized_max_rows,
            semantic_strategy=semantic_strategy,
            screening_engine=selected_engine,
            gemini_web_profile_dir=gemini_web_profile_dir,
            gemini_web_batch_size=gemini_web_batch_size,
            gemini_api_key=gemini_api_key or None,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
        )
    except Exception as e:
        # Do not swallow exceptions: log full traceback.
        traceback.print_exc()
        PROGRESS.fail(job_id, e)
    finally:
        # Ensure job is not left stuck in a "running" state.
        # This prevents future runs from failing with "Another screening job is already running".
        try:
            snapshot = PROGRESS.snapshot()
            if snapshot.get("job_id") == job_id and snapshot.get("status") in {"starting", "running"}:
                PROGRESS.fail(job_id, snapshot.get("error") or "Screening stopped before completion.")
        except Exception:
            # If cleanup itself fails, at least don’t hide the original error.
            traceback.print_exc()


# ===== REPLACED /screen_csv ENDPOINT (NOW ASYNC WITH JOB ID) =====
@app.post("/screen_csv")
async def screen_csv_endpoint(
    question: str = Form(...),
    mode: str = Form("local"),
    model: str = Form(DEFAULT_MODEL),
    file: UploadFile = File(...),
    two_stage_enabled: bool = Form(TWO_STAGE_SCREENING_ENABLED),
    first_stage_model: str = Form(FIRST_STAGE_MODEL),              # FIX 2
    second_stage_model: str = Form(SECOND_STAGE_MODEL),            # FIX 2
    max_rows: int | None = Form(None),
    semantic_strategy: str = Form(DEFAULT_SCREENING_STRATEGY),
    screening_engine: str = Form(DEFAULT_PROCESSING_ENGINE),
    gemini_web_batch_size: int = Form(GEMINI_WEB_BATCH_SIZE),
    gemini_web_profile_dir: str = Form(GEMINI_WEB_PROFILE_DIR),
    gemini_api_key: str = Form(""),
    inclusion_criteria: str = Form(""),
    exclusion_criteria: str = Form(""),
):

    job_id = str(uuid.uuid4())
    if not PROGRESS.start_job(job_id):
        raise HTTPException(
            status_code=409,
            detail="Another screening job is already running."
        )

    csv_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    try:
        with open(csv_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        PROGRESS.fail(job_id, e)
        raise

    normalized_max_rows = _normalize_row_limit(max_rows)

    Thread(
        target=run_screening,
        args=(
            job_id,
            csv_path,
            question,
            mode,
            model,
            two_stage_enabled,
            first_stage_model,
            second_stage_model,
            normalized_max_rows,
            semantic_strategy,
            screening_engine,
            gemini_web_batch_size,
            gemini_web_profile_dir,
            gemini_api_key,
            inclusion_criteria,
            exclusion_criteria,
        ),
        daemon=True,
    ).start()


    return {
        "status": "started",
        "job_id": job_id,
        "screening_engine": normalize_processing_engine(screening_engine),
        "row_limit_applied": normalized_max_rows is not None,
        "row_limit_value": normalized_max_rows or "",
    }

@app.get("/maybe_papers")
async def get_maybe_papers():
    papers = [
        row for row in SCREENING_SESSION.snapshot()
        if row.get("Decision") == "MAYBE"
    ]
    return {"status": "success", "papers": papers}

@app.get("/screening_results")
async def get_screening_results():
    papers = SCREENING_SESSION.snapshot()
    return {
        "status": "success",
        "papers": papers,
        "counts": SCREENING_SESSION.counts(papers),
    }

@app.post("/finalize")
async def finalize_endpoint(req: FinalizeRequest):
    try:
        papers = req.papers

        if not papers and req.titles:
            selected_titles = set(req.titles)
            papers = []
            for row in SCREENING_SESSION.snapshot():
                edited = dict(row)
                if edited.get("Decision") == "MAYBE" and edited.get("Title") in selected_titles:
                    edited["Decision"] = "KEEP"
                papers.append(edited)

        finalized = SCREENING_SESSION.finalize(papers, OUTPUT_DIR)

        files = {}
        for key, path in finalized["files"].items():
            exists = os.path.exists(path)
            files[key] = {
                "available": exists,
                "download_url": f"http://localhost:8000/outputs/{os.path.basename(path)}" if exists else None,
            }

        return {
            "status": "success",
            "counts": finalized["counts"],
            "files": files,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# NEW ENDPOINT: expose progress from bulk_screen
@app.get("/progress")
async def get_progress():
    return PROGRESS.snapshot()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
