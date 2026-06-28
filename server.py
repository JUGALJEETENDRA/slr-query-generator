from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
import instructor
import re

# Keep your imports—these are the "engine"
from extractor import extract_5_facets
from generator import expand_base_synonyms
from acronym_expander import expand_acronym_layer
from classifier import classify_extracted_context
from registries import inject_implicit_academic_layers
from ontology_expander import expand_ontology_layer
from comparator_registry import expand_comparator_registry
from validator import run_validation_sieve
from compiler import compile_boolean_query
from schema import SLRQueryContext
from screener import screen_paper
# CHANGED: import PROGRESS from bulk_screen
from bulk_screen import screen_csv, PROGRESS, SCREENING_SESSION
from litsync import parse_upload_files, deduplicate
import os
import pandas as pd

# ===== NEW IMPORTS FOR ASYNC SCREENING =====
from threading import Thread
import uuid

# ===== IMPORT CONFIG DEFAULTS =====
from config import (
    HYBRID_SCREENING_ENABLED,
    FIRST_STAGE_MODEL,
    SECOND_STAGE_MODEL,
)

# ===== DIRECTORIES – MUST EXIST BEFORE MOUNTING =====
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== FASTAPI APP =====
app = FastAPI(title="SLR Query Generator API")

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

local_client = instructor.from_openai(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama-local"),
    mode=instructor.Mode.MD_JSON
)
LOCAL_MODEL = "qwen2.5:3b"
DEFAULT_MODEL = LOCAL_MODEL  # default model for screen_csv endpoint

class QuestionRequest(BaseModel):
    question: str

class ScreenRequest(BaseModel):
    question: str
    title: str
    abstract: str

class FinalizeRequest(BaseModel):
    titles: List[str] = []
    papers: List[dict] = []

def compress_schema_for_ieee(context: SLRQueryContext) -> SLRQueryContext:
    import copy
    compressed = copy.deepcopy(context)
    merged_tech = list(context.technology[:2]) + list(context.comparison[:1])
    compressed.technology = [t.replace("*", "") for t in merged_tech if t]
    compressed.domain = [t.replace("*", "") for t in context.domain[:2]]
    compressed.outcomes = [t.replace("*", "") for t in context.outcomes[:2]]
    compressed.comparison = []
    compressed.context = []
    return compressed

@app.post("/generate")
async def generate(req: QuestionRequest):
    try:
        # STAGE 1: Extract
        raw = extract_5_facets(local_client, LOCAL_MODEL, req.question)
        s1 = SLRQueryContext(
            technology=raw.primary_paradigm,
            comparison=raw.comparator_baseline,
            domain=raw.domain_context,
            context=[],
            outcomes=raw.outcome_variables
        )

        # STAGE 2: Pipeline
        s2 = expand_base_synonyms(local_client, LOCAL_MODEL, s1)
        s3 = expand_acronym_layer(s2)
        primary_domain = classify_extracted_context(s3)
        s3_hydrated = inject_implicit_academic_layers(s3, primary_domain)
        s4 = expand_ontology_layer(s3_hydrated, primary_domain)
        s4_compared = expand_comparator_registry(s4)
        s5 = run_validation_sieve(s4_compared)

        # STAGE 3: Compile
        base_query = compile_boolean_query(s5).replace("\n", " ")
        ieee_safe = compress_schema_for_ieee(s5)
        ieee_query = compile_boolean_query(ieee_safe).replace("\n", " ")

        return {
            "status": "success",
            "google_scholar": base_query,
            "scopus": f"TITLE-ABS-KEY({base_query})",
            "web_of_science": f"TS=({base_query})",
            "ieee_xplore": ieee_query,
            "pubmed": re.sub(r'"([^"]+)"', r'"\1"[tiab]', base_query),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/screen")
async def screen(req: ScreenRequest):
    try:
        result = screen_paper(
            title=req.title,
            abstract=req.abstract,
            research_question=req.question
        )

        return {
            "status": "success",
            "decision": result["decision"],
            "reason": result["reason"]
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
    hybrid_enabled=False,
    first_stage_model=None,
    second_stage_model=None,
    max_rows=None,
):

    try:
        screen_csv(
            csv_path=csv_path,
            research_question=question,
            mode=mode,
            model=model,
            progress_job_id=job_id,
            hybrid_enabled=hybrid_enabled,
            first_stage_model=first_stage_model,
            second_stage_model=second_stage_model,
            max_rows=max_rows,           # FIX 1: now passed
        )
    except Exception as e:
        PROGRESS.fail(job_id, e)

# ===== REPLACED /screen_csv ENDPOINT (NOW ASYNC WITH JOB ID) =====
@app.post("/screen_csv")
async def screen_csv_endpoint(
    question: str = Form(...),
    mode: str = Form("local"),
    model: str = Form(DEFAULT_MODEL),
    file: UploadFile = File(...),
    hybrid_enabled: bool = Form(HYBRID_SCREENING_ENABLED),          # FIX 2: use config default
    first_stage_model: str = Form(FIRST_STAGE_MODEL),              # FIX 2
    second_stage_model: str = Form(SECOND_STAGE_MODEL),            # FIX 2
    max_rows: int | None = Form(None),
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

    Thread(
        target=run_screening,
        args=(
            job_id,
            csv_path,
            question,
            mode,
            model,
            hybrid_enabled,
            first_stage_model,
            second_stage_model,
            max_rows,
        ),
        daemon=True,
    ).start()


    return {
        "status": "started",
        "job_id": job_id,
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