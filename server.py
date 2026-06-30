from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
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
from bulk_screen import screen_csv
from litsync import parse_upload_files, deduplicate
import os
import pandas as pd

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

@app.get("/")
async def serve_ui():
    return FileResponse("archive/slr_query_generator.html")

local_client = instructor.from_openai(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama-local"),
    mode=instructor.Mode.MD_JSON
)
LOCAL_MODEL = "qwen2.5:7b"

class QuestionRequest(BaseModel):
    question: str

class ScreenRequest(BaseModel):
    question: str
    title: str
    abstract: str

class FinalizeRequest(BaseModel):
    titles: List[str]

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


@app.post("/screen_csv")
async def screen_csv_endpoint(
    question: str = Form(...),
    mode: str = Form("local"),
    embedding_threshold: float = Form(0.35),
    batch_size: int = Form(10),
    file: UploadFile = File(...)
):
    try:
        csv_path = os.path.join(UPLOAD_DIR, os.path.basename(file.filename or "papers.csv"))

        with open(csv_path, "wb") as buffer:
            buffer.write(await file.read())

        summary = await run_in_threadpool(
            screen_csv,
            csv_path=csv_path,
            research_question=question,
            mode=mode,
            embedding_threshold=embedding_threshold,
            batch_size=batch_size,
        )

        return {
            "status": "success",
            "mode": mode,
            "summary": summary,
            "download_url": "http://localhost:8000/outputs/screened.csv",
            "summary_url": "http://localhost:8000/outputs/summary.csv"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/maybe_papers")
async def get_maybe_papers():
    path = os.path.join(OUTPUT_DIR, "maybe_studies.csv")
    if not os.path.exists(path):
        return {"status": "error", "message": "No maybe papers found. Run the screener first."}
    df = pd.read_csv(path)
    papers = df[["Title", "Abstract", "Reason"]].fillna("").to_dict(orient="records")
    return {"status": "success", "papers": papers}


@app.post("/finalize")
async def finalize_endpoint(req: FinalizeRequest):
    try:
        maybe_path    = os.path.join(OUTPUT_DIR, "maybe_studies.csv")
        included_path = os.path.join(OUTPUT_DIR, "included_studies.csv")
        final_path    = os.path.join(OUTPUT_DIR, "final_included.csv")

        if not os.path.exists(maybe_path):
            return {"status": "error", "message": "No maybe papers file found."}

        maybe_df    = pd.read_csv(maybe_path)
        selected_df = maybe_df[maybe_df["Title"].isin(req.titles)]

        base_df = pd.read_csv(included_path) if os.path.exists(included_path) else pd.DataFrame()
        final_df = pd.concat([base_df, selected_df], ignore_index=True).drop_duplicates(subset=["Title"])
        final_df.to_csv(final_path, index=False)
        final_df.to_csv(included_path, index=False)

        return {
            "status": "success",
            "added": len(selected_df),
            "total": len(final_df),
            "download_url": "http://localhost:8000/outputs/included_studies.csv"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
