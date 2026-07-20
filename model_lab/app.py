from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .service import LAB, RUNS_DIR, parse_csv_papers


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static" / "index.html"
app = FastAPI(title="LitSync Local Model Evaluation Lab", docs_url=None, redoc_url=None)


class PaperPayload(BaseModel):
    id: str = ""
    title: str = ""
    abstract: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)
    gold_decision: str = ""


class ExperimentRequest(BaseModel):
    research_question: str
    research_context: str = ""
    inclusion_criteria: str = ""
    exclusion_criteria: str = ""
    models: list[str] = Field(default_factory=list)
    protocol_model: str = ""
    use_critic: bool = False
    critic_model: str = ""
    context_size: int = 4096
    papers: list[PaperPayload] = Field(default_factory=list)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC)


@app.get("/api/status")
def status():
    return LAB.status()


@app.post("/api/import-csv")
async def import_csv(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Upload a CSV file.")
    try:
        papers = parse_csv_papers(await file.read())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"papers": [{
        "id": paper.paper_id, "title": paper.title, "abstract": paper.abstract,
        "metadata": paper.metadata, "gold_decision": paper.gold_decision,
    } for paper in papers], "count": len(papers)}


@app.post("/api/jobs")
def create_job(payload: ExperimentRequest):
    try:
        job = LAB.create_job({**payload.model_dump(), "papers": [paper.model_dump() for paper in payload.papers]})
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(job.snapshot(), status_code=202)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = LAB.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown lab run.")
    return job.snapshot()


@app.get("/api/jobs/{job_id}/results")
def job_results(job_id: str):
    job = LAB.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown lab run.")
    return {"job": job.snapshot(), "results": job.results, "request": job.request}


@app.get("/api/jobs/{job_id}/download")
def download_job(job_id: str):
    target = RUNS_DIR / f"{job_id}.json"
    if not target.exists():
        raise HTTPException(404, "Run has not been saved yet.")
    return FileResponse(target, media_type="application/json", filename=f"litsync-model-lab-{job_id}.json")
