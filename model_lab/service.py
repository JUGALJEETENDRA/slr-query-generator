from __future__ import annotations

import csv
import io
import json
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_ai.contracts import PaperAssessment, ReviewProtocol, ValidationReport, safe_maybe
from local_ai.engine import LocalAIError, OllamaStructuredEngine
from local_ai.evidence import evidence_lookup
from local_ai.hardware import inspect_hardware, resolve_runtime_profile
from local_ai.prompts import SYSTEM_RULES, protocol_critic_prompt, protocol_prompt
from local_ai.three_layer import ThreeLayerLocalOrchestrator
from local_ai.validator import validate_assessment


LAB_VERSION = "local-model-lab-v1"
RUNS_DIR = Path(__file__).resolve().parent / "runs"


@dataclass
class PaperInput:
    paper_id: str
    title: str
    abstract: str
    metadata: dict[str, str] = field(default_factory=dict)
    gold_decision: str = ""


@dataclass
class LabJob:
    job_id: str
    request: dict[str, Any]
    status: str = "queued"
    phase: str = "queued"
    error: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str = ""
    finished_at: str = ""
    current: int = 0
    total: int = 0
    current_model: str = ""
    current_paper: str = ""
    protocol: dict[str, Any] | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "phase": self.phase,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "current": self.current,
            "total": self.total,
            "current_model": self.current_model,
            "current_paper": self.current_paper,
            "summary": self.summary,
            "protocol": self.protocol,
        }


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_csv_papers(raw: bytes) -> list[PaperInput]:
    """Import title/abstract records without applying production eligibility rules."""
    text = raw.decode("utf-8-sig", errors="replace")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return []
    available = {str(key).casefold(): str(key) for key in rows[0].keys() if key}
    title_key = next((available[key.casefold()] for key in (
        "Title", "TI", "Article Title", "Document Title", "paper_title", "Name"
    ) if key.casefold() in available), None)
    abstract_key = next((available[key.casefold()] for key in (
        "Abstract", "AB", "Abstracts", "Summary", "Author Abstract", "Description"
    ) if key.casefold() in available), None)
    gold_key = next((available[key.casefold()] for key in (
        "Gold_Decision", "Gold Decision", "gold_decision"
    ) if key.casefold() in available), None)
    if not title_key and not abstract_key:
        raise ValueError("CSV needs a recognizable title or abstract column.")
    papers: list[PaperInput] = []
    for index, row in enumerate(rows, start=1):
        title = _clean(row.get(title_key or ""))
        abstract = _clean(row.get(abstract_key or ""))
        if not title and not abstract:
            continue
        metadata = {
            key: _clean(row.get(key)) for key in ("Year", "DOI", "Source title")
            if key in row and _clean(row.get(key))
        }
        gold_decision = _clean(row.get(gold_key or "")).upper()
        papers.append(PaperInput(
            str(index), title, abstract, metadata,
            gold_decision=gold_decision if gold_decision in {"KEEP", "REJECT", "UNSURE"} else "",
        ))
    return papers


def _profile_for(model: str, *, context: int, keep_alive: str = "10m"):
    base = resolve_runtime_profile("auto", "balanced")
    return replace(base, fast_model=model, strong_model=model, num_ctx=context, keep_alive=keep_alive, concurrency=1)


def _normalise_protocol(value: dict[str, Any], *, question: str, context: str, inclusion: str, exclusion: str, model: str) -> ReviewProtocol:
    normalised = ThreeLayerLocalOrchestrator._normalize_protocol_provenance(
        value, allow_user=bool(inclusion.strip() or exclusion.strip())
    )
    anchored = ThreeLayerLocalOrchestrator._anchor_rq_contract(normalised)
    protocol = ReviewProtocol.model_validate(anchored).model_copy(update={
        "research_question": question,
        "research_context": context,
        "model": model,
        "prompt_version": LAB_VERSION,
    })
    ThreeLayerLocalOrchestrator._validate_protocol(protocol, inclusion, exclusion)
    return protocol.with_identity()


def _assessment_prompt(protocol: ReviewProtocol, paper: PaperInput) -> str:
    units = list(evidence_lookup(paper.title, paper.abstract).values())
    payload = {
        "protocol": protocol.model_dump(mode="json"),
        "paper": {"id": paper.paper_id, "title": paper.title, "abstract": paper.abstract},
        "evidence_units": units,
    }
    return f"""{SYSTEM_RULES}

Assess this one paper against the immutable protocol. This is a local model evaluation run; make an independent
decision from this paper alone. For every protocol criterion, return exactly one criterion assessment.
Inclusion MET requires direct evidence that the paper's own contribution or explicit review scope satisfies the
criterion. Inclusion NOT_MET requires affirmative contrary evidence; silence is UNCLEAR. Exclusion MET requires
affirmative disqualifying evidence. Do not use keyword overlap, ontologies, or unstated domain assumptions.
KEEP requires every required inclusion MET and no exclusion MET. REJECT requires at least one evidence-backed
required inclusion NOT_MET or exclusion MET. Otherwise choose MAYBE. Cite only supplied evidence-unit IDs, never
invent quotes or IDs. Give concise audit rationales; never reveal hidden chain-of-thought.

INPUT:
{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}

Return a PaperAssessment matching this JSON schema:
{json.dumps(PaperAssessment.model_json_schema(), ensure_ascii=False)}
"""


def _critic_prompt(protocol: ReviewProtocol, paper: PaperInput, primary: PaperAssessment) -> str:
    units = list(evidence_lookup(paper.title, paper.abstract).values())
    payload = {
        "protocol": protocol.model_dump(mode="json"),
        "paper": {"id": paper.paper_id, "title": paper.title, "abstract": paper.abstract},
        "primary_assessment": primary.model_dump(mode="json"),
        "evidence_units": units,
    }
    return f"""{SYSTEM_RULES}

Act as an adversarial evidence critic. Independently re-evaluate the paper and return a complete replacement
PaperAssessment. Challenge unsupported KEEP decisions, absence-based REJECT decisions, reversed exclusion logic,
and evidence IDs that do not establish the claimed criterion. Do not preserve the primary decision merely because
it was supplied. Use only exact supplied evidence-unit IDs. A definitive decision must be evidence-safe; otherwise
return MAYBE. Keep all rationales concise and do not reveal hidden chain-of-thought.

INPUT:
{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}

Return a PaperAssessment matching this JSON schema:
{json.dumps(PaperAssessment.model_json_schema(), ensure_ascii=False)}
"""


def _evidence_payload(assessment: PaperAssessment, paper: PaperInput) -> list[dict[str, str]]:
    units = evidence_lookup(paper.title, paper.abstract)
    resolved: list[dict[str, str]] = []
    for criterion in assessment.criteria:
        for span in criterion.evidence:
            unit = units.get(span.evidence_id)
            if unit and unit["source"] == span.source:
                resolved.append({
                    "criterion_id": criterion.criterion_id,
                    "source": unit["source"],
                    "evidence_id": span.evidence_id,
                    "quote": unit["text"],
                })
    return resolved


def _generation_metrics(generation) -> dict[str, float | int]:
    return {
        "wall_seconds": generation.elapsed_seconds,
        "model_seconds": generation.model_duration_seconds,
        "total_seconds": generation.total_duration_seconds,
        "prompt_tokens": generation.prompt_tokens,
        "output_tokens": generation.output_tokens,
        "tokens_per_second": generation.tokens_per_second,
    }


def _run_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    decisions = {
        decision: sum(row["final"]["assessment"]["decision"] == decision for row in rows)
        for decision in ("KEEP", "MAYBE", "REJECT")
    }
    wall_times = [float(row["final"]["metrics"].get("wall_seconds") or 0.0) for row in rows]
    token_rates = [float(row["final"]["metrics"].get("tokens_per_second") or 0.0) for row in rows]
    return {
        "papers": len(rows),
        "decisions": decisions,
        "validated": sum(row["final"]["validation"]["valid"] for row in rows),
        "unresolved": sum(not row["final"]["validation"]["valid"] for row in rows),
        "critic_calls": sum(len(row["trace"]) > 1 for row in rows),
        "median_wall_seconds": round(statistics.median(wall_times), 3),
        "mean_tokens_per_second": round(statistics.mean(token_rates), 3),
        "exact_evidence_spans": sum(row["final"]["validation"]["exact_quote_count"] for row in rows),
    }


def _comparison(results: list[dict[str, Any]], models: list[str]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    by_model = {model: [row["candidates"][model] for row in results] for model in models}
    for index, left in enumerate(models):
        for right in models[index + 1:]:
            pairs = list(zip(by_model[left], by_model[right]))
            agreement = sum(
                a["final"]["assessment"]["decision"] == b["final"]["assessment"]["decision"]
                for a, b in pairs
            )
            disagreements = [
                {"paper_id": results[item]["paper"]["id"], "title": results[item]["paper"]["title"],
                 "left": a["final"]["assessment"]["decision"], "right": b["final"]["assessment"]["decision"]}
                for item, (a, b) in enumerate(pairs)
                if a["final"]["assessment"]["decision"] != b["final"]["assessment"]["decision"]
            ]
            comparisons.append({
                "left": left, "right": right,
                "agreement_rate": round(agreement / max(1, len(pairs)), 4),
                "disagreements": disagreements,
            })
    return comparisons


def _gold_metrics(results: list[dict[str, Any]], model: str) -> dict[str, Any]:
    labeled = [
        (row["paper"].get("gold_decision", ""), row["candidates"][model]["final"]["assessment"]["decision"])
        for row in results if row["paper"].get("gold_decision") in {"KEEP", "REJECT"}
    ]
    gold_keep = [decision for gold, decision in labeled if gold == "KEEP"]
    definitive_keep = [gold for gold, decision in labeled if decision == "KEEP"]
    return {
        "labeled_papers": len(labeled),
        "unsure_or_unlabeled": len(results) - len(labeled),
        "keep_maybe_recall": round(
            sum(decision in {"KEEP", "MAYBE"} for decision in gold_keep) / max(1, len(gold_keep)), 4
        ) if gold_keep else None,
        "false_reject_rate": round(
            sum(decision == "REJECT" for decision in gold_keep) / max(1, len(gold_keep)), 4
        ) if gold_keep else None,
        "definitive_keep_precision": round(
            sum(gold == "KEEP" for gold in definitive_keep) / max(1, len(definitive_keep)), 4
        ) if definitive_keep else None,
    }


class LocalModelLab:
    def __init__(self, runs_dir: Path = RUNS_DIR):
        self.runs_dir = runs_dir
        self.jobs: dict[str, LabJob] = {}
        self.lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        hardware = inspect_hardware()
        return {
            "lab_version": LAB_VERSION,
            "local_only": True,
            "hardware": asdict(hardware),
            "models": sorted(hardware.installed_models),
        }

    def create_job(self, request: dict[str, Any]) -> LabJob:
        question = _clean(request.get("research_question"))
        models = list(dict.fromkeys(_clean(item) for item in request.get("models", []) if _clean(item)))
        papers = request.get("papers") or []
        if len(question) < 3:
            raise ValueError("Enter a research question first.")
        if not models:
            raise ValueError("Choose at least one installed local model.")
        if not papers:
            raise ValueError("Add at least one paper or upload a CSV.")
        if len(models) > 4:
            raise ValueError("Compare up to four models at once to keep the experiment readable.")
        if len(papers) > 50:
            raise ValueError("Use at most 50 papers per lab run. Start small and iterate.")
        job = LabJob(job_id=uuid.uuid4().hex[:12], request={**request, "models": models})
        with self.lock:
            self.jobs[job.job_id] = job
        thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        thread.start()
        return job

    def get_job(self, job_id: str) -> LabJob | None:
        with self.lock:
            return self.jobs.get(job_id)

    def _compile_protocol(self, job: LabJob, engine: OllamaStructuredEngine, model: str) -> tuple[ReviewProtocol, list[dict[str, Any]]]:
        request = job.request
        question = _clean(request.get("research_question"))
        inclusion = _clean(request.get("inclusion_criteria"))
        exclusion = _clean(request.get("exclusion_criteria"))
        context = _clean(request.get("research_context"))
        initial = engine.generate(model, protocol_prompt(question, inclusion, exclusion, context), ReviewProtocol)
        protocol = _normalise_protocol(initial.value, question=question, context=context, inclusion=inclusion, exclusion=exclusion, model=model)
        critic = engine.generate(model, protocol_critic_prompt(protocol, inclusion, exclusion, context), ReviewProtocol)
        protocol = _normalise_protocol(critic.value, question=question, context=context, inclusion=inclusion, exclusion=exclusion, model=model)
        return protocol, [
            {"stage": "protocol", "model": model, **_generation_metrics(initial)},
            {"stage": "protocol_critic", "model": model, **_generation_metrics(critic)},
        ]

    def _assess(self, engine: OllamaStructuredEngine, model: str, protocol: ReviewProtocol, paper: PaperInput, *, critic_model: str = "") -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        try:
            primary_generation = engine.generate(model, _assessment_prompt(protocol, paper), PaperAssessment)
            primary = PaperAssessment.model_validate(primary_generation.value)
            primary_validation = validate_assessment(primary, protocol, paper.title, paper.abstract)
        except (LocalAIError, ValueError) as exc:
            primary = safe_maybe(f"Primary model failed: {exc}")
            primary_validation = ValidationReport(valid=False, errors=[str(exc)])
            primary_generation = None
        primary_payload = {
            "model": model,
            "assessment": primary.model_dump(mode="json"),
            "validation": primary_validation.model_dump(mode="json"),
            "evidence": _evidence_payload(primary, paper),
            "metrics": _generation_metrics(primary_generation) if primary_generation else {},
        }
        trace.append({"stage": "primary", **primary_payload})
        final = primary_payload
        risk = primary.decision == "MAYBE" or not primary_validation.valid or primary.confidence < 0.85
        if critic_model and risk:
            try:
                generation = engine.generate(critic_model, _critic_prompt(protocol, paper, primary), PaperAssessment)
                assessed = PaperAssessment.model_validate(generation.value)
                validation = validate_assessment(assessed, protocol, paper.title, paper.abstract)
                candidate = {
                    "model": critic_model,
                    "assessment": assessed.model_dump(mode="json"),
                    "validation": validation.model_dump(mode="json"),
                    "evidence": _evidence_payload(assessed, paper),
                    "metrics": _generation_metrics(generation),
                }
                trace.append({"stage": "adversarial_critic", **candidate})
                if validation.valid:
                    final = candidate
            except (LocalAIError, ValueError) as exc:
                trace.append({"stage": "adversarial_critic", "model": critic_model, "error": str(exc)})
        return {"trace": trace, "final": final}

    def _run(self, job: LabJob) -> None:
        request = job.request
        models = request["models"]
        papers = [PaperInput(
            str(item.get("id") or index + 1), _clean(item.get("title")), _clean(item.get("abstract")),
            dict(item.get("metadata") or {}), _clean(item.get("gold_decision")).upper(),
        ) for index, item in enumerate(request["papers"])]
        protocol_model = _clean(request.get("protocol_model")) or models[0]
        critic_model = _clean(request.get("critic_model")) if request.get("use_critic") else ""
        context_size = max(1024, min(8192, int(request.get("context_size") or 4096)))
        job.status, job.phase, job.started_at = "running", "compiling_protocol", datetime.now(timezone.utc).isoformat()
        job.total = len(models) * len(papers)
        try:
            engine = OllamaStructuredEngine(_profile_for(protocol_model, context=context_size))
            protocol, protocol_metrics = self._compile_protocol(job, engine, protocol_model)
            job.protocol = {"protocol": protocol.model_dump(mode="json"), "metrics": protocol_metrics}
            for model in models:
                job.phase, job.current_model = "screening", model
                model_engine = OllamaStructuredEngine(_profile_for(model, context=context_size))
                for paper in papers:
                    job.current_paper = paper.paper_id
                    result = self._assess(model_engine if not critic_model or critic_model == model else model_engine, model, protocol, paper, critic_model=critic_model)
                    existing = next((row for row in job.results if row["paper"]["id"] == paper.paper_id), None)
                    if existing is None:
                        existing = {"paper": {
                            "id": paper.paper_id, "title": paper.title, "abstract": paper.abstract,
                            "metadata": paper.metadata, "gold_decision": paper.gold_decision,
                        }, "candidates": {}}
                        job.results.append(existing)
                    existing["candidates"][model] = result
                    job.current += 1
            per_model = {
                model: _run_metrics([row["candidates"][model] for row in job.results]) | {
                    "gold": _gold_metrics(job.results, model)
                }
                for model in models
            }
            job.summary = {
                "lab_version": LAB_VERSION,
                "papers": len(papers),
                "models": per_model,
                "comparison": _comparison(job.results, models),
                "protocol_model": protocol_model,
                "critic_model": critic_model or None,
            }
            job.status, job.phase = "finished", "complete"
            self._persist(job)
        except Exception as exc:  # User-visible lab job errors must never crash the server.
            job.status, job.phase, job.error = "error", "failed", str(exc)
        finally:
            job.finished_at = datetime.now(timezone.utc).isoformat()

    def _persist(self, job: LabJob) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        payload = {"job": job.snapshot(), "results": job.results, "request": job.request}
        (self.runs_dir / f"{job.job_id}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


LAB = LocalModelLab()
