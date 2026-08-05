from __future__ import annotations

import json
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from pydantic import Field, create_model, field_validator

from litsync_app.screening.local.engine import LocalAIOutputError
from litsync_app.screening.local_v2.contracts import (
    CriterionAssessment,
    EvidenceCitation,
    PolicyResult,
    ScreeningProtocolV2,
    StrictModel,
)
from litsync_app.screening.local_v2.evidence import EvidenceBatchValidation, evidence_lookup
from litsync_app.screening.local_v2.runner import LocalV2Paper, PaperTextStatus


FAST_RUNNER_VERSION = "local-v2-fast-binary-keep-gate-v4"
FAST_CHECKPOINT_VERSION = "local-v2-fast-binary-keep-gate-checkpoint-v4"
DEFAULT_PRIMARY_MODEL = "qwen3.5:4b"
DEFAULT_REVIEW_MODEL = "qwen3:8b"

FastStage = Literal["primary", "reviewer"]
FastRoute = Literal[
    "FAST_BINARY_PRIMARY_KEEP",
    "FAST_BINARY_REVIEWER_KEEP",
    "FAST_BINARY_REVIEWER_REJECT",
]


class FastStructuredEngine(Protocol):
    def generate(self, model: str, prompt: str, schema: type[StrictModel], *, timeout_seconds: float | None = None) -> Any: ...


class FastModelPlan(StrictModel):
    primary_model: str = Field(default=DEFAULT_PRIMARY_MODEL, min_length=1, max_length=200)
    review_model: str = Field(default=DEFAULT_REVIEW_MODEL, min_length=1, max_length=200)
    primary_timeout_seconds: float = Field(default=180.0, gt=0, le=900)
    review_timeout_seconds: float = Field(default=240.0, gt=0, le=900)

    @field_validator("primary_model", "review_model")
    @classmethod
    def strip_model_name(cls, value: str) -> str:
        return value.strip()


class FastDecisionItem(StrictModel):
    p: str = Field(min_length=1, max_length=200)
    d: Literal["K", "R"]
    c: list[str] = Field(default_factory=list, max_length=20)
    e: list[str] = Field(default_factory=list, max_length=2)

    @field_validator("p")
    @classmethod
    def strip_paper_id(cls, value: str) -> str:
        return value.strip()

    @field_validator("c", "e")
    @classmethod
    def clean_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


class FastDecisionBatch(StrictModel):
    items: list[FastDecisionItem] = Field(min_length=1)


@lru_cache(maxsize=32)
def fast_assessment_batch_schema(*, paper_count: int, criterion_count: int) -> type[FastDecisionBatch]:
    if paper_count < 1 or criterion_count < 1:
        raise ValueError("paper_count and criterion_count must be positive")
    return cast(type[FastDecisionBatch], create_model(
        f"FastBinaryDecisionBatch{paper_count}",
        __base__=FastDecisionBatch,
        items=(list[FastDecisionItem], Field(min_length=paper_count, max_length=paper_count)),
    ))


class FastStageOutcome(StrictModel):
    stage: FastStage
    model: str
    decision: Literal["K", "R"]
    criterion_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    assessments: list[CriterionAssessment] = Field(default_factory=list)
    evidence_result: EvidenceBatchValidation | None = None
    elapsed_seconds: float = Field(default=0.0, ge=0)


class FastPaperRunResult(StrictModel):
    runner_version: Literal[FAST_RUNNER_VERSION] = FAST_RUNNER_VERSION
    status: Literal["COMPLETED"] = "COMPLETED"
    protocol_id: str
    paper: LocalV2Paper
    text_status: PaperTextStatus
    route: FastRoute
    primary: FastStageOutcome
    reviewer: FastStageOutcome | None = None
    final_policy: PolicyResult
    model_call_count: int = Field(ge=1, le=2)
    elapsed_seconds: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    safe_fallback: Literal[False] = False


class FastBatchMetrics(StrictModel):
    total_papers: int = Field(ge=1)
    keep_count: int = Field(ge=0)
    maybe_count: Literal[0] = 0
    reject_count: int = Field(ge=0)
    no_screenable_text_count: Literal[0] = 0
    safe_fallback_count: Literal[0] = 0
    resumed_count: int = Field(ge=0)
    fresh_count: int = Field(ge=0)
    model_call_count: int = Field(ge=0)
    primary_batch_count: int = Field(ge=0)
    review_batch_count: int = Field(ge=0)
    review_used_count: int = Field(ge=0)
    primary_papers_assessed: int = Field(ge=0)
    primary_direct_keep_count: int = Field(ge=0)
    reviewer_candidate_count: int = Field(ge=0)
    reviewer_papers_assessed: int = Field(ge=0)
    reviewer_keep_count: int = Field(ge=0)
    reviewer_reject_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    model_elapsed_seconds: float = Field(ge=0)
    checkpoint_write_count: int = Field(ge=0)
    route_counts: dict[str, int] = Field(default_factory=dict)


class FastBatchRunResult(StrictModel):
    batch_runner_version: Literal[FAST_RUNNER_VERSION] = FAST_RUNNER_VERSION
    batch_id: str
    protocol_id: str
    model_plan: FastModelPlan
    results: list[FastPaperRunResult]
    metrics: FastBatchMetrics
    checkpoint_path: str | None = None
    checkpoint_disposition: Literal["NOT_REQUESTED", "DISABLED", "NOT_FOUND", "LOADED", "IGNORED_INVALID", "IGNORED_MISMATCH"]
    checkpoint_warnings: list[str] = Field(default_factory=list)
    resumed_positions: list[int] = Field(default_factory=list)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_protocol_identity(protocol: ScreeningProtocolV2) -> None:
    canonical = protocol.model_copy(update={"protocol_id": ""}).with_identity()
    if not protocol.protocol_id or canonical.protocol_id != protocol.protocol_id:
        raise ValueError("protocol_id does not match the current protocol contents")


def _normalize_papers(papers: Iterable[LocalV2Paper | Mapping[str, Any]]) -> list[LocalV2Paper]:
    normalized = [value if isinstance(value, LocalV2Paper) else LocalV2Paper.model_validate(value) for value in papers]
    if not normalized:
        raise ValueError("at least one paper is required")
    duplicates = [paper_id for paper_id, count in Counter(p.paper_id for p in normalized).items() if count > 1]
    if duplicates:
        raise ValueError("paper_id values must be unique: " + ", ".join(sorted(duplicates)))
    missing = [paper.paper_id for paper in normalized if not paper.title and not paper.abstract]
    if missing:
        raise ValueError("Fast Binary requires a title or abstract; affected source rows: " + ", ".join(missing))
    return normalized


def _paper_text_status(paper: LocalV2Paper) -> PaperTextStatus:
    if paper.title and paper.abstract:
        return "TITLE_AND_ABSTRACT"
    return "TITLE_ONLY" if paper.title else "ABSTRACT_ONLY"


def _text_warnings(status: PaperTextStatus) -> list[str]:
    if status == "TITLE_ONLY":
        return ["Abstract unavailable; binary AI decision used title evidence."]
    if status == "ABSTRACT_ONLY":
        return ["Title unavailable; binary AI decision used abstract evidence."]
    return []


def _paper_fingerprint(paper: LocalV2Paper) -> str:
    return sha256(_canonical_json(paper.model_dump(mode="json")).encode()).hexdigest()


def _build_batch_id(protocol: ScreeningProtocolV2, papers: list[LocalV2Paper], *, plan: FastModelPlan, primary_batch_size: int, review_batch_size: int) -> str:
    return sha256(_canonical_json({
        "runner_version": FAST_RUNNER_VERSION, "checkpoint_version": FAST_CHECKPOINT_VERSION,
        "protocol": protocol.model_dump(mode="json"), "model_plan": plan.model_dump(mode="json"),
        "primary_batch_size": primary_batch_size, "review_batch_size": review_batch_size,
        "papers": [paper.model_dump(mode="json") for paper in papers],
    }).encode()).hexdigest()[:20]


def _prompt_evidence_units(paper: LocalV2Paper) -> list[dict[str, str]]:
    maximum = max(500, int(os.getenv("LOCAL_V2_FAST_EVIDENCE_CHARS_PER_PAPER", "2500")))
    used, output = 0, []
    for unit in evidence_lookup(paper.title, paper.abstract).values():
        text = unit.text[: max(0, maximum - used)]
        if text:
            output.append({"evidence_id": unit.evidence_id, "source": unit.source, "text": text})
            used += len(text)
    return output


def _build_prompt(protocol: ScreeningProtocolV2, papers: list[LocalV2Paper], *, stage: FastStage) -> str:
    if stage == "primary":
        instructions = (
            "You are the first-stage high-precision KEEP gate. Choose K only when the "
            "supplied title and abstract give positive, substantive evidence that the paper "
            "satisfies the required inclusion criteria and does not match an exclusion criterion. "
            "Choose R for every other paper so a stronger independent reviewer can make the "
            "final decision. A topically related paper is not automatically K: incidental mention, "
            "background discussion, an excluded publication type, an excluded application domain, "
            "or lack of the required substantive contribution must be R."
        )
    else:
        instructions = (
            "You are the blind independent final binary reviewer. Make the final K or R decision "
            "using only the supplied research question, research context, criteria, title, abstract, "
            "and evidence units. Choose R when any exclusion criterion is positively supported, "
            "including excluded publication types or application domains. Choose K only when the "
            "paper substantively satisfies the inclusion criteria and required application context. "
            "Incidental mention or weak topical overlap is not enough for K."
        )
    required = {"items": [{"p": paper.paper_id, "d": "K or R", "c": ["criterion ids"], "e": ["exact evidence unit ids"]} for paper in papers]}
    return "\n\n".join([
        instructions + " Return exactly one K or R for every paper. Both K and R require at least one "
        "relevant or decisive protocol criterion id and at least one exact evidence-unit id. Preserve paper ids exactly. "
        "Do not return prose, explanations, or any decision code other than K or R. Return one JSON object only.",
        "PROTOCOL_JSON:\n" + _canonical_json({"protocol_id": protocol.protocol_id, "research_question": protocol.research_question, "research_context": protocol.research_context, "criteria": [{"id": c.id, "role": c.role, "description": c.description} for c in protocol.criteria]}),
        "PAPERS_JSON:\n" + _canonical_json([{ "paper_id": p.paper_id, "evidence_units": _prompt_evidence_units(p)} for p in papers]),
        "REQUIRED_OUTPUT_SHAPE:\n" + _canonical_json(required),
    ])


def _split_for_prompt_limit(protocol: ScreeningProtocolV2, papers: list[LocalV2Paper], *, stage: FastStage, requested_batch_size: int) -> list[list[LocalV2Paper]]:
    maximum = max(4000, int(os.getenv("LOCAL_V2_FAST_MAX_BATCH_PROMPT_CHARS", "14000")))
    pending = [papers[index:index + requested_batch_size] for index in range(0, len(papers), requested_batch_size)]
    output: list[list[LocalV2Paper]] = []
    while pending:
        chunk = pending.pop(0)
        if len(chunk) > 1 and len(_build_prompt(protocol, chunk, stage=stage)) > maximum:
            midpoint = len(chunk) // 2
            pending[0:0] = [chunk[:midpoint], chunk[midpoint:]]
        else:
            output.append(chunk)
    return output


def _validate_item(item: FastDecisionItem, *, protocol: ScreeningProtocolV2, paper: LocalV2Paper, stage: FastStage, model: str, elapsed: float) -> FastStageOutcome:
    allowed = {criterion.id for criterion in protocol.criteria}
    unknown_criteria = sorted(set(item.c) - allowed)
    units = evidence_lookup(paper.title, paper.abstract)
    unknown_evidence = sorted(set(item.e) - set(units))
    if unknown_criteria:
        raise ValueError("invalid criterion ids for " + paper.paper_id + ": " + ", ".join(unknown_criteria))
    if unknown_evidence:
        raise ValueError("invalid exact evidence ids for " + paper.paper_id + ": " + ", ".join(unknown_evidence))
    if not item.c or not item.e:
        raise ValueError("binary decision requires criterion and exact evidence for " + paper.paper_id)
    assessments = [CriterionAssessment(
        criterion_id=criterion_id,
        relation="DIRECT_CONTRADICTION" if item.d == "R" else "DIRECT_SUPPORT",
        rationale="Validated compact binary AI decision.",
        evidence=[EvidenceCitation(evidence_id=evidence_id, source=units[evidence_id].source, quote=units[evidence_id].text) for evidence_id in item.e],
    ) for criterion_id in item.c]
    return FastStageOutcome(stage=stage, model=model, decision=item.d, criterion_ids=item.c, evidence_ids=item.e, assessments=assessments, elapsed_seconds=round(max(elapsed, 0.0001), 4))


def _call_chunk(engine: FastStructuredEngine, protocol: ScreeningProtocolV2, chunk: list[LocalV2Paper], *, stage: FastStage, model: str, timeout_seconds: float) -> tuple[dict[str, FastStageOutcome], int, float]:
    schema = fast_assessment_batch_schema(paper_count=len(chunk), criterion_count=len(protocol.criteria))
    prompt = _build_prompt(protocol, chunk, stage=stage)
    elapsed_total = 0.0
    last_error: Exception | None = None
    for _ in range(2):
        started = time.perf_counter()
        try:
            generated = engine.generate(model, prompt, schema, timeout_seconds=timeout_seconds)
            elapsed = max(time.perf_counter() - started, float(getattr(generated, "elapsed_seconds", 0.0) or 0.0), 0.0001)
            elapsed_total += elapsed
            raw = generated.value if hasattr(generated, "value") else generated
            parsed = schema.model_validate(raw)
            by_id: dict[str, list[FastDecisionItem]] = {}
            for item in parsed.items:
                by_id.setdefault(item.p, []).append(item)
            expected = {paper.paper_id for paper in chunk}
            unknown = sorted(set(by_id) - expected)
            if unknown:
                raise ValueError("unknown paper ids: " + ", ".join(unknown))
            outcomes = {}
            for paper in chunk:
                matches = by_id.get(paper.paper_id, [])
                if len(matches) != 1:
                    raise ValueError(("missing" if not matches else "duplicate") + " paper output: " + paper.paper_id)
                outcomes[paper.paper_id] = _validate_item(matches[0], protocol=protocol, paper=paper, stage=stage, model=model, elapsed=elapsed / len(chunk))
            return outcomes, 1 + (_ > 0), elapsed_total
        except Exception as exc:
            last_error = exc
            elapsed_total += max(time.perf_counter() - started, float(getattr(exc, "elapsed_seconds", 0.0) or 0.0), 0.0001)
    error = LocalAIOutputError(f"Fast Binary {stage} batch failed for " + ", ".join(p.paper_id for p in chunk) + f" after retry: {last_error}", elapsed_seconds=elapsed_total)
    error.fast_call_count = 2
    raise error


def _execute_stage_batches(engine: FastStructuredEngine, protocol: ScreeningProtocolV2, papers: list[LocalV2Paper], *, stage: FastStage, model: str, timeout_seconds: float, batch_size: int, on_batch: Callable[[FastStage, int, int], None] | None) -> tuple[dict[str, FastStageOutcome], int, float, int]:
    outcomes: dict[str, FastStageOutcome] = {}
    calls = 0
    elapsed = 0.0
    completed_batches = 0
    for chunk in _split_for_prompt_limit(protocol, papers, stage=stage, requested_batch_size=batch_size):
        try:
            chunk_outcomes, chunk_calls, chunk_elapsed = _call_chunk(engine, protocol, chunk, stage=stage, model=model, timeout_seconds=timeout_seconds)
            outcomes.update(chunk_outcomes); calls += chunk_calls; elapsed += chunk_elapsed; completed_batches += 1
            if on_batch: on_batch(stage, completed_batches, len(chunk))
        except LocalAIOutputError as exc:
            calls += int(getattr(exc, "fast_call_count", 0) or 0)
            elapsed += float(getattr(exc, "elapsed_seconds", 0.0) or 0.0)
            if len(chunk) == 1:
                raise
            midpoint = len(chunk) // 2
            for smaller in (chunk[:midpoint], chunk[midpoint:]):
                split_outcomes, split_calls, split_elapsed, split_completed = _execute_stage_batches(engine, protocol, smaller, stage=stage, model=model, timeout_seconds=timeout_seconds, batch_size=len(smaller), on_batch=on_batch)
                outcomes.update(split_outcomes); calls += split_calls; elapsed += split_elapsed; completed_batches += split_completed
    return outcomes, calls, round(elapsed, 4), completed_batches


def _policy(decision: Literal["KEEP", "REJECT"], reason: str, criterion_ids: list[str]) -> PolicyResult:
    return PolicyResult(decision=decision, reason=reason, decisive_criterion_ids=criterion_ids)


def _result(protocol: ScreeningProtocolV2, paper: LocalV2Paper, primary: FastStageOutcome, reviewer: FastStageOutcome | None = None) -> FastPaperRunResult:
    if primary.decision == "K":
        final, route, confidence_reason = "KEEP", "FAST_BINARY_PRIMARY_KEEP", "Primary binary screener assigned KEEP."
    elif reviewer is None:
        raise ValueError("primary REJECT requires blind reviewer")
    elif reviewer.decision == "K":
        final, route, confidence_reason = "KEEP", "FAST_BINARY_REVIEWER_KEEP", "Blind binary reviewer overturned the primary rejection with KEEP."
    else:
        final, route, confidence_reason = "REJECT", "FAST_BINARY_REVIEWER_REJECT", "Blind binary reviewer independently assigned REJECT using validated evidence."
    status = _paper_text_status(paper)
    return FastPaperRunResult(protocol_id=protocol.protocol_id, paper=paper, text_status=status, route=route, primary=primary, reviewer=reviewer, final_policy=_policy(final, confidence_reason, reviewer.criterion_ids if reviewer and final == "REJECT" else primary.criterion_ids), model_call_count=1 + int(reviewer is not None), elapsed_seconds=round(primary.elapsed_seconds + (reviewer.elapsed_seconds if reviewer else 0), 4), warnings=_text_warnings(status))


def is_fast_result_resumable(result: FastPaperRunResult) -> bool:
    return result.status == "COMPLETED" and result.final_policy.decision in {"KEEP", "REJECT"} and not result.safe_fallback


def _checkpoint_payload(*, batch_id: str, protocol: ScreeningProtocolV2, plan: FastModelPlan, papers: list[LocalV2Paper], primary_batch_size: int, review_batch_size: int, results_by_position: Mapping[int, FastPaperRunResult]) -> dict[str, Any]:
    return {"checkpoint_version": FAST_CHECKPOINT_VERSION, "batch_runner_version": FAST_RUNNER_VERSION, "batch_id": batch_id, "protocol_id": protocol.protocol_id, "model_plan": plan.model_dump(mode="json"), "primary_batch_size": primary_batch_size, "review_batch_size": review_batch_size, "paper_count": len(papers), "entries": [{"position": position, "paper_fingerprint": _paper_fingerprint(papers[position]), "result": result.model_dump(mode="json")} for position, result in sorted(results_by_position.items()) if is_fast_result_resumable(result)]}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(_canonical_json(payload) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_checkpoint(path: Path, *, batch_id: str, protocol: ScreeningProtocolV2, plan: FastModelPlan, papers: list[LocalV2Paper], primary_batch_size: int, review_batch_size: int) -> tuple[dict[int, FastPaperRunResult], Literal["NOT_FOUND", "LOADED", "IGNORED_INVALID", "IGNORED_MISMATCH"], list[str]]:
    if not path.exists(): return {}, "NOT_FOUND", []
    try: payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc: return {}, "IGNORED_INVALID", [f"Fast Binary checkpoint was invalid and ignored: {exc}"]
    identity = (payload.get("checkpoint_version") == FAST_CHECKPOINT_VERSION and payload.get("batch_runner_version") == FAST_RUNNER_VERSION and payload.get("batch_id") == batch_id and payload.get("protocol_id") == protocol.protocol_id and payload.get("model_plan") == plan.model_dump(mode="json") and payload.get("primary_batch_size") == primary_batch_size and payload.get("review_batch_size") == review_batch_size and payload.get("paper_count") == len(papers))
    if not identity: return {}, "IGNORED_MISMATCH", ["Fast Binary checkpoint identity did not match this exact run."]
    try:
        results = {}
        for entry in payload.get("entries", []):
            position = int(entry["position"])
            if not 0 <= position < len(papers) or entry["paper_fingerprint"] != _paper_fingerprint(papers[position]): raise ValueError("checkpoint paper identity mismatch")
            result = FastPaperRunResult.model_validate(entry["result"])
            if result.paper != papers[position] or not is_fast_result_resumable(result): raise ValueError("checkpoint contains non-binary result")
            results[position] = result
        return results, "LOADED", []
    except Exception as exc:
        return {}, "IGNORED_INVALID", [f"Fast Binary checkpoint entries were invalid and ignored: {exc}"]


def run_compiled_local_v2_fast_batch(engine: FastStructuredEngine, protocol: ScreeningProtocolV2, *, papers: Iterable[LocalV2Paper | Mapping[str, Any]], model_plan: FastModelPlan | None = None, primary_batch_size: int = 4, review_batch_size: int = 4, checkpoint_path: str | os.PathLike[str] | None = None, resume: bool = True, on_result: Callable[[int, FastPaperRunResult, bool], None] | None = None, on_stage_batch: Callable[[FastStage, int, int], None] | None = None) -> FastBatchRunResult:
    _validate_protocol_identity(protocol)
    normalized = _normalize_papers(papers)
    if primary_batch_size < 1 or review_batch_size < 1: raise ValueError("fast batch sizes must be positive")
    plan = model_plan or FastModelPlan()
    batch_id = _build_batch_id(protocol, normalized, plan=plan, primary_batch_size=primary_batch_size, review_batch_size=review_batch_size)
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint is None: results_by_position, disposition, warnings = {}, "NOT_REQUESTED", []
    elif not resume: results_by_position, disposition, warnings = {}, "DISABLED", []
    else: results_by_position, disposition, warnings = _load_checkpoint(checkpoint, batch_id=batch_id, protocol=protocol, plan=plan, papers=normalized, primary_batch_size=primary_batch_size, review_batch_size=review_batch_size)
    resumed = set(results_by_position)
    for position in sorted(resumed):
        if on_result: on_result(position, results_by_position[position], True)
    fresh = [paper for position, paper in enumerate(normalized) if position not in resumed]
    positions = {paper.paper_id: position for position, paper in enumerate(normalized)}
    primary, primary_calls, primary_elapsed, primary_batches = _execute_stage_batches(engine, protocol, fresh, stage="primary", model=plan.primary_model, timeout_seconds=plan.primary_timeout_seconds, batch_size=primary_batch_size, on_batch=on_stage_batch)
    candidates = []
    for paper in fresh:
        if primary[paper.paper_id].decision == "R": candidates.append(paper)
        else:
            result = _result(protocol, paper, primary[paper.paper_id]); results_by_position[positions[paper.paper_id]] = result
            if on_result: on_result(positions[paper.paper_id], result, False)
    review, review_calls, review_elapsed, review_batches = _execute_stage_batches(engine, protocol, candidates, stage="reviewer", model=plan.review_model, timeout_seconds=plan.review_timeout_seconds, batch_size=review_batch_size, on_batch=on_stage_batch)
    for paper in candidates:
        result = _result(protocol, paper, primary[paper.paper_id], review[paper.paper_id]); results_by_position[positions[paper.paper_id]] = result
        if on_result: on_result(positions[paper.paper_id], result, False)
    if checkpoint:
        _atomic_write_json(checkpoint, _checkpoint_payload(batch_id=batch_id, protocol=protocol, plan=plan, papers=normalized, primary_batch_size=primary_batch_size, review_batch_size=review_batch_size, results_by_position=results_by_position))
    ordered = [results_by_position[position] for position in range(len(normalized))]
    decisions, routes = Counter(result.final_policy.decision for result in ordered), Counter(result.route for result in ordered)
    primary_direct_keeps = sum(result.route == "FAST_BINARY_PRIMARY_KEEP" for result in ordered)
    reviewer_keeps = sum(result.route == "FAST_BINARY_REVIEWER_KEEP" for result in ordered)
    reviewer_rejects = sum(result.route == "FAST_BINARY_REVIEWER_REJECT" for result in ordered)
    metrics = FastBatchMetrics(total_papers=len(ordered), keep_count=decisions["KEEP"], reject_count=decisions["REJECT"], resumed_count=len(resumed), fresh_count=len(ordered)-len(resumed), model_call_count=primary_calls + review_calls, primary_batch_count=primary_batches, review_batch_count=review_batches, review_used_count=len(candidates), primary_papers_assessed=len(fresh), primary_direct_keep_count=primary_direct_keeps, reviewer_candidate_count=reviewer_keeps + reviewer_rejects, reviewer_papers_assessed=len(candidates), reviewer_keep_count=reviewer_keeps, reviewer_reject_count=reviewer_rejects, retry_count=max(0, primary_calls + review_calls - primary_batches - review_batches), model_elapsed_seconds=round(primary_elapsed + review_elapsed, 4), checkpoint_write_count=int(checkpoint is not None), route_counts=dict(routes))
    return FastBatchRunResult(batch_id=batch_id, protocol_id=protocol.protocol_id, model_plan=plan, results=ordered, metrics=metrics, checkpoint_path=str(checkpoint) if checkpoint else None, checkpoint_disposition=disposition, checkpoint_warnings=warnings, resumed_positions=sorted(resumed))


def local_v2_fast_result_to_public_result(result: FastPaperRunResult, *, resource_profile: str, resumed: bool) -> dict[str, Any]:
    selected = result.reviewer if result.reviewer and result.final_policy.decision == "REJECT" else result.primary
    evidence = [citation.model_dump(mode="json") for assessment in selected.assessments for citation in assessment.evidence]
    confidence = 0.80 if result.route == "FAST_BINARY_PRIMARY_KEEP" else 0.85 if result.route == "FAST_BINARY_REVIEWER_KEEP" else 0.95
    return {"decision": result.final_policy.decision, "reason": result.final_policy.reason, "confidence": confidence, "protocol_id": result.protocol_id, "evidence": evidence, "criteria": [assessment.model_dump(mode="json") for assessment in selected.assessments], "uncertainty": list(result.warnings), "escalated": result.reviewer is not None, "validation_status": "validated", "validation_errors": [], "schema_version": FAST_RUNNER_VERSION, "model_tier": "local_v2_fast", "resource_profile": resource_profile or "balanced", "model": result.primary.model, "prompt_version": FAST_RUNNER_VERSION, "processing_seconds": 0.0 if resumed else result.elapsed_seconds, "original_processing_seconds": result.elapsed_seconds, "cache_hit": resumed, "runtime_downgrades": [], "layer_trace": [{"name": "local_v2_fast_binary", "route": result.route, "model_calls": result.model_call_count, "safe_fallback": False}], "layer_metrics": [{"processing_seconds": result.elapsed_seconds, "model_calls": result.model_call_count}], "decision_risk": "HIGH" if result.final_policy.decision == "REJECT" else "LOW", "triage_basis": result.route, "rq_frame_id": result.protocol_id, "rq_frame_version": FAST_RUNNER_VERSION, "rq_frame_source": "compiled_local_v2_protocol", "rq_frame_status": "validated", "rq_frame_validation_failures": [], "rq_group_coverage": {}, "local_profile": "local-v2-fast-binary", "protocol_model": "", "deep_model": result.reviewer.model if result.reviewer else "", "edge_model": ""}
