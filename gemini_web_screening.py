from __future__ import annotations

import json
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
from pydantic import ConfigDict, Field

from external_ai.engine import parse_structured_model_output
from external_ai.prompts import protocol_prompt
from gemini_web_automation import GeminiWebAutomation, GeminiWebConfig
from gemini_web_prompt import (
    ScreeningPaper, build_structured_batch_prompt, build_structured_critic_prompt,
)
from local_ai.contracts import (
    SCHEMA_VERSION, PaperAssessment, ReviewProtocol, StrictModel, ValidationReport, safe_maybe,
)
from local_ai.evidence import evidence_lookup
from local_ai.engine import LocalAIOutputError
from local_ai.validator import validate_assessment


GEMINI_WEB_ENGINE = "gemini_web"
GEMINI_WEB_VERSION = "gemini-web-batched-v1"
GEMINI_WEB_BATCH_SIZE = 5


class WebPaperAssessment(PaperAssessment):
    paper_id: str = Field(min_length=1, max_length=100)
    certainty: Literal["HIGH", "BORDERLINE", "LOW"]


class WebAssessmentBatch(StrictModel):
    model_config = ConfigDict(extra="forbid")
    items: list[WebPaperAssessment] = Field(min_length=1, max_length=GEMINI_WEB_BATCH_SIZE)


def _contract_hash(input_fingerprint: str, question: str, context: str, inclusion: str, exclusion: str) -> str:
    payload = {
        "input_fingerprint": input_fingerprint,
        "research_question": question,
        "research_context": context,
        "inclusion_criteria": inclusion,
        "exclusion_criteria": exclusion,
        "version": GEMINI_WEB_VERSION,
        "batch_size": GEMINI_WEB_BATCH_SIZE,
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _protocol_hash(question: str, context: str, inclusion: str, exclusion: str) -> str:
    return sha256(json.dumps({
        "question": question, "context": context, "inclusion": inclusion,
        "exclusion": exclusion, "version": GEMINI_WEB_VERSION,
    }, sort_keys=True).encode("utf-8")).hexdigest()


def _validate_protocol(protocol: ReviewProtocol, inclusion: str, exclusion: str) -> None:
    if any(c.source == "research_question" and c.kind == "exclusion" for c in protocol.criteria):
        raise ValueError("RQ scope must use positive inclusion criteria, not inferred exclusions")
    if inclusion.strip() and not any(c.source == "user" and c.kind == "inclusion" for c in protocol.criteria):
        raise ValueError("explicit user inclusion criteria were not preserved")
    if exclusion.strip() and not any(c.source == "user" and c.kind == "exclusion" for c in protocol.criteria):
        raise ValueError("explicit user exclusion criteria were not preserved")


def _compile_protocol(browser, question: str, context: str, inclusion: str, exclusion: str) -> ReviewProtocol:
    base = protocol_prompt(question, inclusion, exclusion, context)
    last_error: Exception | None = None
    for attempt in range(2):
        prompt = base if attempt == 0 else (
            base + "\n\nCORRECTION: The previous protocol was invalid. Return a corrected JSON protocol only. "
            + str(last_error)
        )
        try:
            raw = browser.submit_prompt_and_get_response(prompt)
            value = parse_structured_model_output(raw, ReviewProtocol)
            protocol = ReviewProtocol.model_validate(value).model_copy(update={
                "research_question": question,
                "research_context": context,
                "prompt_version": GEMINI_WEB_VERSION,
                "model": "gemini-web",
            }).with_identity()
            _validate_protocol(protocol, inclusion, exclusion)
            return protocol
        except (LocalAIOutputError, ValueError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if (
                isinstance(exc, TimeoutError)
                or (isinstance(exc, RuntimeError) and not isinstance(exc, LocalAIOutputError))
            ) and attempt == 0:
                browser.recover_job_chat()
    raise RuntimeError(f"Gemini Web could not create a valid screening protocol: {last_error}")


def _validate_batch(value: dict, expected_ids: set[str]) -> dict[str, WebPaperAssessment]:
    parsed = WebAssessmentBatch.model_validate(value)
    ids = [item.paper_id for item in parsed.items]
    if len(ids) != len(set(ids)):
        raise LocalAIOutputError("Gemini Web returned duplicate paper IDs in a batch")
    if set(ids) != expected_ids:
        raise LocalAIOutputError(
            f"Gemini Web batch IDs did not match; expected {sorted(expected_ids)}, received {sorted(ids)}"
        )
    return {item.paper_id: item for item in parsed.items}


def _fallback_item(paper: ScreeningPaper, protocol: ReviewProtocol, reason: str) -> WebPaperAssessment:
    fallback = safe_maybe(reason)
    criteria = [{
        "criterion_id": criterion.id, "verdict": "UNCLEAR",
        "rationale": "Gemini Web output could not be validated.", "evidence": [],
    } for criterion in protocol.criteria]
    return WebPaperAssessment(
        **fallback.model_dump(exclude={"criteria"}), paper_id=paper.paper_id,
        certainty="LOW", criteria=criteria,
    )


def _execute_batch(
    browser, protocol: ReviewProtocol, papers: list[ScreeningPaper], *, critic: bool,
    prior: dict[str, dict] | None, record_retry: Callable[[], None],
) -> dict[str, WebPaperAssessment]:
    schema = WebAssessmentBatch.model_json_schema()
    prompt = (
        build_structured_critic_prompt(
            protocol=protocol.model_dump(mode="json"), papers=papers,
            prior=prior or {}, schema=schema,
        ) if critic else build_structured_batch_prompt(
            protocol=protocol.model_dump(mode="json"), papers=papers, schema=schema,
        )
    )
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            request = prompt if attempt == 0 else (
                prompt + "\n\nREPAIR: Your previous response was malformed or used incorrect IDs. "
                "Return the complete corrected batch JSON only."
            )
            raw = browser.submit_prompt_and_get_response(request)
            return _validate_batch(
                parse_structured_model_output(raw, WebAssessmentBatch),
                {paper.paper_id for paper in papers},
            )
        except (LocalAIOutputError, ValueError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            record_retry()
            if (
                isinstance(exc, TimeoutError)
                or (isinstance(exc, RuntimeError) and not isinstance(exc, LocalAIOutputError))
            ) and attempt == 0:
                browser.recover_job_chat()

    if len(papers) > 1:
        midpoint = max(1, len(papers) // 2)
        combined = _execute_batch(
            browser, protocol, papers[:midpoint], critic=critic, prior=prior,
            record_retry=record_retry,
        )
        combined.update(_execute_batch(
            browser, protocol, papers[midpoint:], critic=critic, prior=prior,
            record_retry=record_retry,
        ))
        return combined
    paper = papers[0]
    return {paper.paper_id: _fallback_item(
        paper, protocol, f"Gemini Web could not return a valid assessment: {last_error}"
    )}


def _assessment(item: WebPaperAssessment) -> PaperAssessment:
    return PaperAssessment.model_validate(item.model_dump(exclude={"paper_id", "certainty"}))


def _public_result(
    item: WebPaperAssessment, protocol: ReviewProtocol, paper: ScreeningPaper, *,
    stage: str, elapsed: float, prior_trace: list[dict] | None = None,
) -> dict[str, Any]:
    assessment = _assessment(item)
    validation = validate_assessment(assessment, protocol, paper.title, paper.abstract)
    units = evidence_lookup(paper.title, paper.abstract)
    criteria, evidence = [], []
    for criterion in assessment.criteria:
        spans = []
        for reference in criterion.evidence:
            unit = units.get(reference.evidence_id)
            if unit is None or unit["source"] != reference.source:
                continue
            span = {"source": unit["source"], "evidence_id": unit["evidence_id"], "quote": unit["text"]}
            spans.append(span)
            evidence.append({"criterion_id": criterion.criterion_id, **span})
        criteria.append({
            "criterion_id": criterion.criterion_id, "verdict": criterion.verdict,
            "rationale": criterion.rationale, "evidence": spans,
        })
    risk = "LOW" if item.certainty == "HIGH" else ("BORDERLINE" if item.certainty == "BORDERLINE" else "HIGH")
    trace = list(prior_trace or []) + [{
        "name": stage, "decision": assessment.decision, "certainty": item.certainty,
        "validation_status": "validated" if validation.valid else "unresolved",
        "validation_errors": validation.errors,
    }]
    return {
        "schema_version": SCHEMA_VERSION, "decision": assessment.decision,
        "reason": assessment.reason, "confidence": {"HIGH": .92, "BORDERLINE": .68, "LOW": .4}[item.certainty],
        "protocol_id": protocol.protocol_id, "criteria": criteria, "evidence": evidence,
        "summary": assessment.summary, "uncertainty": assessment.uncertainty,
        "missing_information": assessment.missing_information,
        "contradictions": assessment.contradictions,
        "validation_status": "validated" if validation.valid else "unresolved",
        "validation_errors": validation.errors, "validation_warnings": validation.warnings,
        "escalated": stage == "gemini_web_critic", "model": "gemini-web",
        "model_tier": "gemini_web_batched", "resource_profile": "web",
        "prompt_version": GEMINI_WEB_VERSION, "processing_seconds": round(elapsed, 4),
        "original_processing_seconds": round(elapsed, 4), "cache_hit": False,
        "runtime_downgrades": [], "layer_trace": trace, "layer_metrics": [],
        "decision_risk": risk, "triage_basis": "gemini_web_structured_batch",
    }


def _needs_critic(result: dict[str, Any]) -> bool:
    return (
        result["decision"] == "MAYBE"
        or result["decision_risk"] != "LOW"
        or result["validation_status"] != "validated"
        or bool(result.get("contradictions"))
    )


def _row(source: dict, source_index: Any, paper: ScreeningPaper, result: dict[str, Any]) -> dict[str, Any]:
    row = dict(source)
    row.update({
        "Title": paper.title, "Abstract": paper.abstract, "Decision": result["decision"],
        "Reason": result["reason"], "Confidence": result["confidence"],
        "Protocol_ID": result["protocol_id"], "Evidence_JSON": json.dumps(result["evidence"], ensure_ascii=False),
        "Criteria_JSON": json.dumps(result["criteria"], ensure_ascii=False),
        "Uncertainty_JSON": json.dumps(result["uncertainty"], ensure_ascii=False),
        "Contradictions_JSON": json.dumps(result.get("contradictions", []), ensure_ascii=False),
        "Escalated": result["escalated"], "Validation_Status": result["validation_status"],
        "Validation_Errors": json.dumps(result["validation_errors"], ensure_ascii=False),
        "Schema_Version": result["schema_version"], "Model_Tier": result["model_tier"],
        "Resource_Profile": result["resource_profile"], "Model": result["model"],
        "Prompt_Version": result["prompt_version"], "Processing_Seconds": result["processing_seconds"],
        "Original_Processing_Seconds": result["original_processing_seconds"], "Cache_Hit": result["cache_hit"],
        "Runtime_Downgrades": "[]", "Layer_Trace_JSON": json.dumps(result["layer_trace"], ensure_ascii=False),
        "Layer_Metrics_JSON": "[]", "Decision_Risk": result["decision_risk"],
        "Triage_Basis": result["triage_basis"], "Source_Row_Index": source_index,
    })
    return row


def _resume_rows(path: Path, protocol_id: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
        required = {"Source_Row_Index", "Protocol_ID", "Prompt_Version", "Layer_Trace_JSON"}
        if not required.issubset(frame.columns):
            return {}
        frame = frame[
            (frame["Protocol_ID"].astype(str) == protocol_id)
            & (frame["Prompt_Version"].astype(str) == GEMINI_WEB_VERSION)
        ]
        rows = {}
        for _, source_row in frame.iterrows():
            row = source_row.to_dict()
            row["Cache_Hit"] = True
            row["Processing_Seconds"] = 0.0
            rows[str(row["Source_Row_Index"])] = row
        return rows
    except (OSError, ValueError, KeyError):
        return {}


def screen_csv_with_gemini_web(
    *, frame: pd.DataFrame, valid: pd.DataFrame, title_col: str, abstract_col: str,
    research_question: str, research_context: str, inclusion_criteria: str,
    exclusion_criteria: str, output_path: str, job_id: str, input_fingerprint: str,
    resume: bool, limit: int, progress, screening_session,
    browser_factory: Callable[[GeminiWebConfig], Any] = GeminiWebAutomation,
) -> dict[str, Any]:
    contract = _contract_hash(
        input_fingerprint, research_question, research_context, inclusion_criteria, exclusion_criteria
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_root = output.parent.parent / "cache" / "gemini_web"
    checkpoint = cache_root / "checkpoints" / f"{contract}.csv"
    protocol_file = cache_root / "protocols" / f"{_protocol_hash(research_question, research_context, inclusion_criteria, exclusion_criteria)}.json"
    protocol_file.parent.mkdir(parents=True, exist_ok=True)

    papers: dict[str, ScreeningPaper] = {}
    sources: dict[str, tuple[Any, dict]] = {}
    for source_index, source_row in valid.iterrows():
        key = str(source_index)
        title = "" if pd.isna(source_row[title_col]) else str(source_row[title_col])
        abstract = "" if pd.isna(source_row[abstract_col]) else str(source_row[abstract_col])
        papers[key] = ScreeningPaper(key, title, abstract)
        sources[key] = (source_index, source_row.to_dict())

    protocol = None
    try:
        protocol = ReviewProtocol.model_validate_json(protocol_file.read_text(encoding="utf-8"))
        _validate_protocol(protocol, inclusion_criteria, exclusion_criteria)
    except (OSError, ValueError):
        protocol = None

    browser_context = None
    browser = None
    if protocol is None:
        progress.begin_batches(job_id, "gemini_web_protocol", 1, 1, 1)
        browser_context = browser_factory(GeminiWebConfig())
        browser = browser_context.__enter__()
        protocol = _compile_protocol(
            browser, research_question, research_context, inclusion_criteria, exclusion_criteria
        )
        protocol_file.write_text(protocol.model_dump_json(indent=2), encoding="utf-8")
        progress.update_batch(job_id, 1, 1)

    rows = _resume_rows(checkpoint, protocol.protocol_id) if resume else {}
    resumed_count = len(rows)
    progress.set_resumed_count(job_id, resumed_count)
    if rows:
        cached_counts = screening_session.counts(list(rows.values()))
        progress.update_counts(
            job_id, len(rows), cached_counts["keep"], cached_counts["maybe"], cached_counts["reject"]
        )
    pending = [papers[key] for key in papers if key not in rows]

    try:
        if pending and browser is None:
            browser_context = browser_factory(GeminiWebConfig())
            browser = browser_context.__enter__()

        def retry():
            progress.record_retry(job_id)

        primary_batches = (len(pending) + GEMINI_WEB_BATCH_SIZE - 1) // GEMINI_WEB_BATCH_SIZE
        progress.begin_batches(
            job_id, "gemini_web_primary", len(pending), primary_batches, GEMINI_WEB_BATCH_SIZE
        )
        for batch_number in range(primary_batches):
            batch = pending[batch_number * GEMINI_WEB_BATCH_SIZE:(batch_number + 1) * GEMINI_WEB_BATCH_SIZE]
            started = time.perf_counter()
            assessed = _execute_batch(
                browser, protocol, batch, critic=False, prior=None, record_retry=retry
            )
            elapsed_each = (time.perf_counter() - started) / max(1, len(batch))
            for paper in batch:
                result = _public_result(
                    assessed[paper.paper_id], protocol, paper,
                    stage="gemini_web_primary", elapsed=elapsed_each,
                )
                source_index, source = sources[paper.paper_id]
                rows[paper.paper_id] = _row(source, source_index, paper, result)
            ordered = [rows[key] for key in papers if key in rows]
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(ordered).to_csv(checkpoint, index=False)
            pd.DataFrame(ordered).to_csv(output, index=False)
            counts = screening_session.counts(ordered)
            progress.update_batch(job_id, batch_number + 1, min(len(pending), (batch_number + 1) * 5))
            progress.update_counts(job_id, len(ordered), counts["keep"], counts["maybe"], counts["reject"])

        critic_keys = []
        for key in papers:
            row = rows.get(key)
            if not row:
                continue
            try:
                trace = json.loads(str(row.get("Layer_Trace_JSON") or "[]"))
            except json.JSONDecodeError:
                trace = []
            if trace and trace[-1].get("name") == "gemini_web_critic":
                continue
            try:
                contradictions = json.loads(str(row.get("Contradictions_JSON") or "[]"))
            except json.JSONDecodeError:
                contradictions = ["unparseable contradictions"]
            if (
                row.get("Decision") == "MAYBE" or row.get("Decision_Risk") != "LOW"
                or row.get("Validation_Status") != "validated" or bool(contradictions)
            ):
                critic_keys.append(key)

        critic_batches = (len(critic_keys) + 4) // 5
        if critic_keys and browser is None:
            browser_context = browser_factory(GeminiWebConfig())
            browser = browser_context.__enter__()
        progress.begin_batches(job_id, "gemini_web_critic", len(critic_keys), critic_batches, 5)
        for batch_number in range(critic_batches):
            keys = critic_keys[batch_number * 5:(batch_number + 1) * 5]
            batch = [papers[key] for key in keys]
            prior = {key: {
                "decision": rows[key]["Decision"], "reason": rows[key]["Reason"],
                "criteria": json.loads(str(rows[key].get("Criteria_JSON") or "[]")),
                "validation_errors": json.loads(str(rows[key].get("Validation_Errors") or "[]")),
                "contradictions": json.loads(str(rows[key].get("Contradictions_JSON") or "[]")),
            } for key in keys}
            started = time.perf_counter()
            assessed = _execute_batch(
                browser, protocol, batch, critic=True, prior=prior, record_retry=retry
            )
            elapsed_each = (time.perf_counter() - started) / max(1, len(batch))
            for paper in batch:
                primary_trace = json.loads(str(rows[paper.paper_id].get("Layer_Trace_JSON") or "[]"))
                candidate = _public_result(
                    assessed[paper.paper_id], protocol, paper,
                    stage="gemini_web_critic", elapsed=elapsed_each, prior_trace=primary_trace,
                )
                if candidate["validation_status"] != "validated":
                    assessed[paper.paper_id] = _fallback_item(
                        paper, protocol, "Gemini Web critic could not produce an evidence-safe definitive result."
                    )
                    candidate = _public_result(
                        assessed[paper.paper_id], protocol, paper,
                        stage="gemini_web_critic", elapsed=elapsed_each, prior_trace=primary_trace,
                    )
                source_index, source = sources[paper.paper_id]
                rows[paper.paper_id] = _row(source, source_index, paper, candidate)
            ordered = [rows[key] for key in papers]
            pd.DataFrame(ordered).to_csv(checkpoint, index=False)
            pd.DataFrame(ordered).to_csv(output, index=False)
            counts = screening_session.counts(ordered)
            progress.update_batch(job_id, batch_number + 1, min(len(critic_keys), (batch_number + 1) * 5))
            progress.update_counts(job_id, len(ordered), counts["keep"], counts["maybe"], counts["reject"])

        progress.begin_batches(job_id, "gemini_web_finalizing", len(papers), 1, len(papers) or 1)
        ordered = [rows[key] for key in papers]
        pd.DataFrame(ordered).to_csv(output, index=False)
        screening_session.set_results(
            ordered, job_id=job_id, output_path=output_path,
            architecture_version=GEMINI_WEB_VERSION,
        )
        final = screening_session.counts(ordered)
        progress.update_batch(job_id, 1, len(papers))
        progress.update_counts(job_id, len(ordered), final["keep"], final["maybe"], final["reject"])
        progress.finish(job_id)
        return {
            **final, "parse_error": 0, "output_file": output_path,
            "total_papers": len(ordered), "input_total_rows": len(frame),
            "screened_total_rows": len(ordered), "row_limit_applied": bool(limit),
            "row_limit_value": limit or "", "screening_engine": GEMINI_WEB_ENGINE,
            "architecture_version": GEMINI_WEB_VERSION, "resumed_count": resumed_count,
            "schema_version": SCHEMA_VERSION, "protocol_id": protocol.protocol_id,
            "model_tier": "gemini_web_batched", "resource_profile": "web",
            "fast_model": "gemini-web", "strong_model": "gemini-web",
            "escalated_count": sum(bool(row.get("Escalated")) for row in ordered),
        }
    finally:
        if browser_context is not None:
            browser_context.__exit__(None, None, None)
