from __future__ import annotations

import json
import os
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, Field


LOCAL_MODEL = os.getenv("LOCAL_AI_MODEL", "qwen3.5:4b")
ARCHITECTURE_VERSION = "local-ai-simple-v1"
PROMPT_VERSION = "local-ai-screening-v2"
MAX_ATTEMPTS = 3


class LocalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["KEEP", "REJECT", "MAYBE"]
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=500)
    evidence_quote: str = Field(default="", max_length=600)


def screening_prompt(
    *, question: str, context: str, inclusion: str, exclusion: str,
    title: str, abstract: str,
) -> str:
    return f"""Screen this paper for a systematic review using only its title and abstract.

Research question: {question}
Context: {context or '(none supplied)'}
Inclusion criteria: {inclusion or '(use the research question)'}
Exclusion criteria: {exclusion or '(none supplied)'}

Title: {title}
Abstract: {abstract}

Return KEEP when the paper clearly fits, REJECT when it clearly does not fit or
meets an exclusion criterion, and MAYBE when the evidence is insufficient or
ambiguous. Apply explicit exclusions directly: for example, if reviews are
excluded and the title or abstract calls the paper a review, return REJECT.
For KEEP or REJECT, copy one exact continuous quote from the title or abstract.
Prefer copying the complete title when it directly supports the decision; never
add quotation marks inside evidence_quote. Give a brief reason. Return JSON only."""


def _ollama_generate(prompt: str, *, model: str = LOCAL_MODEL, timeout: float = 120) -> dict[str, Any]:
    response = requests.post(
        os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/") + "/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "format": LocalDecision.model_json_schema(),
            "keep_alive": "10m",
            "options": {
                "temperature": 0.1,
                "num_ctx": 8192,
                "num_predict": 256,
                "seed": 17,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "content": payload.get("message", {}).get("content", ""),
        "eval_count": int(payload.get("eval_count") or 0),
        "eval_duration": int(payload.get("eval_duration") or 0),
    }


def _parse_content(raw: str) -> LocalDecision:
    text = str(raw or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return LocalDecision.model_validate(json.loads(text))


def _validate_evidence(decision: LocalDecision, title: str, abstract: str) -> None:
    quote = decision.evidence_quote.strip()
    # Models sometimes include typographic delimiters as part of the JSON value.
    # Removing only balanced quote wrappers does not alter the claimed source span.
    wrappers = (("\\\"", "\\\""), ('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"))
    changed = True
    while changed and len(quote) >= 2:
        changed = False
        for left, right in wrappers:
            if quote.startswith(left) and quote.endswith(right):
                quote = quote[len(left):len(quote) - len(right)].strip()
                changed = True
                break
    decision.evidence_quote = quote
    if decision.decision in {"KEEP", "REJECT"}:
        if not quote:
            raise ValueError("KEEP and REJECT require an evidence quote")
        if quote not in title and quote not in abstract:
            raise ValueError("evidence quote is not an exact title/abstract span")
    elif quote and quote not in title and quote not in abstract:
        raise ValueError("evidence quote is not an exact title/abstract span")


def assess_paper(
    *, question: str, context: str, inclusion: str, exclusion: str,
    title: str, abstract: str, model: str = LOCAL_MODEL,
    generate: Callable[..., dict[str, Any]] = _ollama_generate,
    retry_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    prompt = screening_prompt(
        question=question, context=context, inclusion=inclusion,
        exclusion=exclusion, title=title, abstract=abstract,
    )
    errors: list[str] = []
    started = time.monotonic()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            generated = generate(prompt, model=model)
            parsed = _parse_content(generated["content"])
            _validate_evidence(parsed, title, abstract)
            return {
                **parsed.model_dump(),
                "validation_status": "validated",
                "validation_errors": [],
                "failure_class": "",
                "attempts": attempt,
                "processing_seconds": round(time.monotonic() - started, 3),
                "eval_count": int(generated.get("eval_count") or 0),
            }
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt < MAX_ATTEMPTS:
                if retry_callback:
                    retry_callback()
                evidence_instruction = ""
                if "evidence quote" in str(exc).lower():
                    evidence_instruction = (
                        " To avoid another transcription error, set evidence_quote to this exact"
                        f" complete title: {json.dumps(title, ensure_ascii=False)}."
                    )
                prompt += (
                    "\n\nYour previous response was unusable: " + errors[-1]
                    + ". Return one valid JSON object."
                    + evidence_instruction
                )
    return {
        "decision": "MAYBE",
        "confidence": 0.0,
        "reason": "Local AI did not return a technically valid assessment; manual review is required.",
        "evidence_quote": "",
        "validation_status": "technical_failure",
        "validation_errors": errors,
        "failure_class": "local_inference_failure",
        "attempts": MAX_ATTEMPTS,
        "processing_seconds": round(time.monotonic() - started, 3),
        "eval_count": 0,
    }


def _protocol_id(question: str, context: str, inclusion: str, exclusion: str, model: str) -> str:
    payload = json.dumps({
        "architecture": ARCHITECTURE_VERSION,
        "prompt": PROMPT_VERSION,
        "model": model,
        "question": question,
        "context": context,
        "inclusion": inclusion,
        "exclusion": exclusion,
    }, sort_keys=True, ensure_ascii=False)
    return "local-ai-" + sha256(payload.encode("utf-8")).hexdigest()


def _checkpoint_identity(input_fingerprint: str, protocol_id: str, source_ids: list[str]) -> str:
    payload = json.dumps({
        "input": input_fingerprint,
        "protocol": protocol_id,
        "source_ids": source_ids,
    }, sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _clean_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _reusable(row: dict[str, Any], protocol_id: str) -> bool:
    return (
        str(row.get("Protocol_ID")) == protocol_id
        and str(row.get("Validation_Status")) == "validated"
        and str(row.get("Decision")) in {"KEEP", "REJECT", "MAYBE"}
        and not str(row.get("Failure_Class") or "").strip()
    )


def _result_row(
    source: dict[str, Any], *, source_index: Any, title: str, abstract: str,
    result: dict[str, Any], question: str, context: str, inclusion: str,
    exclusion: str, protocol_id: str, model: str, origin: str,
) -> dict[str, Any]:
    decision = str(result["decision"])
    evidence = str(result.get("evidence_quote") or "")
    assessment = {
        "decision": decision,
        "confidence": float(result.get("confidence") or 0),
        "reason": str(result.get("reason") or ""),
        "evidence_quote": evidence,
    }
    row = dict(source)
    row.update({
        "Title": title,
        "Abstract": abstract,
        "Decision": decision,
        "Original_Decision": decision,
        "Decision_Source": "tool_assisted_screening",
        "Exclusion_Reason": "",
        "Confidence": assessment["confidence"],
        "Reason": assessment["reason"],
        "Evidence_Quote": evidence,
        "Evidence_JSON": json.dumps(
            ([{"source": "title" if evidence in title else "abstract", "quote": evidence}] if evidence else []),
            ensure_ascii=False,
        ),
        "Validation_Status": result.get("validation_status", "technical_failure"),
        "Validation_Errors": json.dumps(result.get("validation_errors", []), ensure_ascii=False),
        "Failure_Class": result.get("failure_class", ""),
        "Protocol_ID": protocol_id,
        "Review_Protocol_ID": protocol_id,
        "Research_Question": question,
        "Research_Context": context,
        "Inclusion_Criteria": inclusion,
        "Exclusion_Criteria": exclusion,
        "Architecture_Version": ARCHITECTURE_VERSION,
        "Prompt_Version": PROMPT_VERSION,
        "Model": model,
        "Model_Tier": "single_model",
        "Resource_Profile": "local",
        "Primary_Decision": decision,
        "Primary_Confidence": assessment["confidence"],
        "Primary_Reason": assessment["reason"],
        "Primary_Evidence_Quote": evidence,
        "Primary_Assessment_JSON": json.dumps(assessment, ensure_ascii=False),
        "Verifier_Decision": "",
        "Verifier_Assessment_JSON": "",
        "Agreement_Status": "single_model",
        "Route_Used": "local_ai",
        "Execution_Origin": origin,
        "Source_Row_Index": source_index,
        "Processing_Seconds": result.get("processing_seconds", 0),
        "Attempt_Count": result.get("attempts", 1),
        "Eval_Token_Count": result.get("eval_count", 0),
    })
    return row


def screen_csv_with_local_ai(
    *, frame: pd.DataFrame, valid: pd.DataFrame, title_col: str, abstract_col: str,
    research_question: str, research_context: str, inclusion_criteria: str,
    exclusion_criteria: str, output_path: str, job_id: str,
    input_fingerprint: str, resume: bool, progress, screening_session,
    model: str = LOCAL_MODEL,
    generate: Callable[..., dict[str, Any]] = _ollama_generate,
) -> dict[str, Any]:
    started = time.monotonic()
    protocol_id = _protocol_id(
        research_question, research_context, inclusion_criteria, exclusion_criteria, model,
    )
    source_ids = [str(index) for index in valid.index]
    output = Path(output_path)
    output_root = output.parent.parent if output.parent.name == "runs" else output.parent
    checkpoint = output_root / "cache" / "local_ai" / (
        _checkpoint_identity(input_fingerprint, protocol_id, source_ids) + ".csv"
    )
    resumed: dict[str, dict[str, Any]] = {}
    if resume and checkpoint.is_file():
        try:
            restored = pd.read_csv(checkpoint, dtype=str, keep_default_na=False, encoding="utf-8-sig")
            resumed = {
                str(row.get("Source_Row_Index")): row
                for row in restored.to_dict(orient="records")
                if _reusable(row, protocol_id) and str(row.get("Source_Row_Index")) in source_ids
            }
        except (OSError, ValueError):
            resumed = {}

    rows_by_id = dict(resumed)
    progress.set_resumed_count(job_id, len(resumed))
    progress.begin_batches(job_id, "local_screening", len(valid), len(valid), 1)
    counts = screening_session.counts(list(rows_by_id.values()))
    progress.update_counts(job_id, len(rows_by_id), counts["keep"], counts["maybe"], counts["reject"])

    retries = 0
    failures = 0
    for position, (source_index, source) in enumerate(valid.iterrows(), start=1):
        source_id = str(source_index)
        if source_id in rows_by_id:
            progress.update_batch(job_id, position, len(rows_by_id))
            continue
        title = _clean_cell(source.get(title_col, ""))
        abstract = _clean_cell(source.get(abstract_col, ""))
        if not title or not abstract:
            result = {
                "decision": "MAYBE", "confidence": 0.0,
                "reason": "The title or abstract is missing; manual review is required.",
                "evidence_quote": "", "validation_status": "technical_failure",
                "validation_errors": ["missing title or abstract"],
                "failure_class": "missing_title_or_abstract", "attempts": 0,
                "processing_seconds": 0, "eval_count": 0,
            }
            failures += 1
            origin = "missing_input"
        else:
            def record_retry() -> None:
                nonlocal retries
                retries += 1
                progress.record_retry(job_id)
            result = assess_paper(
                question=research_question, context=research_context,
                inclusion=inclusion_criteria, exclusion=exclusion_criteria,
                title=title, abstract=abstract, model=model,
                generate=generate,
                retry_callback=record_retry,
            )
            if result.get("failure_class"):
                failures += 1
            origin = "fresh"
        rows_by_id[source_id] = _result_row(
            source.to_dict(), source_index=source_index, title=title, abstract=abstract,
            result=result, question=research_question, context=research_context,
            inclusion=inclusion_criteria, exclusion=exclusion_criteria,
            protocol_id=protocol_id, model=model, origin=origin,
        )
        ordered = [rows_by_id[str(index)] for index in valid.index if str(index) in rows_by_id]
        _atomic_csv(checkpoint, ordered)
        counts = screening_session.counts(ordered)
        progress.update_batch(job_id, position, len(ordered))
        progress.update_counts(job_id, len(ordered), counts["keep"], counts["maybe"], counts["reject"])

    ordered = [rows_by_id[str(index)] for index in valid.index]
    _atomic_csv(checkpoint, ordered)
    _atomic_csv(output, ordered)
    screening_session.set_results(
        ordered, job_id=job_id, output_path=output_path,
        architecture_version=ARCHITECTURE_VERSION,
    )
    counts = screening_session.counts(ordered)
    progress.update_counts(job_id, len(ordered), counts["keep"], counts["maybe"], counts["reject"])
    progress.finish(job_id)
    runtime = round(time.monotonic() - started, 3)
    return {
        **counts,
        "total_papers": len(ordered),
        "output_file": output_path,
        "architecture_version": ARCHITECTURE_VERSION,
        "screening_engine": "local",
        "model": model,
        "protocol_id": protocol_id,
        "runtime_seconds": runtime,
        "retry_count": retries,
        "safe_fallback_count": failures,
        "resumed_count": len(resumed),
        "fresh_primary_count": len(ordered) - len(resumed),
        "input_total_rows": len(frame),
        "screened_total_rows": len(ordered),
    }
