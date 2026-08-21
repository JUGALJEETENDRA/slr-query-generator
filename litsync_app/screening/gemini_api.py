from __future__ import annotations

import json
import os
import random
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, Field


GEMINI_API_MODEL = os.getenv("GEMINI_API_MODEL", "gemini-2.5-flash-lite")
ARCHITECTURE_VERSION = "gemini-api-screening-v1"
PROMPT_VERSION = "gemini-api-screening-v1"
MAX_ATTEMPTS = 4


class GeminiApiDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["KEEP", "REJECT", "MAYBE"]
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=500)
    evidence_quote: str = Field(default="", max_length=600)


class GeminiApiError(RuntimeError):
    """A safe, key-free Gemini API error."""


class GeminiApiRetryableError(GeminiApiError):
    pass


class GeminiApiQuotaError(GeminiApiRetryableError):
    pass


class GeminiApiAuthenticationError(GeminiApiError):
    pass


class GeminiApiInterruptedError(GeminiApiError):
    pass


def screening_prompt(
    *, question: str, context: str, inclusion: str, exclusion: str,
    title: str, abstract: str,
) -> str:
    return f"""Screen this paper for a systematic review using only the supplied review information and the paper title and abstract.

Research question: {question}
Research context: {context or '(none supplied)'}
Inclusion criteria: {inclusion or '(use the research question)'}
Exclusion criteria: {exclusion or '(none supplied)'}

Title: {title}
Abstract: {abstract}

Return KEEP when the paper clearly satisfies the review scope, REJECT when it
clearly does not or meets an exclusion criterion, and MAYBE when the title and
abstract do not provide enough evidence for a safe decision. Apply every
explicit criterion. Do not infer missing study details. A topic mentioned only
as background, motivation, future work, or an incidental component is not
enough for KEEP. For KEEP or REJECT, copy one exact continuous quote from the
title or abstract into evidence_quote. MAYBE may use an empty evidence_quote.
Give a brief reason and return one JSON object only."""


def _safe_error_message(response: requests.Response) -> str:
    # Do not surface provider response text: it is not needed for recovery and
    # could theoretically echo request details. Status is enough for diagnostics.
    return f"Gemini API request failed with HTTP {response.status_code}."


def _request_once(
    prompt: str, *, api_key: str, model: str = GEMINI_API_MODEL,
    post: Callable[..., requests.Response] = requests.post,
    timeout: float = 60,
) -> dict[str, Any]:
    if not str(api_key or "").strip():
        raise GeminiApiAuthenticationError("A Gemini API key is required.")
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    try:
        response = post(
            endpoint,
            headers={
                "x-goog-api-key": str(api_key).strip(),
                "Content-Type": "application/json",
            },
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseJsonSchema": GeminiApiDecision.model_json_schema(),
                    "temperature": 0.1,
                    "maxOutputTokens": 400,
                },
            },
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise GeminiApiRetryableError(
            f"Gemini API connection failed: {type(exc).__name__}."
        ) from exc

    if response.status_code == 429:
        raise GeminiApiQuotaError(
            "Gemini API quota or rate limit was reached."
        )
    if response.status_code in {401, 403}:
        raise GeminiApiAuthenticationError(
            "Gemini API rejected the supplied key or its permissions."
        )
    if response.status_code in {408, 500, 502, 503, 504}:
        raise GeminiApiRetryableError(_safe_error_message(response))
    if not response.ok:
        raise GeminiApiError(_safe_error_message(response))

    try:
        payload = response.json()
        candidates = payload.get("candidates") or []
        parts = candidates[0]["content"]["parts"]
        content = "".join(str(part.get("text") or "") for part in parts)
        if not content.strip():
            raise ValueError("empty model response")
        usage = payload.get("usageMetadata") or {}
        return {
            "content": content,
            "total_tokens": int(usage.get("totalTokenCount") or 0),
        }
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise GeminiApiRetryableError(
            "Gemini API returned no usable model response."
        ) from exc


def _parse_content(raw: str) -> GeminiApiDecision:
    text = str(raw or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return GeminiApiDecision.model_validate(json.loads(text))


def _validate_evidence(decision: GeminiApiDecision, title: str, abstract: str) -> None:
    quote = decision.evidence_quote.strip()
    wrappers = (("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’"))
    changed = True
    while changed and len(quote) >= 2:
        changed = False
        for left, right in wrappers:
            if quote.startswith(left) and quote.endswith(right):
                quote = quote[len(left):len(quote) - len(right)].strip()
                changed = True
                break
    decision.evidence_quote = quote
    if decision.decision in {"KEEP", "REJECT"} and not quote:
        raise ValueError("KEEP and REJECT require an evidence quote")
    if quote and quote not in title and quote not in abstract:
        raise ValueError("evidence quote is not an exact title/abstract span")


def assess_paper(
    *, question: str, context: str, inclusion: str, exclusion: str,
    title: str, abstract: str, api_key: str, model: str = GEMINI_API_MODEL,
    generate: Callable[..., dict[str, Any]] = _request_once,
    retry_callback: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    prompt = screening_prompt(
        question=question, context=context, inclusion=inclusion,
        exclusion=exclusion, title=title, abstract=abstract,
    )
    errors: list[str] = []
    started = time.monotonic()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            generated = generate(prompt, api_key=api_key, model=model)
            parsed = _parse_content(generated["content"])
            evidence_repaired = False
            try:
                _validate_evidence(parsed, title, abstract)
            except ValueError as exc:
                if "evidence quote" not in str(exc).lower():
                    raise
                # This changes only an invalid transcription, never the model's
                # semantic decision.
                parsed.evidence_quote = title if parsed.decision in {"KEEP", "REJECT"} else ""
                _validate_evidence(parsed, title, abstract)
                evidence_repaired = True
            return {
                **parsed.model_dump(),
                "validation_status": "validated",
                "validation_errors": [],
                "retry_errors": list(errors),
                "evidence_repaired": evidence_repaired,
                "failure_class": "",
                "attempts": attempt,
                "processing_seconds": round(time.monotonic() - started, 3),
                "token_count": int(generated.get("total_tokens") or 0),
            }
        except GeminiApiAuthenticationError:
            raise
        except GeminiApiError as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            if not isinstance(exc, GeminiApiRetryableError) or attempt == MAX_ATTEMPTS:
                raise
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt == MAX_ATTEMPTS:
                return {
                    "decision": "MAYBE",
                    "confidence": 0.0,
                    "reason": "Gemini API did not return a technically valid assessment; manual review is required.",
                    "evidence_quote": "",
                    "validation_status": "technical_failure",
                    "validation_errors": errors,
                    "retry_errors": list(errors),
                    "evidence_repaired": False,
                    "failure_class": "invalid_model_response",
                    "attempts": MAX_ATTEMPTS,
                    "processing_seconds": round(time.monotonic() - started, 3),
                    "token_count": 0,
                }
        if retry_callback:
            retry_callback()
        sleep(min(8.0, (2 ** (attempt - 1)) + random.uniform(0, 0.25)))
        prompt += "\n\nReturn exactly one valid JSON object matching the requested fields."
    raise AssertionError("unreachable")


def _protocol_id(
    question: str, context: str, inclusion: str, exclusion: str, model: str,
) -> str:
    payload = json.dumps({
        "architecture": ARCHITECTURE_VERSION,
        "prompt": PROMPT_VERSION,
        "model": model,
        "question": question,
        "context": context,
        "inclusion": inclusion,
        "exclusion": exclusion,
    }, sort_keys=True, ensure_ascii=False)
    return "gemini-api-" + sha256(payload.encode("utf-8")).hexdigest()


def _checkpoint_identity(
    input_fingerprint: str, protocol_id: str, source_ids: list[str],
) -> str:
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
            ([{"source": "title" if evidence in title else "abstract", "quote": evidence}]
             if evidence else []),
            ensure_ascii=False,
        ),
        "Validation_Status": result.get("validation_status", "technical_failure"),
        "Validation_Errors": json.dumps(result.get("validation_errors", []), ensure_ascii=False),
        "Retry_Errors": json.dumps(result.get("retry_errors", []), ensure_ascii=False),
        "Evidence_Repaired": bool(result.get("evidence_repaired")),
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
        "Review_Model": "",
        "Model_Tier": "single_model",
        "Resource_Profile": "remote_user_key",
        "Primary_Decision": decision,
        "Primary_Confidence": assessment["confidence"],
        "Primary_Reason": assessment["reason"],
        "Primary_Evidence_Quote": evidence,
        "Primary_Assessment_JSON": json.dumps(assessment, ensure_ascii=False),
        "Verifier_Decision": "",
        "Verifier_Assessment_JSON": "",
        "Agreement_Status": "single_model",
        "Review_Pending": decision == "MAYBE",
        "Route_Used": "gemini_api",
        "Execution_Origin": origin,
        "Source_Row_Index": source_index,
        "Processing_Seconds": result.get("processing_seconds", 0),
        "Attempt_Count": result.get("attempts", 1),
        "Eval_Token_Count": result.get("token_count", 0),
    })
    return row


def screen_csv_with_gemini_api(
    *, frame: pd.DataFrame, valid: pd.DataFrame, title_col: str, abstract_col: str,
    research_question: str, research_context: str, inclusion_criteria: str,
    exclusion_criteria: str, output_path: str, job_id: str,
    input_fingerprint: str, resume: bool, progress, screening_session,
    api_key: str, model: str = GEMINI_API_MODEL,
    generate: Callable[..., dict[str, Any]] = _request_once,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if not str(api_key or "").strip():
        raise GeminiApiAuthenticationError("A Gemini API key is required.")

    started = time.monotonic()
    protocol_id = _protocol_id(
        research_question, research_context, inclusion_criteria, exclusion_criteria, model,
    )
    source_ids = [str(index) for index in valid.index]
    output = Path(output_path)
    output_root = output.parent.parent if output.parent.name == "runs" else output.parent
    checkpoint = output_root / "cache" / "gemini_api" / (
        _checkpoint_identity(input_fingerprint, protocol_id, source_ids) + ".csv"
    )
    resumed: dict[str, dict[str, Any]] = {}
    if resume and checkpoint.is_file():
        try:
            restored = pd.read_csv(
                checkpoint, dtype=str, keep_default_na=False, encoding="utf-8-sig"
            )
            resumed = {
                str(row.get("Source_Row_Index")): row
                for row in restored.to_dict(orient="records")
                if _reusable(row, protocol_id)
                and str(row.get("Source_Row_Index")) in source_ids
            }
        except (OSError, ValueError):
            resumed = {}

    rows_by_id = dict(resumed)
    progress.set_resumed_count(job_id, len(resumed))
    progress.begin_batches(job_id, "gemini_api_screening", len(valid), len(valid), 1)
    counts = screening_session.counts(list(rows_by_id.values()))
    progress.update_counts(
        job_id, len(rows_by_id), counts["keep"], counts["maybe"], counts["reject"]
    )
    retries = 0

    def ordered_rows() -> list[dict[str, Any]]:
        return [rows_by_id[str(index)] for index in valid.index if str(index) in rows_by_id]

    def persist_partial() -> list[dict[str, Any]]:
        ordered = ordered_rows()
        if ordered:
            _atomic_csv(checkpoint, ordered)
            _atomic_csv(output, ordered)
            screening_session.set_results(
                ordered, job_id=job_id, output_path=output_path,
                architecture_version=ARCHITECTURE_VERSION,
            )
        return ordered

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
                "processing_seconds": 0, "token_count": 0,
            }
            origin = "missing_input"
        else:
            def record_retry() -> None:
                nonlocal retries
                retries += 1
                progress.record_retry(job_id)

            try:
                result = assess_paper(
                    question=research_question, context=research_context,
                    inclusion=inclusion_criteria, exclusion=exclusion_criteria,
                    title=title, abstract=abstract, api_key=api_key, model=model,
                    generate=generate, retry_callback=record_retry, sleep=sleep,
                )
            except GeminiApiError as exc:
                completed = persist_partial()
                if isinstance(exc, GeminiApiQuotaError):
                    problem = "quota or rate limit"
                elif isinstance(exc, GeminiApiAuthenticationError):
                    problem = "key or permission error"
                else:
                    problem = "service interruption"
                raise GeminiApiInterruptedError(
                    f"Gemini API {problem} after {len(completed)} of {len(valid)} papers. "
                    "Completed results were saved. Re-enter a usable key, select resume, "
                    "and run the identical dataset and criteria again."
                ) from exc
            origin = "fresh"

        rows_by_id[source_id] = _result_row(
            source.to_dict(), source_index=source_index, title=title, abstract=abstract,
            result=result, question=research_question, context=research_context,
            inclusion=inclusion_criteria, exclusion=exclusion_criteria,
            protocol_id=protocol_id, model=model, origin=origin,
        )
        ordered = persist_partial()
        counts = screening_session.counts(ordered)
        progress.update_batch(job_id, position, len(ordered))
        progress.update_counts(
            job_id, len(ordered), counts["keep"], counts["maybe"], counts["reject"]
        )

    ordered = persist_partial()
    counts = screening_session.counts(ordered)
    if hasattr(progress, "set_screening_final_metadata"):
        progress.set_screening_final_metadata(
            job_id,
            primary_papers_assessed=len(ordered) - len(resumed),
            primary_direct_keep_count=sum(
                str(row.get("Primary_Decision")) == "KEEP" for row in ordered
            ),
            retry_count=retries,
            fresh_primary_count=len(ordered) - len(resumed),
        )
    progress.update_counts(
        job_id, len(ordered), counts["keep"], counts["maybe"], counts["reject"]
    )
    progress.finish(job_id)
    return {
        **counts,
        "total_papers": len(ordered),
        "output_file": output_path,
        "architecture_version": ARCHITECTURE_VERSION,
        "screening_engine": "gemini_api",
        "model": model,
        "protocol_id": protocol_id,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "retry_count": retries,
        "safe_fallback_count": sum(
            bool(str(row.get("Failure_Class") or "").strip()) for row in ordered
        ),
        "resumed_count": len(resumed),
        "fresh_primary_count": len(ordered) - len(resumed),
        "input_total_rows": len(frame),
        "screened_total_rows": len(ordered),
    }
