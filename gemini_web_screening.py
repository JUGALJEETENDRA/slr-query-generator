from __future__ import annotations

import json
import re
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
    SCHEMA_VERSION, CriterionEvidence, PaperAssessment, ReviewProtocol, StrictModel,
    ValidationReport, safe_maybe,
)
from local_ai.evidence import evidence_lookup
from local_ai.engine import LocalAIOutputError
from local_ai.validator import validate_assessment


GEMINI_WEB_ENGINE = "gemini_web"
GEMINI_WEB_VERSION = "gemini-web-batched-v2.3"
GEMINI_WEB_PROTOCOL_CACHE_VERSION = "gemini-web-protocol-v1"
GEMINI_WEB_LEGACY_PROTOCOL_VERSION = "gemini-web-batched-v2.1"
GEMINI_WEB_BATCH_SIZE = 5
TRANSPORT_TIMEOUT_FAILURE = "transport_timeout"


class WebCriterionEvidence(CriterionEvidence):
    scope_support: Literal["SUBSTANTIVE", "INCIDENTAL", "INSUFFICIENT"]


class WebPaperAssessment(PaperAssessment):
    paper_id: str = Field(min_length=1, max_length=100)
    certainty: Literal["HIGH", "BORDERLINE", "LOW"]
    failure_class: str = ""
    criteria: list[WebCriterionEvidence]


class GeminiWebDiagnostics:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.attempt_count = 0
        self.retry_count = 0
        self.timeout_fallback_count = 0
        self.detector_outcomes: dict[str, int] = {}
        self.recovery_actions: dict[str, int] = {}

    def record(self, event: dict[str, Any]) -> None:
        safe = {
            key: event.get(key, "") for key in (
                "event", "submission_number", "stage", "retry_number", "outcome",
                "recovery_action", "attempt_duration_ms", "response_selector",
                "response_container_count", "response_state", "generation_detected",
                "timeout_stage", "fallback_reason",
            )
        }
        self.attempt_count += int(safe["event"] == "gemini_web_attempt")
        outcome = str(safe["outcome"] or "unknown")
        self.detector_outcomes[outcome] = self.detector_outcomes.get(outcome, 0) + 1
        action = str(safe["recovery_action"] or "")
        if action:
            self.recovery_actions[action] = self.recovery_actions.get(action, 0) + 1
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(safe, ensure_ascii=False) + "\n")

    def record_retry(self) -> None:
        self.retry_count += 1

    def record_timeout_fallback(self, reason: str) -> None:
        self.timeout_fallback_count += 1
        self.record({
            "event": "gemini_web_fallback", "outcome": "safe_maybe",
            "fallback_reason": str(reason),
        })


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


def _protocol_hash(
    question: str, context: str, inclusion: str, exclusion: str, *,
    version: str = GEMINI_WEB_PROTOCOL_CACHE_VERSION,
) -> str:
    return sha256(json.dumps({
        "question": question, "context": context, "inclusion": inclusion,
        "exclusion": exclusion, "version": version,
    }, sort_keys=True).encode("utf-8")).hexdigest()


def _validate_protocol(protocol: ReviewProtocol, inclusion: str, exclusion: str) -> None:
    if any(c.source == "research_question" and c.kind == "exclusion" for c in protocol.criteria):
        raise ValueError("RQ scope must use positive inclusion criteria, not inferred exclusions")
    if inclusion.strip() and not any(c.source == "user" and c.kind == "inclusion" for c in protocol.criteria):
        raise ValueError("explicit user inclusion criteria were not preserved")
    if exclusion.strip() and not any(c.source == "user" and c.kind == "exclusion" for c in protocol.criteria):
        raise ValueError("explicit user exclusion criteria were not preserved")


def _set_attempt_context(browser, stage: str, retry_number: int) -> None:
    method = getattr(browser, "set_attempt_context", None)
    if callable(method):
        method(stage=stage, retry_number=retry_number)


def _recover_browser(browser, action: str, *, exhausted: bool = False) -> None:
    recover_transport = getattr(browser, "recover_transport_failure", None)
    if callable(recover_transport):
        recover_transport(exhausted=exhausted)
        return
    note = getattr(browser, "note_recovery", None)
    if callable(note):
        note(action)
    browser.recover_job_chat()


def _compile_protocol(browser, question: str, context: str, inclusion: str, exclusion: str) -> ReviewProtocol:
    base = protocol_prompt(question, inclusion, exclusion, context)
    last_error: Exception | None = None
    for attempt in range(2):
        prompt = base if attempt == 0 else (
            base + "\n\nCORRECTION: The previous protocol was invalid. Return a corrected JSON protocol only. "
            + str(last_error)
        )
        try:
            _set_attempt_context(browser, "protocol", attempt)
            raw = browser.submit_prompt_and_get_response(prompt)
            value = parse_structured_model_output(raw, ReviewProtocol)
            protocol = ReviewProtocol.model_validate(value).model_copy(update={
                "research_question": question,
                "research_context": context,
                "prompt_version": GEMINI_WEB_PROTOCOL_CACHE_VERSION,
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
                _recover_browser(browser, "new_job_chat")
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


def _fallback_item(
    paper: ScreeningPaper, protocol: ReviewProtocol, reason: str, *, failure_class: str = "",
) -> WebPaperAssessment:
    fallback = safe_maybe(reason)
    criteria = [{
        "criterion_id": criterion.id, "verdict": "UNCLEAR",
        "rationale": "Gemini Web output could not be validated.", "evidence": [],
        "scope_support": "INSUFFICIENT",
    } for criterion in protocol.criteria]
    return WebPaperAssessment(
        **fallback.model_dump(exclude={"criteria"}), paper_id=paper.paper_id,
        certainty="LOW", criteria=criteria, failure_class=failure_class,
    )


def _execute_batch(
    browser, protocol: ReviewProtocol, papers: list[ScreeningPaper], *, critic: bool,
    prior: dict[str, dict] | None, record_retry: Callable[[], None],
    record_timeout_fallback: Callable[[str], None] | None = None,
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
    last_failure_was_transport = False
    for attempt in range(2):
        try:
            _set_attempt_context(browser, "critic" if critic else "primary", attempt)
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
            last_failure_was_transport = (
                isinstance(exc, (TimeoutError, RuntimeError))
                and not isinstance(exc, LocalAIOutputError)
            )
            record_retry()
            if last_failure_was_transport and attempt == 0:
                _recover_browser(browser, "new_job_chat")

    if last_failure_was_transport:
        # Splitting cannot repair a broken browser or timed-out connection and
        # would multiply the timeout across a recursive tree. Preserve one safe
        # result per paper and reset the chat for the next independent batch.
        try:
            _recover_browser(
                browser, "new_job_chat_after_exhausted_retry", exhausted=True,
            )
        except Exception:
            pass
        reason = f"Gemini Web browser request failed after retry: {last_error}"
        if record_timeout_fallback is not None:
            record_timeout_fallback(reason)
        return {
            paper.paper_id: _fallback_item(paper, protocol, reason, failure_class=TRANSPORT_TIMEOUT_FAILURE)
            for paper in papers
        }

    if len(papers) > 1:
        midpoint = max(1, len(papers) // 2)
        combined = _execute_batch(
            browser, protocol, papers[:midpoint], critic=critic, prior=prior,
            record_retry=record_retry, record_timeout_fallback=record_timeout_fallback,
        )
        combined.update(_execute_batch(
            browser, protocol, papers[midpoint:], critic=critic, prior=prior,
            record_retry=record_retry, record_timeout_fallback=record_timeout_fallback,
        ))
        return combined
    paper = papers[0]
    return {paper.paper_id: _fallback_item(
        paper, protocol, f"Gemini Web could not return a valid assessment: {last_error}"
    )}


def _assessment(item: WebPaperAssessment) -> PaperAssessment:
    payload = item.model_dump(exclude={"paper_id", "certainty", "failure_class"})
    payload["criteria"] = [
        {key: value for key, value in criterion.items() if key != "scope_support"}
        for criterion in payload["criteria"]
    ]
    return PaperAssessment.model_validate(payload)


def _scope_support_errors(
    item: WebPaperAssessment, protocol: ReviewProtocol,
) -> list[str]:
    protocol_by_id = {criterion.id: criterion for criterion in protocol.criteria}
    errors: list[str] = []
    for criterion in item.criteria:
        expected = protocol_by_id.get(criterion.criterion_id)
        if not expected or expected.kind != "inclusion" or not expected.required:
            continue
        if criterion.verdict == "MET" and criterion.scope_support != "SUBSTANTIVE":
            errors.append(
                "required inclusion MET lacks substantive study-scope support: "
                f"{criterion.criterion_id} ({criterion.scope_support})"
            )
        if (
            criterion.scope_support in {"INCIDENTAL", "INSUFFICIENT"}
            and criterion.verdict != "UNCLEAR"
        ):
            errors.append(
                "incidental or insufficient required inclusion support must be UNCLEAR: "
                f"{criterion.criterion_id} ({criterion.verdict})"
            )
    return errors


def _is_failure_fallback(item: WebPaperAssessment) -> bool:
    return (
        item.decision == "MAYBE"
        and item.certainty == "LOW"
        and item.confidence == 0
        and item.summary == "Assessment could not be validated."
    )


_GROUNDING_STOPWORDS = {
    "about", "against", "application", "applied", "applies", "criterion",
    "directly", "evidence", "include", "included", "inclusion", "method",
    "paper", "provides", "required", "research", "study", "technology",
    "their", "these", "those", "using", "within",
}


def _words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value or "").casefold())


def _acronyms(value: str) -> set[str]:
    candidates = re.findall(r"\b[A-Za-z][A-Za-z0-9]{1,9}\b", str(value or ""))
    return {
        candidate for candidate in candidates
        if sum(character.isupper() for character in candidate) >= 2
    }


def _acronym_expansion(acronym: str, source_text: str) -> str:
    escaped = re.escape(acronym)
    word = r"[A-Za-z][A-Za-z0-9/-]*"
    before = re.search(
        rf"(?P<long>{word}(?:\s+{word}){{1,8}})\s*\(\s*{escaped}\s*\)",
        source_text,
    )
    if before:
        return before.group("long")
    after = re.search(
        rf"\b{escaped}\b\s*\(\s*(?P<long>{word}(?:\s+{word}){{1,8}})\s*\)",
        source_text,
    )
    return after.group("long") if after else ""


def _criterion_support_phrases(description: str, expected_evidence: str) -> set[tuple[str, ...]]:
    tokens = [
        token for token in _words(f"{description} {expected_evidence}")
        if token not in _GROUNDING_STOPWORDS
    ]
    phrases: set[tuple[str, ...]] = set()
    for size in (4, 3, 2):
        phrases.update(tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1))
    phrases.update((token,) for token in tokens if len(token) >= 7)
    return phrases


def _contains_phrase(source_tokens: list[str], phrase: tuple[str, ...]) -> bool:
    size = len(phrase)
    return any(
        tuple(source_tokens[index:index + size]) == phrase
        for index in range(len(source_tokens) - size + 1)
    )


def _acronym_grounding_errors(
    assessment: PaperAssessment,
    protocol: ReviewProtocol,
    title: str,
    abstract: str,
) -> list[str]:
    if assessment.decision != "KEEP":
        return []
    source_text = f"{title}\n{abstract}"
    source_tokens = _words(source_text)
    source_units = evidence_lookup(title, abstract)
    protocol_by_id = {criterion.id: criterion for criterion in protocol.criteria}
    errors: list[str] = []
    for item in assessment.criteria:
        criterion = protocol_by_id.get(item.criterion_id)
        if not criterion or criterion.kind != "inclusion" or not criterion.required or item.verdict != "MET":
            continue
        cited_text = " ".join(
            str(source_units[span.evidence_id]["text"])
            for span in item.evidence
            if span.evidence_id in source_units
            and source_units[span.evidence_id].get("source") == span.source
        )
        cited_acronyms = _acronyms(cited_text)
        if not cited_acronyms:
            continue
        criterion_text = f"{criterion.description} {criterion.expected_evidence}"
        if any(re.search(rf"\b{re.escape(acronym)}\b", criterion_text) for acronym in cited_acronyms):
            continue
        phrases = _criterion_support_phrases(criterion.description, criterion.expected_evidence)
        if any(_contains_phrase(source_tokens, phrase) for phrase in phrases):
            continue
        unsupported = []
        criterion_tokens = {
            token for token in _words(criterion_text)
            if token not in _GROUNDING_STOPWORDS and len(token) >= 4
        }
        for acronym in sorted(cited_acronyms):
            expansion = _acronym_expansion(acronym, source_text)
            expansion_tokens = {
                token for token in _words(expansion)
                if token not in _GROUNDING_STOPWORDS and len(token) >= 4
            }
            if not expansion_tokens or not criterion_tokens.intersection(expansion_tokens):
                unsupported.append(acronym)
        if unsupported:
            errors.append(
                "required criterion relies on an acronym whose eligibility meaning is not grounded "
                f"in the supplied source: {criterion.id} ({', '.join(unsupported)})"
            )
    return errors


def _public_result(
    item: WebPaperAssessment, protocol: ReviewProtocol, paper: ScreeningPaper, *,
    stage: str, elapsed: float, prior_trace: list[dict] | None = None,
) -> dict[str, Any]:
    assessment = _assessment(item)
    validation = validate_assessment(assessment, protocol, paper.title, paper.abstract)
    grounding_errors = _acronym_grounding_errors(
        assessment, protocol, paper.title, paper.abstract,
    )
    scope_support_errors = _scope_support_errors(item, protocol)
    if grounding_errors or scope_support_errors:
        validation = validation.model_copy(update={
            "valid": False,
            "errors": [*validation.errors, *grounding_errors, *scope_support_errors],
        })
    units = evidence_lookup(paper.title, paper.abstract)
    scope_support_by_id = {
        criterion.criterion_id: criterion.scope_support for criterion in item.criteria
    }
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
            "scope_support": scope_support_by_id[criterion.criterion_id],
        })
    certainty_cap = {"HIGH": .92, "BORDERLINE": .68, "LOW": .4}[item.certainty]
    reported_confidence = round(min(certainty_cap, assessment.confidence), 2)
    if not validation.valid or assessment.contradictions:
        risk = "HIGH"
    elif item.certainty == "HIGH" and reported_confidence >= .75:
        risk = "LOW"
    elif item.certainty == "BORDERLINE" or reported_confidence >= .5:
        risk = "BORDERLINE"
    else:
        risk = "HIGH"
    trace = list(prior_trace or []) + [{
        "name": stage, "decision": assessment.decision, "certainty": item.certainty,
        "validation_status": "validated" if validation.valid else "unresolved",
        "validation_errors": validation.errors,
    }]
    return {
        "schema_version": SCHEMA_VERSION, "decision": assessment.decision,
        "reason": assessment.reason, "confidence": reported_confidence,
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
        "failure_class": item.failure_class, "critic_route": "",
        "verification_status": "not_required",
    }


def _critic_route(result: dict[str, Any], protocol: ReviewProtocol) -> str:
    if result["validation_status"] != "validated":
        return "validation_failure"
    if result.get("contradictions"):
        return "contradiction"
    if result["decision"] == "REJECT":
        verdicts = {
            str(item.get("criterion_id") or ""): str(item.get("verdict") or "")
            for item in result.get("criteria", []) if isinstance(item, dict)
        }
        unmet_required = any(
            criterion.kind == "inclusion" and criterion.required
            and verdicts.get(criterion.id) == "NOT_MET"
            for criterion in protocol.criteria
        )
        met_exclusion = any(
            criterion.kind == "exclusion" and verdicts.get(criterion.id) == "MET"
            for criterion in protocol.criteria
        )
        if unmet_required and not met_exclusion:
            return "inclusion_only_reject"
    if result["decision"] in {"KEEP", "REJECT"} and result["decision_risk"] != "LOW":
        return "risky_definitive"
    return ""


def _needs_critic(result: dict[str, Any], protocol: ReviewProtocol | None = None) -> bool:
    if protocol is None:
        return (
            result["validation_status"] != "validated"
            or bool(result.get("contradictions"))
            or (
                result["decision"] in {"KEEP", "REJECT"}
                and result["decision_risk"] != "LOW"
            )
        )
    return bool(_critic_route(result, protocol))


def _safe_json_list(value: Any, *, invalid: list | None = None) -> list:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value or "[]"))
        return parsed if isinstance(parsed, list) else list(invalid or [])
    except (json.JSONDecodeError, TypeError):
        return list(invalid or [])


def _is_json_list(value: Any) -> bool:
    if isinstance(value, list):
        return True
    try:
        return isinstance(json.loads(str(value)), list)
    except (json.JSONDecodeError, TypeError):
        return False


def _write_rows(path: Path, rows: list[dict]) -> None:
    """Atomically publish a checkpoint so interruption cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def _write_protocol(path: Path, protocol: ReviewProtocol) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(protocol.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_cached_protocol(
    cache_root: Path, question: str, context: str, inclusion: str, exclusion: str,
) -> tuple[ReviewProtocol | None, Path]:
    canonical = cache_root / "protocols" / f"{_protocol_hash(question, context, inclusion, exclusion)}.json"
    legacy = cache_root / "protocols" / f"{_protocol_hash(
        question, context, inclusion, exclusion, version=GEMINI_WEB_LEGACY_PROTOCOL_VERSION,
    )}.json"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    for candidate in (canonical, legacy):
        try:
            protocol = ReviewProtocol.model_validate_json(candidate.read_text(encoding="utf-8"))
            _validate_protocol(protocol, inclusion, exclusion)
        except (OSError, ValueError):
            continue
        if candidate != canonical and not canonical.exists():
            _write_protocol(canonical, protocol)
        return protocol, canonical
    return None, canonical


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
        "Triage_Basis": result["triage_basis"], "Failure_Class": result.get("failure_class", ""),
        "Critic_Route": result.get("critic_route", ""),
        "Verification_Status": result.get("verification_status", "not_required"),
        "Source_Row_Index": source_index,
    })
    return row


def _resume_rows(path: Path, protocol_id: str, expected_ids: set[str] | None = None) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
        required = {
            "Source_Row_Index", "Protocol_ID", "Prompt_Version", "Layer_Trace_JSON",
            "Decision", "Validation_Status", "Criteria_JSON", "Evidence_JSON",
            "Critic_Route", "Verification_Status",
        }
        if not required.issubset(frame.columns):
            return {}
        frame = frame[
            (frame["Protocol_ID"].astype(str) == protocol_id)
            & (frame["Prompt_Version"].astype(str) == GEMINI_WEB_VERSION)
        ]
        rows = {}
        for _, source_row in frame.iterrows():
            row = source_row.to_dict()
            key = str(row["Source_Row_Index"])
            if expected_ids is not None and key not in expected_ids:
                continue
            if row.get("Decision") not in {"KEEP", "MAYBE", "REJECT"}:
                continue
            if (
                str(row.get("Failure_Class") or "") == TRANSPORT_TIMEOUT_FAILURE
                or str(row.get("Reason") or "").startswith("Gemini Web browser request failed after retry:")
            ):
                continue
            if not all(_is_json_list(row.get(column)) for column in (
                "Layer_Trace_JSON", "Criteria_JSON", "Evidence_JSON",
            )):
                continue
            row["Cache_Hit"] = True
            row["Processing_Seconds"] = 0.0
            rows[key] = row
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
    run_started = time.perf_counter()
    contract = _contract_hash(
        input_fingerprint, research_question, research_context, inclusion_criteria, exclusion_criteria
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_root = output.parent.parent / "cache" / "gemini_web"
    checkpoint = cache_root / "checkpoints" / f"{contract}.csv"
    diagnostics = GeminiWebDiagnostics(cache_root / "diagnostics" / f"{contract}-{job_id}.jsonl")
    protocol, protocol_file = _load_cached_protocol(
        cache_root, research_question, research_context, inclusion_criteria, exclusion_criteria,
    )

    papers: dict[str, ScreeningPaper] = {}
    sources: dict[str, tuple[Any, dict]] = {}
    for source_index, source_row in valid.iterrows():
        key = str(source_index)
        title = "" if pd.isna(source_row[title_col]) else str(source_row[title_col])
        abstract = "" if pd.isna(source_row[abstract_col]) else str(source_row[abstract_col])
        papers[key] = ScreeningPaper(key, title, abstract)
        sources[key] = (source_index, source_row.to_dict())

    browser_context = None
    browser = None
    if protocol is None:
        progress.begin_batches(job_id, "gemini_web_protocol", 1, 1, 1)
        browser_context = browser_factory(GeminiWebConfig(diagnostic_sink=diagnostics.record))
        try:
            browser = browser_context.__enter__()
            protocol = _compile_protocol(
                browser, research_question, research_context, inclusion_criteria, exclusion_criteria
            )
            _write_protocol(protocol_file, protocol)
            progress.update_batch(job_id, 1, 1)
            note = getattr(browser, "note_recovery", None)
            if callable(note):
                note("protocol_to_primary_clean_chat")
            browser.start_new_job_chat()
        except Exception:
            browser_context.__exit__(None, None, None)
            browser_context = None
            browser = None
            raise

    rows = _resume_rows(checkpoint, protocol.protocol_id, set(papers)) if resume else {}
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
            browser_context = browser_factory(GeminiWebConfig(diagnostic_sink=diagnostics.record))
            browser = browser_context.__enter__()

        def retry():
            progress.record_retry(job_id)
            diagnostics.record_retry()

        primary_batches = (len(pending) + GEMINI_WEB_BATCH_SIZE - 1) // GEMINI_WEB_BATCH_SIZE
        progress.begin_batches(
            job_id, "gemini_web_primary", len(pending), primary_batches, GEMINI_WEB_BATCH_SIZE
        )
        for batch_number in range(primary_batches):
            batch = pending[batch_number * GEMINI_WEB_BATCH_SIZE:(batch_number + 1) * GEMINI_WEB_BATCH_SIZE]
            started = time.perf_counter()
            assessed = _execute_batch(
                browser, protocol, batch, critic=False, prior=None, record_retry=retry,
                record_timeout_fallback=diagnostics.record_timeout_fallback,
            )
            elapsed_each = (time.perf_counter() - started) / max(1, len(batch))
            for paper in batch:
                result = _public_result(
                    assessed[paper.paper_id], protocol, paper,
                    stage="gemini_web_primary", elapsed=elapsed_each,
                )
                route = _critic_route(result, protocol)
                result["critic_route"] = route
                result["verification_status"] = "pending" if route else "not_required"
                source_index, source = sources[paper.paper_id]
                rows[paper.paper_id] = _row(source, source_index, paper, result)
            ordered = [rows[key] for key in papers if key in rows]
            _write_rows(checkpoint, ordered)
            _write_rows(output, ordered)
            counts = screening_session.counts(ordered)
            progress.update_batch(job_id, batch_number + 1, min(len(pending), (batch_number + 1) * 5))
            progress.update_counts(job_id, len(ordered), counts["keep"], counts["maybe"], counts["reject"])

        critic_keys = []
        for key in papers:
            row = rows.get(key)
            if not row:
                continue
            trace = _safe_json_list(row.get("Layer_Trace_JSON"))
            if trace and trace[-1].get("name") == "gemini_web_critic":
                continue
            contradictions = _safe_json_list(
                row.get("Contradictions_JSON"), invalid=["unparseable contradictions"]
            )
            criteria = _safe_json_list(row.get("Criteria_JSON"), invalid=[{"invalid": True}])
            route = _critic_route({
                "decision": row.get("Decision"),
                "decision_risk": row.get("Decision_Risk"),
                "validation_status": row.get("Validation_Status"),
                "contradictions": contradictions,
                "criteria": criteria,
            }, protocol)
            row["Critic_Route"] = route
            row["Verification_Status"] = "pending" if route else "not_required"
            if route:
                critic_keys.append(key)

        critic_batches = (len(critic_keys) + 4) // 5
        if critic_keys and browser is None:
            browser_context = browser_factory(GeminiWebConfig(diagnostic_sink=diagnostics.record))
            browser = browser_context.__enter__()
        if critic_keys:
            note = getattr(browser, "note_recovery", None)
            if callable(note):
                note("primary_to_critic_clean_chat")
            browser.start_new_job_chat()
        progress.begin_batches(job_id, "gemini_web_critic", len(critic_keys), critic_batches, 5)
        for batch_number in range(critic_batches):
            keys = critic_keys[batch_number * 5:(batch_number + 1) * 5]
            batch = [papers[key] for key in keys]
            prior = {key: {
                "validation_errors": _safe_json_list(rows[key].get("Validation_Errors")),
                "contradictions": _safe_json_list(rows[key].get("Contradictions_JSON")),
            } for key in keys}
            started = time.perf_counter()
            assessed = _execute_batch(
                browser, protocol, batch, critic=True, prior=prior, record_retry=retry,
                record_timeout_fallback=diagnostics.record_timeout_fallback,
            )
            elapsed_each = (time.perf_counter() - started) / max(1, len(batch))
            for paper in batch:
                primary_row = rows[paper.paper_id]
                primary_trace = _safe_json_list(primary_row.get("Layer_Trace_JSON"))
                candidate = _public_result(
                    assessed[paper.paper_id], protocol, paper,
                    stage="gemini_web_critic", elapsed=elapsed_each, prior_trace=primary_trace,
                )
                route = str(primary_row.get("Critic_Route") or "")
                primary_valid = primary_row.get("Validation_Status") == "validated"
                primary_decision = str(primary_row.get("Decision") or "")
                if (
                    candidate["validation_status"] != "validated"
                    or _is_failure_fallback(assessed[paper.paper_id])
                ):
                    failure_class = assessed[paper.paper_id].failure_class
                    candidate = _public_result(
                        _fallback_item(
                            paper, protocol,
                            "Independent verification was unavailable; no provisional definitive decision was retained.",
                            failure_class=failure_class,
                        ),
                        protocol, paper, stage="gemini_web_critic", elapsed=elapsed_each,
                        prior_trace=primary_trace,
                    )
                    candidate["verification_status"] = "failed"
                elif primary_valid and primary_decision in {"KEEP", "REJECT"} and candidate["decision"] != primary_decision:
                    status = "uncertain" if candidate["decision"] == "MAYBE" else "disagreed"
                    candidate = _public_result(
                        _fallback_item(
                            paper, protocol,
                            "Independent evidence-valid screeners did not agree on a definitive decision."
                        ),
                        protocol, paper, stage="gemini_web_critic", elapsed=elapsed_each,
                        prior_trace=primary_trace,
                    )
                    candidate["verification_status"] = status
                elif candidate["decision"] == "MAYBE":
                    candidate["verification_status"] = "uncertain"
                else:
                    candidate["verification_status"] = "agreed"
                candidate["critic_route"] = route
                source_index, source = sources[paper.paper_id]
                rows[paper.paper_id] = _row(source, source_index, paper, candidate)
            ordered = [rows[key] for key in papers]
            _write_rows(checkpoint, ordered)
            _write_rows(output, ordered)
            counts = screening_session.counts(ordered)
            progress.update_batch(job_id, batch_number + 1, min(len(critic_keys), (batch_number + 1) * 5))
            progress.update_counts(job_id, len(ordered), counts["keep"], counts["maybe"], counts["reject"])

        progress.begin_batches(job_id, "gemini_web_finalizing", len(papers), 1, len(papers) or 1)
        ordered = [rows[key] for key in papers]
        _write_rows(output, ordered)
        screening_session.set_results(
            ordered, job_id=job_id, output_path=output_path,
            architecture_version=GEMINI_WEB_VERSION,
        )
        final = screening_session.counts(ordered)
        progress.update_batch(job_id, 1, len(papers))
        progress.update_counts(job_id, len(ordered), final["keep"], final["maybe"], final["reject"])
        progress.finish(job_id)
        runtime_seconds = round(time.perf_counter() - run_started, 4)
        critic_route_counts: dict[str, int] = {}
        verification_outcomes: dict[str, int] = {}
        for row in ordered:
            route = str(row.get("Critic_Route") or "")
            status = str(row.get("Verification_Status") or "not_required")
            if route:
                critic_route_counts[route] = critic_route_counts.get(route, 0) + 1
            verification_outcomes[status] = verification_outcomes.get(status, 0) + 1
        clean_chat_rotations = sum(
            diagnostics.recovery_actions.get(action, 0)
            for action in ("protocol_to_primary_clean_chat", "primary_to_critic_clean_chat")
        )
        diagnostics_summary = {
            "runtime_seconds": runtime_seconds,
            "retry_count": diagnostics.retry_count,
            "timeout_fallback_count": diagnostics.timeout_fallback_count,
            "attempt_count": diagnostics.attempt_count,
            "detector_outcomes": diagnostics.detector_outcomes,
            "recovery_actions": diagnostics.recovery_actions,
            "critic_route_counts": critic_route_counts,
            "verification_outcomes": verification_outcomes,
            "verified_reject_count": sum(
                row.get("Decision") == "REJECT"
                and row.get("Verification_Status") == "agreed"
                for row in ordered
            ),
            "verification_fallback_count": sum(
                row.get("Verification_Status") in {"disagreed", "uncertain", "failed"}
                for row in ordered
            ),
            "protocol_cache_version": GEMINI_WEB_PROTOCOL_CACHE_VERSION,
            "clean_chat_rotations": clean_chat_rotations,
            "diagnostics_path": str(diagnostics.path),
        }
        diagnostics.path.with_suffix(".summary.json").write_text(
            json.dumps(diagnostics_summary, indent=2), encoding="utf-8"
        )
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
            **diagnostics_summary,
        }
    finally:
        if browser_context is not None:
            browser_context.__exit__(None, None, None)
