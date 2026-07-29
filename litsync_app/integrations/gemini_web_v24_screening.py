from __future__ import annotations

import json
import re
import time
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
from pydantic import ConfigDict, Field, model_validator

from litsync_app.screening.external.engine import parse_structured_model_output
from litsync_app.integrations.gemini_web_v24_automation import GeminiWebV24Automation, GeminiWebV24Config
from litsync_app.integrations.gemini_web_v24_prompt import (
    V24Paper,
    authoritative_criterion_entries,
    build_primary_prompt,
    build_protocol_prompt,
    build_verification_prompt,
)
from litsync_app.screening.local.contracts import SCHEMA_VERSION, StrictModel
from litsync_app.screening.local.engine import LocalAIOutputError
from litsync_app.screening.local.evidence import evidence_lookup
from litsync_app.benchmarking.provenance import (
    screening_output_fingerprint,
    source_dataframe_fingerprint,
)


GEMINI_WEB_V24_ENGINE = "gemini_web_v24"
GEMINI_WEB_V24_VERSION = "gemini-web-batched-v2.4"
GEMINI_WEB_V24_PROTOCOL_VERSION = "gemini-web-v2.4-protocol-v3"
GEMINI_WEB_V24_ASSESSMENT_PROMPT_VERSION = "gemini-web-v2.4-assessment-prompt-v5"
GEMINI_WEB_V24_CACHE_VERSION = "gemini-web-v2.4-assessment-v5"
MAX_BATCH_PAPERS = 5
MAX_ESTIMATED_OUTPUT_BYTES = 8192
MAX_CRITERION_OBJECTS_PER_BATCH = 25
BASE_OUTPUT_BYTES_PER_PAPER = 400
OUTPUT_BYTES_PER_CRITERION = 310
# Compatibility import for callers which only need the historical upper bound.
GEMINI_WEB_V24_BATCH_SIZE = MAX_BATCH_PAPERS
GEMINI_WEB_V24_MAX_CRITERIA_PER_KIND = 20
V24_STRUCTURED_OUTPUT_FAILURE = "invalid_structured_response"
V24_TRANSPORT_FAILURE = "transport_timeout"
V24_VERIFICATION_FAILURE = "verification_transport_failure"


class V24Criterion(StrictModel):
    id: str = Field(min_length=1, max_length=80)
    kind: Literal["inclusion", "exclusion"]
    description: str = Field(min_length=3, max_length=600)
    required: bool = True
    expected_evidence: str = Field(min_length=3, max_length=600)
    source: Literal["research_question", "user"] = "research_question"
    authoritative_text: str = Field(default="", max_length=4000)
    is_composite_relationship: bool = False


class V24Protocol(StrictModel):
    protocol_version: str = GEMINI_WEB_V24_PROTOCOL_VERSION
    protocol_id: str = ""
    research_question: str = Field(min_length=3)
    objective: str = Field(min_length=3, max_length=1200)
    population_or_subject: list[str] = Field(default_factory=list, max_length=20)
    methods_or_interventions: list[str] = Field(default_factory=list, max_length=20)
    target_tasks_or_outcomes: list[str] = Field(default_factory=list, max_length=20)
    application_context: list[str] = Field(default_factory=list, max_length=20)
    required_inclusion_criteria: list[V24Criterion] = Field(min_length=1, max_length=20)
    exclusion_boundaries: list[V24Criterion] = Field(default_factory=list, max_length=20)
    ambiguities: list[str] = Field(default_factory=list, max_length=20)
    synonyms_and_equivalent_concepts: list[str] = Field(default_factory=list, max_length=30)
    near_neighbor_but_out_of_scope_concepts: list[str] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def validate_criteria(self):
        criteria = [*self.required_inclusion_criteria, *self.exclusion_boundaries]
        identifiers = [criterion.id for criterion in criteria]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("protocol criterion IDs must be unique")
        if any(criterion.kind != "inclusion" for criterion in self.required_inclusion_criteria):
            raise ValueError("required inclusion criteria must use inclusion kind")
        if any(criterion.kind != "exclusion" for criterion in self.exclusion_boundaries):
            raise ValueError("exclusion boundaries must use exclusion kind")
        return self

    @property
    def criteria(self) -> list[V24Criterion]:
        return [*self.required_inclusion_criteria, *self.exclusion_boundaries]

    def with_identity(self) -> "V24Protocol":
        payload = self.model_dump(exclude={"protocol_id"}, mode="json")
        digest = sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return self.model_copy(update={"protocol_id": digest})


class V24EvidenceReference(StrictModel):
    source: Literal["title", "abstract"]
    evidence_id: str = Field(min_length=1, max_length=40)


class V24CriterionAssessment(StrictModel):
    criterion_id: str = Field(min_length=1, max_length=80)
    verdict: Literal["MET", "NOT_MET", "UNCLEAR"]
    scope_support: Literal["SUBSTANTIVE", "INCIDENTAL", "INSUFFICIENT"]
    evidence_relationship: Literal["SUPPORTS", "CONFLICTS", "INCIDENTAL", "INSUFFICIENT"]
    rationale: str = Field(min_length=1, max_length=600)
    evidence: list[V24EvidenceReference] = Field(default_factory=list, max_length=2)


class V24Assessment(StrictModel):
    paper_id: str = Field(min_length=1, max_length=100)
    decision: Literal["KEEP", "MAYBE", "REJECT"]
    confidence: float = Field(ge=0, le=1)
    decision_risk: Literal["LOW", "BORDERLINE", "HIGH"]
    reason: str = Field(min_length=1, max_length=600)
    criterion_assessments: list[V24CriterionAssessment] = Field(min_length=1, max_length=40)


class V24AssessmentBatch(StrictModel):
    model_config = ConfigDict(extra="forbid")
    items: list[V24Assessment] = Field(min_length=1, max_length=MAX_BATCH_PAPERS)


class V24CompactEvidenceReference(StrictModel):
    s: Literal["title", "abstract"]
    e: str = Field(min_length=1, max_length=40)

    def expand(self) -> V24EvidenceReference:
        return V24EvidenceReference(source=self.s, evidence_id=self.e)


class V24CompactCriterionAssessment(StrictModel):
    c: str = Field(min_length=1, max_length=80)
    v: Literal["MET", "NOT_MET", "UNCLEAR"]
    u: Literal["SUBSTANTIVE", "INCIDENTAL", "INSUFFICIENT"]
    l: Literal["SUPPORTS", "CONFLICTS", "INCIDENTAL", "INSUFFICIENT"]
    r: str = Field(min_length=1, max_length=600)
    e: list[V24CompactEvidenceReference] = Field(default_factory=list, max_length=2)

    def expand(self) -> V24CriterionAssessment:
        return V24CriterionAssessment(
            criterion_id=self.c,
            verdict=self.v,
            scope_support=self.u,
            evidence_relationship=self.l,
            rationale=self.r,
            evidence=[reference.expand() for reference in self.e],
        )


class V24CompactAssessment(StrictModel):
    p: str = Field(min_length=1, max_length=100)
    d: Literal["KEEP", "MAYBE", "REJECT"]
    f: float = Field(ge=0, le=1)
    k: Literal["LOW", "BORDERLINE", "HIGH"]
    r: str = Field(min_length=1, max_length=600)
    c: list[V24CompactCriterionAssessment] = Field(min_length=1, max_length=40)

    def expand(self) -> V24Assessment:
        return V24Assessment(
            paper_id=self.p,
            decision=self.d,
            confidence=self.f,
            decision_risk=self.k,
            reason=self.r,
            criterion_assessments=[criterion.expand() for criterion in self.c],
        )


class V24CompactAssessmentBatch(StrictModel):
    model_config = ConfigDict(extra="forbid")
    items: list[V24CompactAssessment] = Field(min_length=1, max_length=MAX_BATCH_PAPERS)


def _assessment_contract() -> dict[str, Any]:
    """Compact strict wire contract shared by primary and verification."""
    text = {"type": "string"}
    evidence = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "s": {"enum": ["title", "abstract"]},
            "e": text,
        },
        "required": ["s", "e"],
    }
    criterion = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "c": text,
            "v": {"enum": ["MET", "NOT_MET", "UNCLEAR"]},
            "u": {"enum": ["SUBSTANTIVE", "INCIDENTAL", "INSUFFICIENT"]},
            "l": {"enum": ["SUPPORTS", "CONFLICTS", "INCIDENTAL", "INSUFFICIENT"]},
            "r": text,
            "e": {"type": "array", "maxItems": 2, "items": evidence},
        },
        "required": ["c", "v", "u", "l", "r", "e"],
    }
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "p": text,
            "d": {"enum": ["KEEP", "MAYBE", "REJECT"]},
            "f": {"type": "number", "minimum": 0, "maximum": 1},
            "k": {"enum": ["LOW", "BORDERLINE", "HIGH"]},
            "r": text,
            "c": {"type": "array", "minItems": 1, "maxItems": 40, "items": criterion},
        },
        "required": ["p", "d", "f", "k", "r", "c"],
    }
    return {
        "name": "v24_compact_assessment_batch_v5",
        "criterion_array_field": "c",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_BATCH_PAPERS,
                    "items": item,
                },
            },
            "required": ["items"],
        },
    }


def _criterion_objects_per_item(
    stage: Literal["primary", "verification"],
    protocol: V24Protocol,
) -> int:
    # Both stages currently use the same strict contract. Keep this stage-aware
    # so a future verifier contract cannot silently inherit primary budgeting.
    contract = _assessment_contract()
    if contract["criterion_array_field"] != "c":
        raise ValueError(f"unsupported {stage} assessment response contract")
    return len(protocol.criteria)


def plan_assessment_batch_size(
    protocol: V24Protocol,
    *,
    stage: Literal["primary", "verification"],
) -> tuple[int, bool]:
    criterion_objects = _criterion_objects_per_item(stage, protocol)
    estimated_per_paper = (
        BASE_OUTPUT_BYTES_PER_PAPER
        + OUTPUT_BYTES_PER_CRITERION * criterion_objects
    )
    for paper_count in range(MAX_BATCH_PAPERS, 0, -1):
        if (
            estimated_per_paper * paper_count <= MAX_ESTIMATED_OUTPUT_BYTES
            and criterion_objects * paper_count
            <= MAX_CRITERION_OBJECTS_PER_BATCH
        ):
            return paper_count, False
    return 1, True


class V24Diagnostics:
    APPROVED_FIELDS = (
        "event", "submission_number", "stage", "retry_number", "outcome",
        "recovery_action", "attempt_duration_ms", "response_selector",
        "response_container_count", "response_state", "generation_detected",
        "timeout_stage", "fallback_reason", "failure_class", "paper_count",
        "paper_ids", "batch_id", "subgroup_id", "criterion_count",
        "expected_criterion_object_count", "prompt_utf8_bytes",
        "response_utf8_bytes", "parsed_item_count", "over_budget",
    )

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.retry_count = 0
        self.fallback_count = 0
        self.attempt_count = 0
        self.outcomes: dict[str, int] = {}
        self.recoveries: dict[str, int] = {}
        self.degraded_subgroup_replay_count = 0
        self.degraded_subgroup_replay_success_count = 0
        self.degraded_subgroup_replay_exhaustion_count = 0
        self.papers_recovered_through_replay = 0
        self.assessment_batches_submitted = {
            "primary": 0,
            "verification": 0,
        }

    def record(self, event: dict[str, Any]) -> None:
        safe = {field: event.get(field, "") for field in self.APPROVED_FIELDS}
        self.attempt_count += int(safe["event"] == "gemini_web_attempt")
        outcome = str(safe["outcome"] or "unknown")
        self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1
        action = str(safe["recovery_action"] or "")
        if action:
            self.recoveries[action] = self.recoveries.get(action, 0) + 1
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(safe, ensure_ascii=False) + "\n")

    def retry(self) -> None:
        self.retry_count += 1

    def assessment_attempt(
        self,
        *,
        stage: str,
        batch_id: str,
        subgroup_id: str,
        criterion_count: int,
        expected_criterion_object_count: int,
        paper_count: int,
        prompt_utf8_bytes: int,
        response_utf8_bytes: int,
        parsed_item_count: int,
        failure_class: str,
        over_budget: bool,
    ) -> None:
        self.assessment_batches_submitted[stage] += 1
        self.record({
            "event": "gemini_web_assessment_attempt",
            "stage": f"v24_{stage}",
            "outcome": "failed" if failure_class else "completed",
            "batch_id": batch_id,
            "subgroup_id": subgroup_id,
            "criterion_count": criterion_count,
            "expected_criterion_object_count": expected_criterion_object_count,
            "paper_count": paper_count,
            "prompt_utf8_bytes": prompt_utf8_bytes,
            "response_utf8_bytes": response_utf8_bytes,
            "parsed_item_count": parsed_item_count,
            "failure_class": failure_class,
            "over_budget": over_budget,
        })

    def fallback(self, reason: str) -> None:
        self.fallback_count += 1
        self.record({
            "event": "gemini_web_fallback",
            "outcome": "safe_maybe",
            "fallback_reason": reason,
        })

    def degraded_subgroup(
        self,
        *,
        stage: str,
        outcome: str,
        paper_ids: list[str],
        failure_class: str = "",
    ) -> None:
        self.record({
            "event": "gemini_web_degraded_subgroup",
            "stage": f"v24_{stage}",
            "outcome": outcome,
            "failure_class": failure_class,
            "paper_count": len(paper_ids),
            "paper_ids": paper_ids,
        })


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _fingerprint(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _protocol_cache_key(question: str, context: str, inclusion: str, exclusion: str) -> str:
    return _fingerprint({
        "version": GEMINI_WEB_V24_PROTOCOL_VERSION,
        "research_question": question,
        "research_context": context,
        "inclusion_criteria": inclusion,
        "exclusion_criteria": exclusion,
    })


def _assessment_cache_key(protocol_id: str, paper: V24Paper) -> str:
    return _fingerprint({
        "assessment_cache_version": GEMINI_WEB_V24_CACHE_VERSION,
        "assessment_prompt_version": GEMINI_WEB_V24_ASSESSMENT_PROMPT_VERSION,
        "protocol_id": protocol_id,
        "title": _normalized_text(paper.title),
        "abstract": _normalized_text(paper.abstract),
    })


def _contract_key(input_fingerprint: str, protocol_id: str) -> str:
    return _fingerprint({
        "architecture_version": GEMINI_WEB_V24_VERSION,
        "assessment_cache_version": GEMINI_WEB_V24_CACHE_VERSION,
        "assessment_prompt_version": GEMINI_WEB_V24_ASSESSMENT_PROMPT_VERSION,
        "input_fingerprint": input_fingerprint,
        "protocol_id": protocol_id,
        "adaptive_output_budget": {
            "max_batch_papers": MAX_BATCH_PAPERS,
            "max_estimated_output_bytes": MAX_ESTIMATED_OUTPUT_BYTES,
            "max_criterion_objects_per_batch": MAX_CRITERION_OBJECTS_PER_BATCH,
            "base_output_bytes_per_paper": BASE_OUTPUT_BYTES_PER_PAPER,
            "output_bytes_per_criterion": OUTPUT_BYTES_PER_CRITERION,
        },
    })


def _validate_protocol_sources(
    protocol: V24Protocol, inclusion_criteria: str, exclusion_criteria: str,
) -> None:
    inclusions = authoritative_criterion_entries(inclusion_criteria)
    exclusions = authoritative_criterion_entries(exclusion_criteria)
    if len(inclusions) + 1 > GEMINI_WEB_V24_MAX_CRITERIA_PER_KIND:
        raise ValueError(
            "explicit researcher inclusion criteria exceed protocol capacity "
            "after reserving the required composite relationship criterion"
        )
    if len(exclusions) > GEMINI_WEB_V24_MAX_CRITERIA_PER_KIND:
        raise ValueError("explicit researcher exclusion criteria exceed protocol capacity")

    expected = Counter(
        [("inclusion", item) for item in inclusions]
        + [("exclusion", item) for item in exclusions]
    )
    user_criteria = [
        criterion for criterion in protocol.criteria
        if criterion.source == "user"
    ]
    actual = Counter(
        (criterion.kind, _normalized_text(criterion.authoritative_text))
        for criterion in user_criteria
    )
    if actual != expected:
        raise ValueError(
            "compiled protocol omitted, merged, weakened, invented, or changed "
            "the polarity of one or more authoritative user criteria"
        )
    if any(not criterion.required for criterion in user_criteria):
        raise ValueError("authoritative user criteria must remain required")
    if any(criterion.is_composite_relationship for criterion in user_criteria):
        raise ValueError(
            "the research-question composite relationship must remain separate "
            "from authoritative user criteria"
        )
    if any(
        criterion.authoritative_text.strip()
        for criterion in protocol.criteria
        if criterion.source == "research_question"
    ):
        raise ValueError(
            "research-question criteria cannot claim authoritative user text"
        )

    composites = [
        criterion for criterion in protocol.required_inclusion_criteria
        if (
            criterion.source == "research_question"
            and criterion.required
            and criterion.is_composite_relationship
        )
    ]
    if not composites:
        raise ValueError(
            "compiled protocol omitted the required composite research relationship"
        )
    if any(
        criterion.is_composite_relationship
        for criterion in protocol.criteria
        if criterion not in composites
    ):
        raise ValueError(
            "composite relationship criteria must be required research-question inclusions"
        )


def _compile_protocol(
    browser,
    *,
    question: str,
    context: str,
    inclusion: str,
    exclusion: str,
) -> V24Protocol:
    schema = V24Protocol.model_json_schema()
    base = build_protocol_prompt(
        research_question=question,
        research_context=context,
        inclusion_criteria=inclusion,
        exclusion_criteria=exclusion,
        schema=schema,
    )
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            browser.set_attempt_context(stage="v24_protocol", retry_number=attempt)
            prompt = base if attempt == 0 else (
                base + "\n\nREPAIR: The previous protocol was structurally invalid. "
                "Return a complete corrected protocol JSON object only."
            )
            raw = browser.submit_prompt_and_get_response(prompt)
            value = parse_structured_model_output(raw, V24Protocol)
            protocol = V24Protocol.model_validate(value).model_copy(update={
                "research_question": question,
                "protocol_version": GEMINI_WEB_V24_PROTOCOL_VERSION,
            }).with_identity()
            _validate_protocol_sources(protocol, inclusion, exclusion)
            return protocol
        except (LocalAIOutputError, ValueError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if attempt == 0:
                browser.recover_transport_failure()
    raise RuntimeError(f"Gemini Web v2.4 could not compile a valid protocol: {last_error}")


def _load_protocol(
    root: Path, question: str, context: str, inclusion: str, exclusion: str,
) -> tuple[V24Protocol | None, Path]:
    path = root / "protocols" / f"{_protocol_cache_key(question, context, inclusion, exclusion)}.json"
    try:
        protocol = V24Protocol.model_validate_json(path.read_text(encoding="utf-8"))
        _validate_protocol_sources(protocol, inclusion, exclusion)
        return protocol, path
    except (OSError, ValueError):
        return None, path


def _execute_batch(
    browser,
    protocol: V24Protocol,
    papers: list[V24Paper],
    *,
    verification: bool,
    flags: dict[str, dict] | None,
    diagnostics: V24Diagnostics,
    max_attempts: int = 2,
    repair_only: bool = False,
    retry_offset: int = 0,
    batch_id: str = "",
    subgroup_id: str = "",
    over_budget: bool = False,
) -> tuple[dict[str, V24Assessment], str, str]:
    stage = "verification" if verification else "primary"
    compact_schema = _assessment_contract()["schema"]
    prompt = (
        build_verification_prompt(
            protocol=protocol.model_dump(mode="json"),
            papers=papers,
            flags=flags or {},
            schema=compact_schema,
        )
        if verification
        else build_primary_prompt(
            protocol=protocol.model_dump(mode="json"),
            papers=papers,
            schema=compact_schema,
        )
    )
    expected = {paper.paper_id for paper in papers}
    last_error: Exception | None = None
    transport_failure = False
    for attempt in range(max_attempts):
        request = prompt if attempt == 0 and not repair_only else (
            prompt + "\n\nREPAIR: Return the complete corrected batch JSON only. "
            "Do not omit papers or criterion assessments."
        )
        raw = ""
        parsed_item_count = 0
        attempt_failure_class = ""
        try:
            browser.set_attempt_context(
                stage=f"v24_{stage}",
                retry_number=retry_offset + attempt,
            )
            raw = browser.submit_prompt_and_get_response(request)
            compact = V24CompactAssessmentBatch.model_validate(
                parse_structured_model_output(raw, V24CompactAssessmentBatch)
            )
            parsed_item_count = len(compact.items)
            parsed = [item.expand() for item in compact.items]
            identifiers = [item.paper_id for item in parsed]
            if len(identifiers) != len(set(identifiers)) or set(identifiers) != expected:
                raise LocalAIOutputError("Gemini Web v2.4 returned incorrect or duplicate paper IDs")
            diagnostics.assessment_attempt(
                stage=stage,
                batch_id=batch_id,
                subgroup_id=subgroup_id,
                criterion_count=len(protocol.criteria),
                expected_criterion_object_count=(
                    len(papers) * _criterion_objects_per_item(stage, protocol)
                ),
                paper_count=len(papers),
                prompt_utf8_bytes=len(request.encode("utf-8")),
                response_utf8_bytes=len(raw.encode("utf-8")),
                parsed_item_count=parsed_item_count,
                failure_class="",
                over_budget=over_budget,
            )
            return {item.paper_id: item for item in parsed}, "", ""
        except (LocalAIOutputError, ValueError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            transport_failure = isinstance(exc, (TimeoutError, RuntimeError)) and not isinstance(
                exc, LocalAIOutputError
            )
            attempt_failure_class = (
                V24_TRANSPORT_FAILURE
                if transport_failure
                else V24_STRUCTURED_OUTPUT_FAILURE
            )
            diagnostics.assessment_attempt(
                stage=stage,
                batch_id=batch_id,
                subgroup_id=subgroup_id,
                criterion_count=len(protocol.criteria),
                expected_criterion_object_count=(
                    len(papers) * _criterion_objects_per_item(stage, protocol)
                ),
                paper_count=len(papers),
                prompt_utf8_bytes=len(request.encode("utf-8")),
                response_utf8_bytes=len(raw.encode("utf-8")),
                parsed_item_count=parsed_item_count,
                failure_class=attempt_failure_class,
                over_budget=over_budget,
            )
            diagnostics.retry()
            if attempt + 1 < max_attempts:
                if transport_failure:
                    browser.recover_transport_failure()
                else:
                    browser.start_new_job_chat()
    failure_class = (
        V24_TRANSPORT_FAILURE if transport_failure else V24_STRUCTURED_OUTPUT_FAILURE
    )
    failure_label = (
        "after one retry" if max_attempts > 1 else "during bounded degraded retry"
    )
    reason = f"Gemini Web v2.4 request failed {failure_label}: {last_error}"
    return {}, reason, failure_class


def _recover_failed_batch(browser, failure_class: str, action: str) -> None:
    browser.note_recovery(action)
    if failure_class == V24_TRANSPORT_FAILURE:
        browser.recover_transport_failure(exhausted=True)
    else:
        browser.start_new_job_chat()


def _execute_batch_with_degraded_retry(
    browser,
    protocol: V24Protocol,
    papers: list[V24Paper],
    *,
    verification: bool,
    flags: dict[str, dict] | None,
    diagnostics: V24Diagnostics,
    batch_id: str = "",
    over_budget: bool = False,
) -> tuple[dict[str, V24Assessment], dict[str, tuple[str, str]]]:
    assessed, reason, failure_class = _execute_batch(
        browser,
        protocol,
        papers,
        verification=verification,
        flags=flags,
        diagnostics=diagnostics,
        batch_id=batch_id,
        over_budget=over_budget,
    )
    if assessed:
        return assessed, {}
    if len(papers) <= 1:
        if failure_class == V24_TRANSPORT_FAILURE:
            browser.recover_transport_failure(exhausted=True)
        diagnostics.fallback(reason)
        return {}, {paper.paper_id: (reason, failure_class) for paper in papers}

    stage = "verification" if verification else "primary"
    recovery = (
        "transport_recovery"
        if failure_class == V24_TRANSPORT_FAILURE
        else "structured_clean_chat"
    )
    _recover_failed_batch(
        browser,
        failure_class,
        f"v24_{stage}_degraded_retry_{recovery}",
    )

    midpoint = len(papers) // 2
    subgroups = (papers[:midpoint], papers[midpoint:])
    merged: dict[str, V24Assessment] = {}
    failures: dict[str, tuple[str, str]] = {}
    for subgroup_index, subgroup in enumerate(subgroups):
        subgroup_ids = {paper.paper_id for paper in subgroup}
        ordered_subgroup_ids = [paper.paper_id for paper in subgroup]
        subgroup_flags = (
            {
                paper_id: value
                for paper_id, value in (flags or {}).items()
                if paper_id in subgroup_ids
            }
            if verification
            else None
        )
        subgroup_assessed, subgroup_reason, subgroup_failure_class = _execute_batch(
            browser,
            protocol,
            subgroup,
            verification=verification,
            flags=subgroup_flags,
            diagnostics=diagnostics,
            max_attempts=1,
            repair_only=True,
            retry_offset=2 + (subgroup_index * 2),
            batch_id=batch_id,
            subgroup_id=str(subgroup_index + 1),
            over_budget=over_budget,
        )
        if subgroup_assessed:
            merged.update(subgroup_assessed)
            continue

        if subgroup_failure_class == V24_TRANSPORT_FAILURE:
            diagnostics.degraded_subgroup(
                stage=stage,
                outcome="transport_failure",
                paper_ids=ordered_subgroup_ids,
                failure_class=subgroup_failure_class,
            )
            _recover_failed_batch(
                browser,
                subgroup_failure_class,
                f"v24_{stage}_degraded_subgroup_transport_replay_recovery",
            )
            diagnostics.degraded_subgroup(
                stage=stage,
                outcome="transport_recovery",
                paper_ids=ordered_subgroup_ids,
                failure_class=subgroup_failure_class,
            )
            diagnostics.degraded_subgroup_replay_count += 1
            replay_assessed, replay_reason, replay_failure_class = _execute_batch(
                browser,
                protocol,
                subgroup,
                verification=verification,
                flags=subgroup_flags,
                diagnostics=diagnostics,
                max_attempts=1,
                repair_only=True,
                retry_offset=3 + (subgroup_index * 2),
                batch_id=batch_id,
                subgroup_id=str(subgroup_index + 1),
                over_budget=over_budget,
            )
            if replay_assessed:
                merged.update(replay_assessed)
                diagnostics.degraded_subgroup_replay_success_count += 1
                diagnostics.papers_recovered_through_replay += len(subgroup)
                diagnostics.degraded_subgroup(
                    stage=stage,
                    outcome="transport_replay_succeeded",
                    paper_ids=ordered_subgroup_ids,
                )
                continue

            subgroup_reason = replay_reason
            subgroup_failure_class = replay_failure_class
            diagnostics.degraded_subgroup_replay_exhaustion_count += 1
            diagnostics.degraded_subgroup(
                stage=stage,
                outcome="transport_replay_exhausted",
                paper_ids=ordered_subgroup_ids,
                failure_class=subgroup_failure_class,
            )
            if subgroup_failure_class == V24_STRUCTURED_OUTPUT_FAILURE:
                diagnostics.degraded_subgroup(
                    stage=stage,
                    outcome="structured_output_terminal",
                    paper_ids=ordered_subgroup_ids,
                    failure_class=subgroup_failure_class,
                )
        elif subgroup_failure_class == V24_STRUCTURED_OUTPUT_FAILURE:
            diagnostics.degraded_subgroup(
                stage=stage,
                outcome="structured_output_terminal",
                paper_ids=ordered_subgroup_ids,
                failure_class=subgroup_failure_class,
            )

        diagnostics.fallback(subgroup_reason)
        failures.update({
            paper.paper_id: (subgroup_reason, subgroup_failure_class)
            for paper in subgroup
        })
        subgroup_recovery = (
            "transport_recovery"
            if subgroup_failure_class == V24_TRANSPORT_FAILURE
            else "structured_clean_chat"
        )
        _recover_failed_batch(
            browser,
            subgroup_failure_class,
            f"v24_{stage}_degraded_subgroup_{subgroup_recovery}",
        )
    return merged, failures


def _validate_and_decide(
    item: V24Assessment, protocol: V24Protocol, paper: V24Paper, *,
    stage: str = "primary",
) -> dict[str, Any]:
    units = evidence_lookup(paper.title, paper.abstract)
    criteria_by_id = {criterion.id: criterion for criterion in protocol.criteria}
    expected_ids = set(criteria_by_id)
    received_ids = [assessment.criterion_id for assessment in item.criterion_assessments]
    errors: list[str] = []
    if len(received_ids) != len(set(received_ids)) or set(received_ids) != expected_ids:
        errors.append("criterion assessments do not exactly match the immutable protocol")

    public_criteria: list[dict[str, Any]] = []
    evidence: list[dict[str, str]] = []
    for assessment in item.criterion_assessments:
        criterion = criteria_by_id.get(assessment.criterion_id)
        spans: list[dict[str, str]] = []
        for reference in assessment.evidence:
            unit = units.get(reference.evidence_id)
            if unit is None or unit.get("source") != reference.source:
                errors.append(f"invalid evidence reference for {assessment.criterion_id}")
                continue
            span = {
                "source": str(unit["source"]),
                "evidence_id": str(unit["evidence_id"]),
                "quote": str(unit["text"]),
            }
            spans.append(span)
            evidence.append({"criterion_id": assessment.criterion_id, **span})
        if criterion is not None:
            if assessment.verdict == "MET" and (
                assessment.scope_support != "SUBSTANTIVE"
                or assessment.evidence_relationship != "SUPPORTS"
                or not spans
            ):
                errors.append(f"MET lacks substantive supporting evidence: {criterion.id}")
            if assessment.verdict == "NOT_MET" and (
                assessment.scope_support != "SUBSTANTIVE"
                or assessment.evidence_relationship != "CONFLICTS"
                or not spans
            ):
                errors.append(f"NOT_MET lacks affirmative conflicting evidence: {criterion.id}")
            if assessment.verdict == "UNCLEAR" and assessment.evidence_relationship not in {
                "INCIDENTAL", "INSUFFICIENT"
            }:
                errors.append(f"UNCLEAR has an incompatible evidence relationship: {criterion.id}")
            if assessment.scope_support in {"INCIDENTAL", "INSUFFICIENT"} and assessment.verdict != "UNCLEAR":
                errors.append(f"non-substantive support must be UNCLEAR: {criterion.id}")
        public_criteria.append({
            **assessment.model_dump(mode="json"),
            "evidence": spans,
        })

    verdicts = {
        assessment.criterion_id: assessment.verdict
        for assessment in item.criterion_assessments
        if assessment.criterion_id in criteria_by_id
    }
    required = protocol.required_inclusion_criteria
    exclusions = protocol.exclusion_boundaries
    if errors:
        decision = "MAYBE"
    elif any(verdicts.get(criterion.id) == "NOT_MET" for criterion in required):
        decision = "REJECT"
    elif any(verdicts.get(criterion.id) == "MET" for criterion in exclusions):
        decision = "REJECT"
    elif (
        all(verdicts.get(criterion.id) == "MET" for criterion in required)
        and not any(verdicts.get(criterion.id) == "MET" for criterion in exclusions)
        and not any(
            criterion.source == "user"
            and verdicts.get(criterion.id) == "UNCLEAR"
            for criterion in exclusions
        )
    ):
        decision = "KEEP"
    else:
        decision = "MAYBE"

    contradiction = item.decision != decision
    if contradiction:
        errors.append(
            f"model decision {item.decision} conflicts with deterministic decision {decision}"
        )
    validation_status = "validated" if not errors else "unresolved"
    risk = item.decision_risk
    if validation_status != "validated":
        risk = "HIGH"
    elif decision == "MAYBE" and risk != "HIGH":
        risk = "BORDERLINE"
    result = {
        "decision": decision,
        "confidence": round(item.confidence, 2),
        "decision_risk": risk,
        "reason": item.reason,
        "criteria": public_criteria,
        "evidence": evidence,
        "validation_status": validation_status,
        "validation_errors": errors,
        "model_decision": item.decision,
        "route_used": "primary",
        "verification_status": "not_required",
        "fallback_reason": "",
        "failure_class": "",
    }
    result["assessment_trace"] = [{
        "stage": stage,
        "model_decision": item.decision,
        "deterministic_decision": decision,
        "validation_status": validation_status,
        "validation_errors": list(errors),
        "criteria": public_criteria,
    }]
    return result


def _safe_maybe(
    protocol: V24Protocol,
    reason: str,
    *,
    route: str,
    verification_status: str,
    failure_class: str = "",
) -> dict[str, Any]:
    return {
        "decision": "MAYBE",
        "confidence": 0.0,
        "decision_risk": "HIGH",
        "reason": reason[:600],
        "criteria": [{
            "criterion_id": criterion.id,
            "verdict": "UNCLEAR",
            "scope_support": "INSUFFICIENT",
            "evidence_relationship": "INSUFFICIENT",
            "rationale": "Assessment could not be validated.",
            "evidence": [],
        } for criterion in protocol.criteria],
        "evidence": [],
        "validation_status": "validated",
        "validation_errors": [],
        "model_decision": "MAYBE",
        "route_used": route,
        "verification_status": verification_status,
        "fallback_reason": reason,
        "failure_class": failure_class,
        "assessment_trace": [{
            "stage": route,
            "model_decision": "MAYBE",
            "deterministic_decision": "MAYBE",
            "validation_status": "validated_safe_fallback",
            "validation_errors": [],
        }],
    }


def _verification_route(result: dict[str, Any], protocol: V24Protocol) -> str:
    if result["validation_status"] != "validated":
        return "validation_tension"

    criteria_by_id = {
        criterion.id: criterion
        for criterion in protocol.criteria
    }

    if result["decision"] == "MAYBE":
        substantive_unresolved = any(
            item.get("verdict") == "UNCLEAR"
            and item.get("scope_support") == "SUBSTANTIVE"
            and bool(item.get("evidence"))
            and (
                criteria_by_id.get(str(item.get("criterion_id") or "")) is not None
                and (
                    criteria_by_id[str(item.get("criterion_id") or "")].kind
                    == "inclusion"
                    or criteria_by_id[str(item.get("criterion_id") or "")].source
                    == "user"
                )
            )
            for item in result["criteria"]
        )

        if substantive_unresolved and result["decision_risk"] == "HIGH":
            return "borderline_primary"

        return ""

    if result["decision_risk"] != "LOW" or result["confidence"] < 0.8:
        return "risky_definitive"

    return ""


def _cacheable(result: dict[str, Any]) -> bool:
    return (
        result.get("validation_status") == "validated"
        and not result.get("fallback_reason")
        and result.get("verification_status") not in {"failed", "disagreed", "uncertain"}
    )


def _load_assessment_cache(root: Path, protocol_id: str, paper: V24Paper) -> dict[str, Any] | None:
    path = root / "assessments" / f"{_assessment_cache_key(protocol_id, paper)}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("cache_version") != GEMINI_WEB_V24_CACHE_VERSION
            or value.get("assessment_prompt_version")
            != GEMINI_WEB_V24_ASSESSMENT_PROMPT_VERSION
            or value.get("protocol_id") != protocol_id
            or not _cacheable(value.get("result", {}))
        ):
            return None
        result = dict(value["result"])
        result["cache_hit"] = True
        result["runtime_seconds"] = 0.0
        result["route_used"] = "validated_cache"
        return result
    except (OSError, ValueError, TypeError):
        return None


def _save_assessment_cache(
    root: Path, protocol_id: str, paper: V24Paper, result: dict[str, Any],
) -> None:
    if not _cacheable(result):
        return
    path = root / "assessments" / f"{_assessment_cache_key(protocol_id, paper)}.json"
    cached = dict(result)
    cached["cache_hit"] = False
    _atomic_json(path, {
        "cache_version": GEMINI_WEB_V24_CACHE_VERSION,
        "assessment_prompt_version": GEMINI_WEB_V24_ASSESSMENT_PROMPT_VERSION,
        "protocol_id": protocol_id,
        "result": cached,
    })


def _criterion_ids(
    result: dict[str, Any], protocol: V24Protocol, verdict: str,
) -> str:
    required_ids = {criterion.id for criterion in protocol.required_inclusion_criteria}
    return json.dumps([
        item["criterion_id"] for item in result["criteria"]
        if item.get("criterion_id") in required_ids and item.get("verdict") == verdict
    ])


def _row(
    source: dict[str, Any],
    source_index: Any,
    paper: V24Paper,
    protocol: V24Protocol,
    result: dict[str, Any],
    *,
    execution_origin: str,
    direct_handling_reason: str = "",
) -> dict[str, Any]:
    row = dict(source)
    evidence_summary = " | ".join(
        f'{item["criterion_id"]}: "{item["quote"]}"' for item in result["evidence"]
    )[:2000]
    row.update({
        "Title": paper.title,
        "Abstract": paper.abstract,
        "Decision": result["decision"],
        "Confidence": result["confidence"],
        "Decision_Risk": result["decision_risk"],
        "Reason": result["reason"],
        "Required_Criteria_Met": _criterion_ids(result, protocol, "MET"),
        "Required_Criteria_Not_Met": _criterion_ids(result, protocol, "NOT_MET"),
        "Required_Criteria_Unclear": _criterion_ids(result, protocol, "UNCLEAR"),
        "Evidence_Summary": evidence_summary,
        "Evidence_JSON": json.dumps(result["evidence"], ensure_ascii=False),
        "Criteria_JSON": json.dumps(result["criteria"], ensure_ascii=False),
        "Layer_Trace_JSON": json.dumps(
            result.get("assessment_trace", []),
            ensure_ascii=False,
        ),
        "Uncertainty_JSON": json.dumps(
            result["validation_errors"] if result["decision"] == "MAYBE" else [],
            ensure_ascii=False,
        ),
        "Contradictions_JSON": json.dumps(
            [
                error for error in result["validation_errors"]
                if error.startswith("model decision ")
            ],
            ensure_ascii=False,
        ),
        "Route_Used": result["route_used"],
        "Critic_Route": (
            "" if result["verification_status"] == "not_required"
            else result["route_used"]
        ),
        "Verification_Status": result["verification_status"],
        "Validation_Status": result["validation_status"],
        "Validation_Errors": json.dumps(result["validation_errors"], ensure_ascii=False),
        "Fallback_Reason": result["fallback_reason"],
        "Failure_Class": result["failure_class"],
        "Protocol_ID": protocol.protocol_id,
        "Prompt_Version": GEMINI_WEB_V24_ASSESSMENT_PROMPT_VERSION,
        "Schema_Version": SCHEMA_VERSION,
        "Model": "gemini-web",
        "Model_Decision": result.get("model_decision", ""),
        "Model_Tier": "gemini_web_v24",
        "Escalated": result["verification_status"] != "not_required",
        "Cache_Hit": bool(result.get("cache_hit", False)),
        "Runtime_Seconds": result.get("runtime_seconds", 0.0),
        "Processing_Seconds": result.get("runtime_seconds", 0.0),
        "Original_Processing_Seconds": result.get("runtime_seconds", 0.0),
        "Source_Row_Index": str(source_index),
        "Execution_Origin": execution_origin,
        "Direct_Handling_Reason": direct_handling_reason,
    })
    return row


def _resume_rows(path: Path, protocol_id: str, expected: set[str]) -> dict[str, dict[str, Any]]:
    try:
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    except (OSError, ValueError):
        return {}
    required = {
        "Source_Row_Index", "Protocol_ID", "Prompt_Version", "Decision",
        "Validation_Status", "Verification_Status", "Criteria_JSON", "Evidence_JSON",
    }
    if not required.issubset(frame.columns):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for _, record in frame.iterrows():
        row = record.to_dict()
        key = str(row["Source_Row_Index"])
        if (
            key not in expected
            or str(row["Protocol_ID"]) != protocol_id
            or str(row["Prompt_Version"]) != GEMINI_WEB_V24_ASSESSMENT_PROMPT_VERSION
            or str(row.get("Failure_Class") or "") in {
                V24_STRUCTURED_OUTPUT_FAILURE,
                V24_TRANSPORT_FAILURE,
                V24_VERIFICATION_FAILURE,
            }
            or str(row.get("Verification_Status") or "") in {"pending", "failed"}
        ):
            continue
        row["Execution_Origin"] = "resumed"
        row["Direct_Handling_Reason"] = ""
        rows[key] = row
    return rows


def screen_csv_with_gemini_web_v24(
    *,
    frame: pd.DataFrame,
    valid: pd.DataFrame,
    title_col: str,
    abstract_col: str,
    research_question: str,
    research_context: str,
    inclusion_criteria: str,
    exclusion_criteria: str,
    output_path: str,
    job_id: str,
    input_fingerprint: str,
    resume: bool,
    limit: int,
    progress,
    screening_session,
    source_dataset_fingerprint: str = "",
    browser_factory: Callable[[GeminiWebV24Config], Any] = GeminiWebV24Automation,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_root = output.parent.parent / "cache" / "gemini_web_v24"
    diagnostics = V24Diagnostics(cache_root / "diagnostics" / f"{job_id}.jsonl")
    protocol, protocol_path = _load_protocol(
        cache_root,
        research_question,
        research_context,
        inclusion_criteria,
        exclusion_criteria,
    )

    papers: dict[str, V24Paper] = {}
    sources: dict[str, tuple[Any, dict[str, Any]]] = {}
    for source_index, source_row in valid.iterrows():
        key = str(source_index)
        paper = V24Paper(
            paper_id=key,
            title="" if pd.isna(source_row[title_col]) else str(source_row[title_col]),
            abstract="" if pd.isna(source_row[abstract_col]) else str(source_row[abstract_col]),
        )
        papers[key] = paper
        sources[key] = (source_index, source_row.to_dict())

    browser_context = None
    browser = None
    if protocol is None:
        progress.begin_batches(job_id, "gemini_web_v24_protocol", 1, 1, 1)
        browser_context = browser_factory(GeminiWebV24Config(diagnostic_sink=diagnostics.record))
        try:
            browser = browser_context.__enter__()
            protocol = _compile_protocol(
                browser,
                question=research_question,
                context=research_context,
                inclusion=inclusion_criteria,
                exclusion=exclusion_criteria,
            )
            _atomic_json(protocol_path, protocol.model_dump(mode="json"))
            progress.update_batch(job_id, 1, 1)
            browser.note_recovery("v24_protocol_to_primary_clean_chat")
            browser.start_new_job_chat()
        except Exception:
            browser_context.__exit__(None, None, None)
            browser_context = None
            browser = None
            raise

    checkpoint = cache_root / "checkpoints" / f"{_contract_key(input_fingerprint, protocol.protocol_id)}.csv"
    rows = _resume_rows(checkpoint, protocol.protocol_id, set(papers)) if resume else {}
    progress.set_resumed_count(job_id, len(rows))
    pending: list[V24Paper] = []
    for key, paper in papers.items():
        if key in rows:
            continue
        if not paper.abstract.strip():
            result = _safe_maybe(
                protocol,
                "The abstract is missing, so title-only evidence cannot safely establish every required relationship.",
                route="missing_abstract",
                verification_status="not_required",
            )
            source_index, source = sources[key]
            rows[key] = _row(
                source,
                source_index,
                paper,
                protocol,
                result,
                execution_origin="directly_handled_without_primary",
                direct_handling_reason="missing_abstract",
            )
            continue
        cached = _load_assessment_cache(cache_root, protocol.protocol_id, paper)
        if cached is None:
            pending.append(paper)
            continue
        source_index, source = sources[key]
        rows[key] = _row(
            source,
            source_index,
            paper,
            protocol,
            cached,
            execution_origin="assessment_cache_hit",
        )

    try:
        if pending and browser is None:
            browser_context = browser_factory(GeminiWebV24Config(diagnostic_sink=diagnostics.record))
            browser = browser_context.__enter__()

        primary_batch_size, primary_over_budget = plan_assessment_batch_size(
            protocol, stage="primary"
        )
        primary_batches = (
            len(pending) + primary_batch_size - 1
        ) // primary_batch_size
        progress.begin_batches(
            job_id, "gemini_web_v24_primary", len(pending), primary_batches,
            primary_batch_size,
        )
        verification_keys: list[str] = []
        route_by_key: dict[str, str] = {}
        for batch_number in range(primary_batches):
            batch = pending[
                batch_number * primary_batch_size:
                (batch_number + 1) * primary_batch_size
            ]
            started = time.perf_counter()
            assessed, failures = _execute_batch_with_degraded_retry(
                browser, protocol, batch, verification=False, flags=None,
                diagnostics=diagnostics,
                batch_id=f"primary-{batch_number + 1:04d}",
                over_budget=primary_over_budget,
            )
            elapsed = (time.perf_counter() - started) / max(1, len(batch))
            for paper in batch:
                if paper.paper_id not in assessed:
                    failure, failure_class = failures[paper.paper_id]
                    result = _safe_maybe(
                        protocol,
                        failure,
                        route="technical_failure",
                        verification_status="failed",
                        failure_class=failure_class,
                    )
                else:
                    result = _validate_and_decide(assessed[paper.paper_id], protocol, paper)
                    result["runtime_seconds"] = round(elapsed, 4)
                    result["cache_hit"] = False
                    route = _verification_route(result, protocol)
                    route_by_key[paper.paper_id] = route
                    if route:
                        result["route_used"] = route
                        result["verification_status"] = "pending"
                        verification_keys.append(paper.paper_id)
                    else:
                        _save_assessment_cache(cache_root, protocol.protocol_id, paper, result)
                source_index, source = sources[paper.paper_id]
                rows[paper.paper_id] = _row(
                    source,
                    source_index,
                    paper,
                    protocol,
                    result,
                    execution_origin="fresh_primary",
                )
            ordered = [rows[key] for key in papers if key in rows]
            _atomic_csv(checkpoint, ordered)
            _atomic_csv(output, ordered)
            counts = screening_session.counts(ordered)
            progress.update_batch(
                job_id, batch_number + 1,
                min(len(pending), (batch_number + 1) * primary_batch_size),
            )
            progress.update_counts(
                job_id, len(ordered), counts["keep"], counts["maybe"], counts["reject"],
            )

        verification_batch_size, verification_over_budget = plan_assessment_batch_size(
            protocol, stage="verification"
        )
        verification_batches = (
            len(verification_keys) + verification_batch_size - 1
        ) // verification_batch_size
        if verification_keys:
            browser.note_recovery("v24_primary_to_verification_clean_chat")
            browser.start_new_job_chat()
        progress.begin_batches(
            job_id, "gemini_web_v24_verification", len(verification_keys),
            verification_batches, verification_batch_size,
        )
        for batch_number in range(verification_batches):
            keys = verification_keys[
                batch_number * verification_batch_size:
                (batch_number + 1) * verification_batch_size
            ]
            batch = [papers[key] for key in keys]
            flags = {}
            for key in keys:
                errors = json.loads(str(rows[key].get("Validation_Errors") or "[]"))
                criteria = json.loads(str(rows[key].get("Criteria_JSON") or "[]"))
                flags[key] = {
                    "validation_errors": errors,
                    "unresolved_criterion_ids": [
                        item.get("criterion_id") for item in criteria
                        if item.get("verdict") == "UNCLEAR"
                    ],
                }
            started = time.perf_counter()
            assessed, failures = _execute_batch_with_degraded_retry(
                browser, protocol, batch, verification=True, flags=flags,
                diagnostics=diagnostics,
                batch_id=f"verification-{batch_number + 1:04d}",
                over_budget=verification_over_budget,
            )
            elapsed = (time.perf_counter() - started) / max(1, len(batch))
            for paper in batch:
                primary = rows[paper.paper_id]
                primary_decision = str(primary["Decision"])
                primary_trace = json.loads(
                    str(primary.get("Layer_Trace_JSON") or "[]")
                )
                if paper.paper_id not in assessed:
                    failure, failure_class = failures[paper.paper_id]
                    result = _safe_maybe(
                        protocol,
                        "Independent verification was unavailable; the provisional decision was not retained.",
                        route=route_by_key[paper.paper_id],
                        verification_status="failed",
                        failure_class=failure_class,
                    )
                    result["fallback_reason"] = failure
                    result["assessment_trace"] = [
                        *primary_trace,
                        {
                            "stage": "verification_resolution",
                            "deterministic_decision": "MAYBE",
                            "validation_status": "verification_unavailable",
                            "validation_errors": [],
                        },
                    ]
                else:
                    verified = _validate_and_decide(
                        assessed[paper.paper_id], protocol, paper,
                        stage="verification",
                    )
                    verified["runtime_seconds"] = round(
                        float(primary.get("Runtime_Seconds") or 0) + elapsed, 4
                    )
                    verified["cache_hit"] = False
                    verified["route_used"] = route_by_key[paper.paper_id]
                    if (
                        verified["validation_status"] == "validated"
                        and verified["decision"] == primary_decision
                    ):
                        verified["verification_status"] = "agreed"
                        verified["assessment_trace"] = [
                            *primary_trace, *verified["assessment_trace"],
                        ]
                        result = verified
                    else:
                        status = (
                            "failed" if verified["validation_status"] != "validated"
                            else "disagreed"
                        )
                        result = _safe_maybe(
                            protocol,
                            "Independent evidence-first assessments did not agree on a validated decision.",
                            route=route_by_key[paper.paper_id],
                            verification_status=status,
                        )
                        result["runtime_seconds"] = verified["runtime_seconds"]
                        result["cache_hit"] = False
                        result["assessment_trace"] = [
                            *primary_trace,
                            *verified["assessment_trace"],
                            {
                                "stage": "verification_resolution",
                                "deterministic_decision": "MAYBE",
                                "validation_status": status,
                                "validation_errors": [],
                            },
                        ]
                if _cacheable(result):
                    _save_assessment_cache(cache_root, protocol.protocol_id, paper, result)
                source_index, source = sources[paper.paper_id]
                rows[paper.paper_id] = _row(
                    source,
                    source_index,
                    paper,
                    protocol,
                    result,
                    execution_origin="fresh_primary",
                )
            ordered = [rows[key] for key in papers if key in rows]
            _atomic_csv(checkpoint, ordered)
            _atomic_csv(output, ordered)
            counts = screening_session.counts(ordered)
            progress.update_batch(
                job_id, batch_number + 1,
                min(len(verification_keys), (batch_number + 1) * verification_batch_size),
            )
            progress.update_counts(
                job_id, len(ordered), counts["keep"], counts["maybe"], counts["reject"],
            )

        ordered = [rows[key] for key in papers]
        _atomic_csv(checkpoint, ordered)
        _atomic_csv(output, ordered)
        screening_session.set_results(
            ordered,
            job_id=job_id,
            output_path=output_path,
            architecture_version=GEMINI_WEB_V24_VERSION,
        )
        counts = screening_session.counts(ordered)
        runtime = round(time.perf_counter() - run_started, 4)
        route_counts: dict[str, int] = {}
        verification_outcomes: dict[str, int] = {}
        for row in ordered:
            route = str(row.get("Route_Used") or "")
            verification = str(row.get("Verification_Status") or "not_required")
            route_counts[route] = route_counts.get(route, 0) + 1
            verification_outcomes[verification] = verification_outcomes.get(verification, 0) + 1
        origin_ids = {
            origin: [
                str(row["Source_Row_Index"])
                for row in ordered
                if str(row.get("Execution_Origin") or "") == origin
            ]
            for origin in (
                "resumed",
                "assessment_cache_hit",
                "fresh_primary",
                "directly_handled_without_primary",
            )
        }
        direct_handling_reasons = {
            str(row["Source_Row_Index"]): str(row.get("Direct_Handling_Reason") or "")
            for row in ordered
            if str(row.get("Execution_Origin") or "")
            == "directly_handled_without_primary"
        }
        missing_abstract_source_row_ids = [
            str(row["Source_Row_Index"])
            for row in ordered
            if not str(row.get("Abstract") or "").strip()
        ]
        summary = {
            "job_id": job_id,
            "run_status": "complete",
            "runtime_seconds": runtime,
            "papers_per_minute": round(len(ordered) * 60 / runtime, 2) if runtime else 0,
            "attempt_count": diagnostics.attempt_count,
            "retry_count": diagnostics.retry_count,
            "timeout_fallback_count": sum(
                str(row.get("Failure_Class") or "") in {
                    V24_TRANSPORT_FAILURE, V24_VERIFICATION_FAILURE,
                }
                for row in ordered
            ),
            "structured_output_fallback_count": sum(
                str(row.get("Failure_Class") or "") == V24_STRUCTURED_OUTPUT_FAILURE
                for row in ordered
            ),
            "technical_fallback_count": sum(
                str(row.get("Failure_Class") or "") in {
                    V24_STRUCTURED_OUTPUT_FAILURE,
                    V24_TRANSPORT_FAILURE,
                    V24_VERIFICATION_FAILURE,
                }
                for row in ordered
            ),
            "cache_hit_count": sum(bool(row.get("Cache_Hit")) for row in ordered),
            "assessment_cache_hits_loaded": len(origin_ids["assessment_cache_hit"]),
            "assessment_cache_hit_source_row_ids": origin_ids["assessment_cache_hit"],
            "resumed_count": len(origin_ids["resumed"]),
            "resumed_source_row_ids": origin_ids["resumed"],
            "fresh_primary_papers": len(origin_ids["fresh_primary"]),
            "fresh_primary_source_row_ids": origin_ids["fresh_primary"],
            "directly_handled_without_primary_count": len(
                origin_ids["directly_handled_without_primary"]
            ),
            "directly_handled_without_primary_source_row_ids": (
                origin_ids["directly_handled_without_primary"]
            ),
            "direct_handling_reasons": direct_handling_reasons,
            "missing_abstract_count": len(missing_abstract_source_row_ids),
            "missing_abstract_source_row_ids": missing_abstract_source_row_ids,
            "run_selected_count": len(papers),
            "run_selected_source_row_ids": list(papers),
            "verification_count": sum(
                str(row.get("Verification_Status")) != "not_required" for row in ordered
            ),
            "primary_batches_submitted": diagnostics.assessment_batches_submitted["primary"],
            "primary_papers_requested": len(origin_ids["fresh_primary"]),
            "primary_structured_failures": sum(
                str(row.get("Route_Used") or "") == "technical_failure"
                and str(row.get("Failure_Class") or "") == V24_STRUCTURED_OUTPUT_FAILURE
                for row in ordered
            ),
            "primary_technical_fallbacks": sum(
                str(row.get("Route_Used") or "") == "technical_failure"
                for row in ordered
            ),
            "verification_batches_submitted": (
                diagnostics.assessment_batches_submitted["verification"]
            ),
            "verification_papers_requested": len(verification_keys),
            "verification_structured_failures": sum(
                str(row.get("Route_Used") or "") != "technical_failure"
                and str(row.get("Verification_Status") or "") == "failed"
                and str(row.get("Failure_Class") or "") == V24_STRUCTURED_OUTPUT_FAILURE
                for row in ordered
            ),
            "verification_semantic_validation_failures": sum(
                str(row.get("Route_Used") or "") != "technical_failure"
                and str(row.get("Verification_Status") or "") == "failed"
                and not str(row.get("Failure_Class") or "")
                for row in ordered
            ),
            "verification_validated_agreements": sum(
                str(row.get("Verification_Status") or "") == "agreed"
                for row in ordered
            ),
            "verification_validated_disagreements": sum(
                str(row.get("Verification_Status") or "") == "disagreed"
                for row in ordered
            ),
            "summary_compatibility_aliases": {
                "verification_count": (
                    "legacy row count where Verification_Status is not not_required; "
                    "use verification_papers_requested for actual verifier inputs"
                ),
                "structured_output_fallback_count": (
                    "legacy final-row total across stages"
                ),
            },
            "route_counts": route_counts,
            "verification_outcomes": verification_outcomes,
            "detector_outcomes": diagnostics.outcomes,
            "recovery_actions": diagnostics.recoveries,
            "degraded_subgroup_replay_count": diagnostics.degraded_subgroup_replay_count,
            "degraded_subgroup_replay_success_count": (
                diagnostics.degraded_subgroup_replay_success_count
            ),
            "degraded_subgroup_replay_exhaustion_count": (
                diagnostics.degraded_subgroup_replay_exhaustion_count
            ),
            "papers_recovered_through_replay": (
                diagnostics.papers_recovered_through_replay
            ),
            "diagnostics_path": str(diagnostics.path),
            "source_dataset_fingerprint": (
                source_dataset_fingerprint
                or source_dataframe_fingerprint(frame)
            ),
            "screening_input_fingerprint": input_fingerprint,
            "screening_output_fingerprint": screening_output_fingerprint(ordered),
            "architecture_version": GEMINI_WEB_V24_VERSION,
            "protocol_id": protocol.protocol_id,
            "protocol_cache_version": GEMINI_WEB_V24_PROTOCOL_VERSION,
            "assessment_cache_version": GEMINI_WEB_V24_CACHE_VERSION,
            "assessment_prompt_version": GEMINI_WEB_V24_ASSESSMENT_PROMPT_VERSION,
            "primary_batch_size": primary_batch_size,
            "verification_batch_size": verification_batch_size,
            "primary_batch_over_budget": primary_over_budget,
            "verification_batch_over_budget": verification_over_budget,
        }
        _atomic_json(diagnostics.path.with_suffix(".summary.json"), summary)
        progress.finish(job_id)
        return {
            **counts,
            **summary,
            "parse_error": 0,
            "output_file": output_path,
            "total_papers": len(ordered),
            "input_total_rows": len(frame),
            "screened_total_rows": len(ordered),
            "row_limit_applied": bool(limit),
            "row_limit_value": limit or "",
            "screening_engine": GEMINI_WEB_V24_ENGINE,
            "architecture_version": GEMINI_WEB_V24_VERSION,
            "resumed_count": summary["resumed_count"],
            "schema_version": SCHEMA_VERSION,
            "protocol_id": protocol.protocol_id,
            "model_tier": "gemini_web_v24",
            "resource_profile": "web",
            "fast_model": "gemini-web",
            "strong_model": "gemini-web",
            "escalated_count": summary["verification_count"],
        }
    finally:
        if browser_context is not None:
            browser_context.__exit__(None, None, None)
