from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

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


RUNTIME_CATEGORY_NAMES = (
    "total_job",
    "protocol_cache_load",
    "protocol_compilation",
    "protocol_cache_write",
    "checkpoint_load",
    "assessment_cache_load",
    "primary_stage_total",
    "verification_stage_total",
    "direct_handling_total",
    "parsing",
    "result_validation",
    "assessment_cache_write",
    "checkpoint_write",
    "output_csv_write",
    "screening_session_update",
    "progress_update",
    "prisma_generation",
    "diagnostics_summary_write",
    "local_processing_total",
    "gemini_browser_total",
)
RUNTIME_CALL_NAMES = (
    "protocol",
    "primary",
    "verification",
    "structured_repair",
    "bounded_retry",
    "degraded_subgroup_attempt",
    "subgroup_transport_replay",
)
RUNTIME_BROWSER_NAMES = (
    "preparation_before_submission",
    "prompt_box_discovery_reload_readiness",
    "prompt_fill_and_submit",
    "response_wait",
    "final_response_capture_diagnostics",
    "browser_recovery",
    "new_chat_creation",
    "browser_context_recycle",
    "recovery_backoff_sleep",
)
RUNTIME_DIMENSIONS = (
    "stage",
    "attempt_type",
    "outcome",
    "paper_count",
    "batch_id",
    "subgroup_id",
    "retry_number",
)


class V24RuntimeMetrics:
    """Best-effort runtime observations; never owns screening error handling."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self.clock = clock
        self.observations: list[dict[str, Any]] = []
        self.instrumentation_errors: list[str] = []

    def _error(self, exc: Exception) -> None:
        try:
            self.instrumentation_errors.append(
                f"{type(exc).__name__}: {str(exc)}"[:300]
            )
        except Exception:
            pass

    def now(self) -> float | None:
        try:
            return float(self.clock())
        except Exception as exc:
            self._error(exc)
            return None

    def record(
        self,
        metric: str,
        duration_seconds: float,
        *,
        family: str = "categories",
        success: bool = True,
        **metadata: Any,
    ) -> None:
        try:
            duration = max(0.0, float(duration_seconds))
            self.observations.append({
                "metric": str(metric),
                "family": str(family),
                "duration_seconds": duration,
                "success": bool(success),
                **{
                    key: metadata.get(key)
                    for key in RUNTIME_DIMENSIONS
                    if metadata.get(key) not in (None, "")
                },
            })
        except Exception as exc:
            self._error(exc)

    @contextmanager
    def observe(
        self,
        metric: str,
        *,
        family: str = "categories",
        **metadata: Any,
    ) -> Iterator[None]:
        started = self.now()
        try:
            yield
        except Exception:
            ended = self.now()
            if started is not None and ended is not None:
                self.record(
                    metric, ended - started, family=family, success=False, **metadata
                )
            raise
        else:
            ended = self.now()
            if started is not None and ended is not None:
                self.record(
                    metric, ended - started, family=family, success=True, **metadata
                )

    @staticmethod
    def _aggregate(observations: list[dict[str, Any]]) -> dict[str, Any]:
        durations = sorted(float(item["duration_seconds"]) for item in observations)
        count = len(durations)
        if not count:
            return {
                "count": 0,
                "total_seconds": 0.0,
                "mean_seconds": 0.0,
                "p50_seconds": 0.0,
                "p95_seconds": 0.0,
                "min_seconds": 0.0,
                "max_seconds": 0.0,
                "success_count": 0,
                "failure_count": 0,
            }

        def percentile(fraction: float) -> float:
            rank = max(1, min(count, math.ceil(fraction * count)))
            return durations[rank - 1]

        total = sum(durations)
        rounded = lambda value: round(float(value), 6)
        return {
            "count": count,
            "total_seconds": rounded(total),
            "mean_seconds": rounded(total / count),
            "p50_seconds": rounded(percentile(0.50)),
            "p95_seconds": rounded(percentile(0.95)),
            "min_seconds": rounded(durations[0]),
            "max_seconds": rounded(durations[-1]),
            "success_count": sum(bool(item.get("success")) for item in observations),
            "failure_count": sum(not bool(item.get("success")) for item in observations),
        }

    def _family(
        self, family: str, names: tuple[str, ...] = (),
    ) -> dict[str, dict[str, Any]]:
        observed_names = sorted({
            str(item["metric"])
            for item in self.observations
            if item.get("family") == family
        })
        return {
            name: self._aggregate([
                item for item in self.observations
                if item.get("family") == family and item.get("metric") == name
            ])
            for name in dict.fromkeys((*names, *observed_names))
        }

    def serialize(
        self,
        *,
        selected_papers: int = 0,
        fresh_primary_papers: int = 0,
        submitted_batches: int = 0,
    ) -> dict[str, Any]:
        categories = self._family("categories", RUNTIME_CATEGORY_NAMES)
        total = categories["total_job"]["total_seconds"]
        browser = categories["gemini_browser_total"]["total_seconds"]
        local = max(0.0, total - browser)
        categories["local_processing_total"] = self._aggregate([{
            "duration_seconds": local,
            "success": True,
        }])
        dimensions: dict[str, dict[str, Any]] = {}
        call_observations = [
            item for item in self.observations if item.get("family") == "gemini_calls"
        ]
        for dimension in RUNTIME_DIMENSIONS:
            groups: dict[str, Any] = {}
            values = sorted({
                str(item[dimension])
                for item in call_observations
                if dimension in item
            })
            for value in values:
                groups[value] = self._aggregate([
                    item for item in call_observations
                    if str(item.get(dimension, "")) == value
                ])
            dimensions[f"by_{dimension}"] = groups
        percentage = lambda value: round(value * 100 / total, 6) if total else 0.0
        per = lambda value, count: round(value / count, 6) if count else 0.0
        return {
            "schema_version": "gemini-web-v2.4-runtime-v1",
            "status": "complete" if not self.instrumentation_errors else "partial",
            "clock": "perf_counter",
            "rounding_decimal_places": 6,
            "percentile_method": "nearest_rank",
            "total_job_seconds": total,
            "categories": categories,
            "gemini_calls": self._family("gemini_calls", RUNTIME_CALL_NAMES),
            "browser_transport": self._family(
                "browser_transport", RUNTIME_BROWSER_NAMES
            ),
            "views": {
                "wall_clock_stages": [
                    "protocol_cache_load", "protocol_compilation",
                    "protocol_cache_write", "checkpoint_load",
                    "assessment_cache_load", "direct_handling_total",
                    "primary_stage_total", "verification_stage_total",
                    "screening_session_update", "diagnostics_summary_write",
                ],
                "execution_components": [
                    "gemini_browser_total", "local_processing_total",
                ],
                "nested_details": [
                    "parsing", "result_validation", "assessment_cache_write",
                    "checkpoint_write", "output_csv_write",
                    "progress_update", "prisma_generation",
                ],
            },
            "dimensions": dimensions,
            "derived": {
                "gemini_browser_percentage_of_total": percentage(browser),
                "local_processing_percentage_of_total": percentage(local),
                "effective_seconds_per_selected_paper": per(total, selected_papers),
                "effective_seconds_per_fresh_primary_paper": per(
                    total, fresh_primary_papers
                ),
                "seconds_per_batch": per(total, submitted_batches),
            },
            "definitions": {
                "local_processing_total": (
                    "Residual non-browser wall time; not CPU-processing time."
                ),
                "double_counting": (
                    "Wall-clock stages, execution components, and nested details "
                    "are separate views and must not be summed together."
                ),
            },
            "instrumentation_errors": list(self.instrumentation_errors),
        }


class V24Diagnostics:
    APPROVED_FIELDS = (
        "event", "submission_number", "stage", "retry_number", "outcome",
        "recovery_action", "attempt_duration_ms", "response_selector",
        "response_container_count", "response_state", "generation_detected",
        "timeout_stage", "fallback_reason", "failure_class", "paper_count",
        "paper_ids", "batch_id", "subgroup_id", "criterion_count",
        "expected_criterion_object_count", "prompt_utf8_bytes",
        "response_utf8_bytes", "parsed_item_count", "over_budget",
        "runtime_metric", "runtime_family", "duration_seconds",
        "attempt_type", "exception_type", "exception_message",
        "response_empty", "syntactically_valid_json",
        "structured_failure_code", "parser_total_candidate_count",
        "parser_json_decodable_candidate_count",
        "parser_dictionary_candidate_count",
        "parser_schema_validation_failure_count",
        "parser_validation_error_count", "parser_validation_error_types",
        "parser_validation_error_locations", "parser_validation_error_messages",
        "parser_candidate_source",
        "parser_full_response_json_decodable",
        "parser_full_response_top_level_type",
        "parser_full_response_schema_valid",
        "parser_full_response_json_error_type",
        "parser_full_response_json_error_message",
        "parser_full_response_json_error_position",
        "parser_full_response_json_error_line",
        "parser_full_response_json_error_column",
        "parser_full_response_json_error_position_ratio",
        "parser_full_response_starts_with_object",
        "parser_full_response_starts_with_array",
        "parser_full_response_ends_with_object",
        "parser_full_response_ends_with_array",
        "parser_full_response_brace_balance",
        "parser_full_response_bracket_balance",
        "parser_full_response_inside_string_at_end",
        "parser_full_response_escape_pending_at_end",
        "parser_full_response_trailing_nonwhitespace_characters",
        "parser_full_response_raw_decode_succeeded",
        "parser_full_response_raw_decode_consumed_ratio",
        "response_return_reason", "response_complete_json_at_capture",
        "response_generation_detected_at_capture",
        "response_stable_duration_ms", "response_utf8_bytes_at_capture",
        "response_selector_at_capture", "response_container_count_at_capture",
    )

    def __init__(self, path: Path, runtime_metrics: V24RuntimeMetrics | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.runtime_metrics = runtime_metrics
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
        if event.get("event") == "gemini_web_runtime":
            runtime = self.runtime_metrics
            if runtime is not None:
                runtime.record(
                    str(event.get("runtime_metric") or "unknown"),
                    float(event.get("duration_seconds") or 0),
                    family=str(event.get("runtime_family") or "browser_transport"),
                    success=str(event.get("outcome") or "") != "failed",
                    stage=event.get("stage"),
                    attempt_type=event.get("attempt_type"),
                    outcome=event.get("outcome"),
                    retry_number=event.get("retry_number"),
                )
            safe = {field: event.get(field, "") for field in self.APPROVED_FIELDS}
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(safe, ensure_ascii=False) + "\n")
            return
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
        retry_number: int = 0,
        exception_type: str = "",
        exception_message: str = "",
        response_empty: bool = False,
        syntactically_valid_json: bool | None = None,
        structured_failure_code: str = "",
        parser_total_candidate_count: int = 0,
        parser_json_decodable_candidate_count: int = 0,
        parser_dictionary_candidate_count: int = 0,
        parser_schema_validation_failure_count: int = 0,
        parser_validation_error_count: int = 0,
        parser_validation_error_types: list[str] | None = None,
        parser_validation_error_locations: list[list[str | int]] | None = None,
        parser_validation_error_messages: list[str] | None = None,
        parser_candidate_source: str = "",
        parser_full_response_json_decodable: bool | None = None,
        parser_full_response_top_level_type: str = "",
        parser_full_response_schema_valid: bool | None = None,
        parser_full_response_json_error_type: str = "",
        parser_full_response_json_error_message: str = "",
        parser_full_response_json_error_position: int | None = None,
        parser_full_response_json_error_line: int | None = None,
        parser_full_response_json_error_column: int | None = None,
        parser_full_response_json_error_position_ratio: float | None = None,
        parser_full_response_starts_with_object: bool | None = None,
        parser_full_response_starts_with_array: bool | None = None,
        parser_full_response_ends_with_object: bool | None = None,
        parser_full_response_ends_with_array: bool | None = None,
        parser_full_response_brace_balance: int | None = None,
        parser_full_response_bracket_balance: int | None = None,
        parser_full_response_inside_string_at_end: bool | None = None,
        parser_full_response_escape_pending_at_end: bool | None = None,
        parser_full_response_trailing_nonwhitespace_characters: int | None = None,
        parser_full_response_raw_decode_succeeded: bool | None = None,
        parser_full_response_raw_decode_consumed_ratio: float | None = None,
        response_return_reason: str = "",
        response_complete_json_at_capture: bool | None = None,
        response_generation_detected_at_capture: bool | None = None,
        response_stable_duration_ms: int = 0,
        response_utf8_bytes_at_capture: int = 0,
        response_selector_at_capture: str = "",
        response_container_count_at_capture: int = 0,
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
            "retry_number": retry_number,
            "exception_type": exception_type,
            "exception_message": exception_message,
            "response_empty": response_empty,
            "syntactically_valid_json": syntactically_valid_json,
            "structured_failure_code": structured_failure_code,
            "parser_total_candidate_count": parser_total_candidate_count,
            "parser_json_decodable_candidate_count": (
                parser_json_decodable_candidate_count
            ),
            "parser_dictionary_candidate_count": parser_dictionary_candidate_count,
            "parser_schema_validation_failure_count": (
                parser_schema_validation_failure_count
            ),
            "parser_validation_error_count": parser_validation_error_count,
            "parser_validation_error_types": parser_validation_error_types or [],
            "parser_validation_error_locations": (
                parser_validation_error_locations or []
            ),
            "parser_validation_error_messages": parser_validation_error_messages or [],
            "parser_candidate_source": parser_candidate_source,
            "parser_full_response_json_decodable": (
                parser_full_response_json_decodable
            ),
            "parser_full_response_top_level_type": (
                parser_full_response_top_level_type
            ),
            "parser_full_response_schema_valid": parser_full_response_schema_valid,
            "parser_full_response_json_error_type": (
                parser_full_response_json_error_type
            ),
            "parser_full_response_json_error_message": (
                parser_full_response_json_error_message
            ),
            "parser_full_response_json_error_position": (
                parser_full_response_json_error_position
            ),
            "parser_full_response_json_error_line": (
                parser_full_response_json_error_line
            ),
            "parser_full_response_json_error_column": (
                parser_full_response_json_error_column
            ),
            "parser_full_response_json_error_position_ratio": (
                parser_full_response_json_error_position_ratio
            ),
            "parser_full_response_starts_with_object": (
                parser_full_response_starts_with_object
            ),
            "parser_full_response_starts_with_array": (
                parser_full_response_starts_with_array
            ),
            "parser_full_response_ends_with_object": (
                parser_full_response_ends_with_object
            ),
            "parser_full_response_ends_with_array": (
                parser_full_response_ends_with_array
            ),
            "parser_full_response_brace_balance": (
                parser_full_response_brace_balance
            ),
            "parser_full_response_bracket_balance": (
                parser_full_response_bracket_balance
            ),
            "parser_full_response_inside_string_at_end": (
                parser_full_response_inside_string_at_end
            ),
            "parser_full_response_escape_pending_at_end": (
                parser_full_response_escape_pending_at_end
            ),
            "parser_full_response_trailing_nonwhitespace_characters": (
                parser_full_response_trailing_nonwhitespace_characters
            ),
            "parser_full_response_raw_decode_succeeded": (
                parser_full_response_raw_decode_succeeded
            ),
            "parser_full_response_raw_decode_consumed_ratio": (
                parser_full_response_raw_decode_consumed_ratio
            ),
            "response_return_reason": response_return_reason,
            "response_complete_json_at_capture": response_complete_json_at_capture,
            "response_generation_detected_at_capture": (
                response_generation_detected_at_capture
            ),
            "response_stable_duration_ms": response_stable_duration_ms,
            "response_utf8_bytes_at_capture": response_utf8_bytes_at_capture,
            "response_selector_at_capture": response_selector_at_capture,
            "response_container_count_at_capture": (
                response_container_count_at_capture
            ),
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


@contextmanager
def _null_observer() -> Iterator[None]:
    yield


def _timed_gemini_call(
    browser,
    prompt: str,
    runtime_metrics: V24RuntimeMetrics | None,
    *,
    call_metrics: list[str],
    **metadata: Any,
) -> str:
    if runtime_metrics is None:
        return browser.submit_prompt_and_get_response(prompt)
    started = runtime_metrics.now()
    try:
        response = browser.submit_prompt_and_get_response(prompt)
    except Exception:
        ended = runtime_metrics.now()
        if started is not None and ended is not None:
            duration = ended - started
            runtime_metrics.record(
                "gemini_browser_total", duration, success=False, outcome="failed",
                **metadata,
            )
            for metric in dict.fromkeys(call_metrics):
                runtime_metrics.record(
                    metric, duration, family="gemini_calls", success=False,
                    outcome="failed", **metadata,
                )
        raise
    ended = runtime_metrics.now()
    if started is not None and ended is not None:
        duration = ended - started
        runtime_metrics.record(
            "gemini_browser_total", duration, success=True, outcome="completed",
            **metadata,
        )
        for metric in dict.fromkeys(call_metrics):
            runtime_metrics.record(
                metric, duration, family="gemini_calls", success=True,
                outcome="completed", **metadata,
            )
    return response


def _timed_browser_operation(
    runtime_metrics: V24RuntimeMetrics | None,
    operation: Callable[[], Any],
    *,
    stage: str,
) -> Any:
    if runtime_metrics is None:
        return operation()
    with runtime_metrics.observe("gemini_browser_total", stage=stage):
        return operation()


def _compile_protocol(
    browser,
    *,
    question: str,
    context: str,
    inclusion: str,
    exclusion: str,
    runtime_metrics: V24RuntimeMetrics | None = None,
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
            call_metrics = ["protocol"]
            attempt_type = "initial" if attempt == 0 else "structured_repair"
            if attempt:
                call_metrics.extend(("structured_repair", "bounded_retry"))
            raw = _timed_gemini_call(
                browser,
                prompt,
                runtime_metrics,
                call_metrics=call_metrics,
                stage="protocol",
                attempt_type=attempt_type,
                retry_number=attempt,
                paper_count=0,
            )
            parser = (
                runtime_metrics.observe("parsing", stage="protocol")
                if runtime_metrics is not None
                else _null_observer()
            )
            with parser:
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
                _timed_browser_operation(
                    runtime_metrics,
                    browser.recover_transport_failure,
                    stage="protocol_recovery",
                )
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


def _structured_failure_diagnostic_metadata(
    raw: str,
    exc: Exception,
    parser_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build bounded metadata without invoking or duplicating the parser."""
    try:
        exception_type = type(exc).__name__
    except Exception:
        exception_type = ""
    try:
        if isinstance(exc, LocalAIOutputError):
            exception_message = str(exc)[:300]
        elif callable(getattr(exc, "errors", None)):
            exception_message = json.dumps(
                exc.errors(include_input=False, include_url=False),
                ensure_ascii=True,
            )[:300]
        else:
            exception_message = "Structured response validation failed."
    except Exception:
        exception_message = ""
    try:
        response_empty = not str(raw or "").strip()
    except Exception:
        response_empty = False
    parser_diagnostics = parser_diagnostics or {}
    json_decodable_count = parser_diagnostics.get(
        "json_decodable_candidate_count"
    )
    return {
        "exception_type": exception_type,
        "exception_message": exception_message,
        "response_empty": response_empty,
        "syntactically_valid_json": (
            bool(json_decodable_count)
            if json_decodable_count is not None
            else None
        ),
        "structured_failure_code": parser_diagnostics.get("failure_code", ""),
        "parser_total_candidate_count": parser_diagnostics.get(
            "total_candidate_count", 0
        ),
        "parser_json_decodable_candidate_count": json_decodable_count or 0,
        "parser_dictionary_candidate_count": parser_diagnostics.get(
            "dictionary_candidate_count", 0
        ),
        "parser_schema_validation_failure_count": parser_diagnostics.get(
            "schema_validation_failure_count", 0
        ),
        "parser_validation_error_count": parser_diagnostics.get(
            "validation_error_count", 0
        ),
        "parser_validation_error_types": parser_diagnostics.get(
            "validation_error_types", []
        ),
        "parser_validation_error_locations": parser_diagnostics.get(
            "validation_error_locations", []
        ),
        "parser_validation_error_messages": parser_diagnostics.get(
            "validation_error_messages", []
        ),
        "parser_candidate_source": parser_diagnostics.get("candidate_source", ""),
        **_parser_full_response_assessment_metadata(parser_diagnostics),
    }


def _parser_full_response_assessment_metadata(
    parser_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    source = parser_diagnostics or {}
    names = (
        "json_decodable",
        "top_level_type",
        "schema_valid",
        "json_error_type",
        "json_error_message",
        "json_error_position",
        "json_error_line",
        "json_error_column",
        "json_error_position_ratio",
        "starts_with_object",
        "starts_with_array",
        "ends_with_object",
        "ends_with_array",
        "brace_balance",
        "bracket_balance",
        "inside_string_at_end",
        "escape_pending_at_end",
        "trailing_nonwhitespace_characters",
        "raw_decode_succeeded",
        "raw_decode_consumed_ratio",
    )
    return {
        f"parser_full_response_{name}": source.get(f"full_response_{name}")
        for name in names
    }


def _browser_capture_assessment_metadata(browser: Any) -> dict[str, Any]:
    names = (
        "response_return_reason",
        "response_complete_json_at_capture",
        "response_generation_detected_at_capture",
        "response_stable_duration_ms",
        "response_utf8_bytes_at_capture",
        "response_selector_at_capture",
        "response_container_count_at_capture",
    )
    try:
        source = getattr(browser, "last_response_capture_metadata", {})
        if callable(source):
            source = source()
        if not isinstance(source, dict):
            source = {}
        snapshot = dict(source)
    except Exception:
        snapshot = {}
    return {
        name: snapshot.get(name)
        for name in names
        if name in snapshot
    }


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
    runtime_metrics: V24RuntimeMetrics | None = None,
    subgroup_replay: bool = False,
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
        parser_diagnostics: dict[str, Any] = {}
        browser_capture_diagnostics: dict[str, Any] = {}
        attempt_failure_class = ""
        retry_number = retry_offset + attempt
        try:
            browser.set_attempt_context(
                stage=f"v24_{stage}",
                retry_number=retry_number,
            )
            attempt_type = (
                "subgroup_transport_replay"
                if subgroup_replay
                else (
                    "structured_repair"
                    if attempt > 0 or repair_only
                    else "initial"
                )
            )
            call_metrics = [stage]
            if attempt > 0 or repair_only:
                call_metrics.append("structured_repair")
            if retry_number > 0:
                call_metrics.append("bounded_retry")
            if subgroup_id:
                call_metrics.append("degraded_subgroup_attempt")
            if subgroup_replay:
                call_metrics.append("subgroup_transport_replay")
            raw = _timed_gemini_call(
                browser,
                request,
                runtime_metrics,
                call_metrics=call_metrics,
                stage=stage,
                attempt_type=attempt_type,
                retry_number=retry_number,
                paper_count=len(papers),
                batch_id=batch_id,
                subgroup_id=subgroup_id,
            )
            browser_capture_diagnostics = _browser_capture_assessment_metadata(
                browser
            )
            parser = (
                runtime_metrics.observe(
                    "parsing",
                    stage=stage,
                    attempt_type=attempt_type,
                    paper_count=len(papers),
                    batch_id=batch_id,
                    subgroup_id=subgroup_id,
                    retry_number=retry_number,
                )
                if runtime_metrics is not None
                else _null_observer()
            )
            with parser:
                compact = V24CompactAssessmentBatch.model_validate(
                    parse_structured_model_output(
                        raw,
                        V24CompactAssessmentBatch,
                        diagnostic_sink=parser_diagnostics.update,
                    )
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
                retry_number=retry_number,
                **_parser_full_response_assessment_metadata(parser_diagnostics),
                **browser_capture_diagnostics,
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
            structured_diagnostics: dict[str, Any] = {}
            if attempt_failure_class == V24_STRUCTURED_OUTPUT_FAILURE:
                try:
                    structured_diagnostics = _structured_failure_diagnostic_metadata(
                        raw, exc, parser_diagnostics
                    )
                except Exception:
                    # Diagnostic inspection cannot alter the parse failure.
                    structured_diagnostics = {
                        "exception_type": "",
                        "exception_message": "",
                        "response_empty": not bool(raw),
                        "syntactically_valid_json": None,
                    }
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
                retry_number=retry_number,
                **browser_capture_diagnostics,
                **structured_diagnostics,
            )
            diagnostics.retry()
            if attempt + 1 < max_attempts:
                if transport_failure:
                    _timed_browser_operation(
                        runtime_metrics,
                        browser.recover_transport_failure,
                        stage=f"{stage}_recovery",
                    )
                else:
                    _timed_browser_operation(
                        runtime_metrics,
                        browser.start_new_job_chat,
                        stage=f"{stage}_repair_chat",
                    )
    failure_class = (
        V24_TRANSPORT_FAILURE if transport_failure else V24_STRUCTURED_OUTPUT_FAILURE
    )
    failure_label = (
        "after one retry" if max_attempts > 1 else "during bounded degraded retry"
    )
    reason = f"Gemini Web v2.4 request failed {failure_label}: {last_error}"
    return {}, reason, failure_class


def _recover_failed_batch(
    browser,
    failure_class: str,
    action: str,
    runtime_metrics: V24RuntimeMetrics | None = None,
) -> None:
    browser.note_recovery(action)
    if failure_class == V24_TRANSPORT_FAILURE:
        _timed_browser_operation(
            runtime_metrics,
            lambda: browser.recover_transport_failure(exhausted=True),
            stage="degraded_recovery",
        )
    else:
        _timed_browser_operation(
            runtime_metrics,
            browser.start_new_job_chat,
            stage="degraded_repair_chat",
        )


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
    runtime_metrics: V24RuntimeMetrics | None = None,
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
        runtime_metrics=runtime_metrics,
    )
    if assessed:
        return assessed, {}
    if len(papers) <= 1:
        if failure_class == V24_TRANSPORT_FAILURE:
            _timed_browser_operation(
                runtime_metrics,
                lambda: browser.recover_transport_failure(exhausted=True),
                stage="single_paper_exhausted_recovery",
            )
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
        runtime_metrics,
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
            runtime_metrics=runtime_metrics,
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
                runtime_metrics,
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
                runtime_metrics=runtime_metrics,
                subgroup_replay=True,
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
            runtime_metrics,
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
    runtime_metrics: V24RuntimeMetrics | None = None,
) -> dict[str, Any]:
    runtime_metrics = runtime_metrics or V24RuntimeMetrics()
    run_started = runtime_metrics.now()
    run_succeeded = False
    primary_stage_started: float | None = None
    primary_stage_recorded = False
    verification_stage_started: float | None = None
    verification_stage_recorded = False

    def finish_metric(
        metric: str,
        started: float | None,
        *,
        success: bool = True,
        **metadata: Any,
    ) -> None:
        ended = runtime_metrics.now()
        if started is not None and ended is not None:
            runtime_metrics.record(
                metric, ended - started, success=success, **metadata
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_root = output.parent.parent / "cache" / "gemini_web_v24"
    diagnostics = V24Diagnostics(
        cache_root / "diagnostics" / f"{job_id}.jsonl",
        runtime_metrics=runtime_metrics,
    )
    with runtime_metrics.observe("protocol_cache_load"):
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
    browser_context_entered = False
    browser = None
    if protocol is None:
        with runtime_metrics.observe("progress_update", stage="protocol"):
            progress.begin_batches(job_id, "gemini_web_v24_protocol", 1, 1, 1)
        browser_context = browser_factory(GeminiWebV24Config(diagnostic_sink=diagnostics.record))
        try:
            browser = _timed_browser_operation(
                runtime_metrics,
                browser_context.__enter__,
                stage="browser_start",
            )
            browser_context_entered = True
            with runtime_metrics.observe("protocol_compilation"):
                protocol = _compile_protocol(
                    browser,
                    question=research_question,
                    context=research_context,
                    inclusion=inclusion_criteria,
                    exclusion=exclusion_criteria,
                    runtime_metrics=runtime_metrics,
                )
            with runtime_metrics.observe("protocol_cache_write"):
                _atomic_json(protocol_path, protocol.model_dump(mode="json"))
            with runtime_metrics.observe("progress_update", stage="protocol"):
                progress.update_batch(job_id, 1, 1)
            browser.note_recovery("v24_protocol_to_primary_clean_chat")
            _timed_browser_operation(
                runtime_metrics,
                browser.start_new_job_chat,
                stage="protocol_to_primary_chat",
            )
        except Exception:
            if browser_context is not None and browser_context_entered:
                browser_context_entered = False
                browser_context.__exit__(None, None, None)
            browser_context = None
            browser = None
            raise

    checkpoint = cache_root / "checkpoints" / f"{_contract_key(input_fingerprint, protocol.protocol_id)}.csv"
    with runtime_metrics.observe("checkpoint_load"):
        rows = _resume_rows(checkpoint, protocol.protocol_id, set(papers)) if resume else {}
    with runtime_metrics.observe("progress_update", stage="resume"):
        progress.set_resumed_count(job_id, len(rows))
    pending: list[V24Paper] = []
    for key, paper in papers.items():
        if key in rows:
            continue
        if not paper.abstract.strip():
            with runtime_metrics.observe(
                "direct_handling_total", paper_count=1, outcome="missing_abstract"
            ):
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
        with runtime_metrics.observe("assessment_cache_load", paper_count=1):
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

    progress.set_prisma_timing_observer(
        job_id,
        lambda duration, success: runtime_metrics.record(
            "prisma_generation", duration, success=success
        ),
    )
    try:
        if pending and browser is None:
            browser_context = browser_factory(GeminiWebV24Config(diagnostic_sink=diagnostics.record))
            browser = _timed_browser_operation(
                runtime_metrics,
                browser_context.__enter__,
                stage="browser_start",
            )
            browser_context_entered = True

        primary_batch_size, primary_over_budget = plan_assessment_batch_size(
            protocol, stage="primary"
        )
        primary_batches = (
            len(pending) + primary_batch_size - 1
        ) // primary_batch_size
        primary_stage_started = runtime_metrics.now()
        with runtime_metrics.observe("progress_update", stage="primary"):
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
                runtime_metrics=runtime_metrics,
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
                    with runtime_metrics.observe(
                        "result_validation", stage="primary", paper_count=1
                    ):
                        result = _validate_and_decide(
                            assessed[paper.paper_id], protocol, paper
                        )
                    result["runtime_seconds"] = round(elapsed, 4)
                    result["cache_hit"] = False
                    route = _verification_route(result, protocol)
                    route_by_key[paper.paper_id] = route
                    if route:
                        result["route_used"] = route
                        result["verification_status"] = "pending"
                        verification_keys.append(paper.paper_id)
                    else:
                        with runtime_metrics.observe(
                            "assessment_cache_write", stage="primary", paper_count=1
                        ):
                            _save_assessment_cache(
                                cache_root, protocol.protocol_id, paper, result
                            )
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
            with runtime_metrics.observe("checkpoint_write", stage="primary"):
                _atomic_csv(checkpoint, ordered)
            with runtime_metrics.observe("output_csv_write", stage="primary"):
                _atomic_csv(output, ordered)
            counts = screening_session.counts(ordered)
            with runtime_metrics.observe("progress_update", stage="primary"):
                progress.update_batch(
                    job_id, batch_number + 1,
                    min(len(pending), (batch_number + 1) * primary_batch_size),
                )
            with runtime_metrics.observe("progress_update", stage="primary"):
                progress.update_counts(
                    job_id, len(ordered), counts["keep"], counts["maybe"], counts["reject"],
                )

        finish_metric("primary_stage_total", primary_stage_started)
        primary_stage_recorded = True
        verification_batch_size, verification_over_budget = plan_assessment_batch_size(
            protocol, stage="verification"
        )
        verification_batches = (
            len(verification_keys) + verification_batch_size - 1
        ) // verification_batch_size
        verification_stage_started = runtime_metrics.now()
        if verification_keys:
            browser.note_recovery("v24_primary_to_verification_clean_chat")
            _timed_browser_operation(
                runtime_metrics,
                browser.start_new_job_chat,
                stage="primary_to_verification_chat",
            )
        with runtime_metrics.observe("progress_update", stage="verification"):
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
                runtime_metrics=runtime_metrics,
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
                    with runtime_metrics.observe(
                        "result_validation", stage="verification", paper_count=1
                    ):
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
                    with runtime_metrics.observe(
                        "assessment_cache_write",
                        stage="verification",
                        paper_count=1,
                    ):
                        _save_assessment_cache(
                            cache_root, protocol.protocol_id, paper, result
                        )
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
            with runtime_metrics.observe("checkpoint_write", stage="verification"):
                _atomic_csv(checkpoint, ordered)
            with runtime_metrics.observe("output_csv_write", stage="verification"):
                _atomic_csv(output, ordered)
            counts = screening_session.counts(ordered)
            with runtime_metrics.observe("progress_update", stage="verification"):
                progress.update_batch(
                    job_id, batch_number + 1,
                    min(len(verification_keys), (batch_number + 1) * verification_batch_size),
                )
            with runtime_metrics.observe("progress_update", stage="verification"):
                progress.update_counts(
                    job_id, len(ordered), counts["keep"], counts["maybe"], counts["reject"],
                )

        finish_metric("verification_stage_total", verification_stage_started)
        verification_stage_recorded = True
        ordered = [rows[key] for key in papers]
        with runtime_metrics.observe("checkpoint_write", stage="finalization"):
            _atomic_csv(checkpoint, ordered)
        with runtime_metrics.observe("output_csv_write", stage="finalization"):
            _atomic_csv(output, ordered)
        with runtime_metrics.observe("screening_session_update"):
            screening_session.set_results(
                ordered,
                job_id=job_id,
                output_path=output_path,
                architecture_version=GEMINI_WEB_V24_VERSION,
            )
        counts = screening_session.counts(ordered)
        current_time = runtime_metrics.now()
        runtime = round(
            current_time - run_started, 4
        ) if current_time is not None and run_started is not None else 0.0
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
        summary["runtime_metrics"] = {
            "schema_version": "gemini-web-v2.4-runtime-v1",
            "status": "finalizing",
        }
        with runtime_metrics.observe("diagnostics_summary_write"):
            _atomic_json(diagnostics.path.with_suffix(".summary.json"), summary)
        with runtime_metrics.observe("progress_update", stage="finish"):
            progress.finish(job_id)
        if browser_context is not None and browser_context_entered:
            entered_context = browser_context
            browser_context_entered = False
            _timed_browser_operation(
                runtime_metrics,
                lambda: entered_context.__exit__(None, None, None),
                stage="browser_close",
            )
            browser_context = None
            browser = None
        finish_metric("total_job", run_started)
        try:
            summary["runtime_metrics"] = runtime_metrics.serialize(
                selected_papers=len(ordered),
                fresh_primary_papers=len(origin_ids["fresh_primary"]),
                submitted_batches=(
                    diagnostics.assessment_batches_submitted["primary"]
                    + diagnostics.assessment_batches_submitted["verification"]
                ),
            )
        except Exception as exc:
            runtime_metrics._error(exc)
            summary["runtime_metrics"] = {
                "schema_version": "gemini-web-v2.4-runtime-v1",
                "status": "serialization_failed",
                "instrumentation_errors": list(runtime_metrics.instrumentation_errors),
            }
        try:
            _atomic_json(diagnostics.path.with_suffix(".summary.json"), summary)
        except Exception as exc:
            # The ordinary summary was already persisted; only its additive
            # runtime finalization failed.
            runtime_metrics._error(exc)
        run_succeeded = True
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
        if primary_stage_started is not None and not primary_stage_recorded:
            finish_metric("primary_stage_total", primary_stage_started, success=False)
        if verification_stage_started is not None and not verification_stage_recorded:
            finish_metric(
                "verification_stage_total",
                verification_stage_started,
                success=False,
            )
        if not run_succeeded:
            finish_metric("total_job", run_started, success=False)
        progress.clear_prisma_timing_observer(job_id)
        if browser_context is not None and browser_context_entered:
            browser_context_entered = False
            browser_context.__exit__(None, None, None)
