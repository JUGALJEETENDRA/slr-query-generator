from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SPEC_SCHEMA_VERSION = "litsync-screening-benchmark-spec-v1"
RESULT_SCHEMA_VERSION = "litsync-screening-benchmark-result-v1"
COMPARISON_SCHEMA_VERSION = "litsync-screening-benchmark-comparison-v1"
IDENTIFIER_MAX_LENGTH = 128
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def validate_identifier(value: str, *, label: str) -> str:
    selected = str(value or "")
    if (
        not selected
        or len(selected) > IDENTIFIER_MAX_LENGTH
        or selected != selected.strip()
        or selected.endswith(".")
        or selected in {".", ".."}
    ):
        raise ValueError(f"invalid {label}")
    if not selected.isascii():
        raise ValueError(f"invalid {label}")
    if not selected[0].isalnum() or any(
        not (character.isalnum() or character in "._-")
        for character in selected
    ):
        raise ValueError(f"invalid {label}")
    device_stem = selected.split(".", 1)[0].upper()
    if device_stem in WINDOWS_RESERVED_NAMES:
        raise ValueError(f"invalid {label}")
    return selected


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoldLabel(str, Enum):
    KEEP = "KEEP"
    REJECT = "REJECT"
    UNSURE = "UNSURE"


class ScreeningDecision(str, Enum):
    KEEP = "KEEP"
    MAYBE = "MAYBE"
    REJECT = "REJECT"


class ProvenanceClass(str, Enum):
    COLD = "COLD"
    WARM_CACHE = "WARM_CACHE"
    PARTIALLY_RESUMED = "PARTIALLY_RESUMED"
    FULLY_RESUMED = "FULLY_RESUMED"
    INVALID_PROVENANCE = "INVALID_PROVENANCE"


class BenchmarkVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INVALID = "INVALID"
    PROVISIONAL = "PROVISIONAL"


class ExpectedProtocolIdentity(StrictModel):
    protocol_id: str = Field(min_length=1)
    protocol_cache_version: str = Field(min_length=1)


class ExpectedAssessmentIdentity(StrictModel):
    architecture_version: str = Field(min_length=1)
    assessment_prompt_version: str = Field(min_length=1)
    assessment_cache_version: str = Field(min_length=1)


class Threshold(StrictModel):
    comparator: Literal["ge", "gt", "le", "lt", "eq"]
    value: float


class ReleaseThresholds(StrictModel):
    quality: dict[str, Threshold]
    reliability: dict[str, Threshold]
    require_cold: bool = True


class BenchmarkSpec(StrictModel):
    spec_schema_version: Literal["litsync-screening-benchmark-spec-v1"] = SPEC_SCHEMA_VERSION
    benchmark_id: str
    benchmark_version: str
    name: str = Field(min_length=1)
    research_question: str = Field(min_length=1)
    research_context: str = ""
    inclusion_criteria: str = ""
    exclusion_criteria: str = ""
    gold_label_file: str = Field(min_length=1)
    source_dataset_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    screening_input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    gold_file_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_spec_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_selected_source_row_ids: list[str] = Field(min_length=1)
    gold_selected_source_row_ids: list[str] = Field(min_length=1)
    unresolved_label_policy: Literal["exclude_from_resolved_metrics"]
    metric_definitions: dict[str, str]
    release_thresholds: ReleaseThresholds
    expected_protocol_identity: ExpectedProtocolIdentity
    expected_assessment_identity: ExpectedAssessmentIdentity
    notes: str = ""

    @field_validator("benchmark_id", "benchmark_version")
    @classmethod
    def validate_safe_identifiers(cls, value: str, info) -> str:
        return validate_identifier(value, label=info.field_name.replace("_", " "))

    @model_validator(mode="after")
    def validate_populations(self) -> "BenchmarkSpec":
        run_ids = [value.strip() for value in self.run_selected_source_row_ids]
        gold_ids = [value.strip() for value in self.gold_selected_source_row_ids]
        if any(not value for value in run_ids + gold_ids):
            raise ValueError("benchmark row IDs must be non-empty strings")
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("run_selected_source_row_ids must be unique")
        if len(gold_ids) != len(set(gold_ids)):
            raise ValueError("gold_selected_source_row_ids must be unique")
        if not set(gold_ids).issubset(run_ids):
            raise ValueError("gold-selected rows must be a subset of run-selected rows")
        if "manual_review_burden" in self.release_thresholds.quality:
            raise ValueError("manual review thresholds must name an explicit population")
        return self


class ConfidenceInterval(StrictModel):
    level: float = 0.95
    lower: float
    upper: float


class MetricValue(StrictModel):
    numerator: float | int | None
    denominator: float | int | None
    value: float | int | None
    confidence_interval: ConfidenceInterval | None = None
    population_scope: str
    metric_definition_version: str
    artifact_source: str
    artifact_field: str


class GateFailure(StrictModel):
    rule: str
    reason: str
    observed: float | int | str | None = None
    comparator: str | None = None
    threshold: float | int | str | None = None
    numerator: float | int | None = None
    denominator: float | int | None = None
    population_scope: str | None = None
    metric_definition_version: str | None = None


class GateOutcome(StrictModel):
    verdict: BenchmarkVerdict
    failures: list[GateFailure] = Field(default_factory=list)


class RunProvenance(StrictModel):
    classification: ProvenanceClass
    valid: bool
    reasons: list[str]
    job_id: str
    run_selected_source_row_ids: list[str]
    resumed_source_row_ids: list[str]
    cache_hit_source_row_ids: list[str]
    fresh_primary_source_row_ids: list[str]
    directly_handled_without_primary_source_row_ids: list[str]
    direct_handling_reasons: dict[str, str]
    missing_abstract_source_row_ids: list[str]
    source_dataset_fingerprint: str
    screening_input_fingerprint: str
    screening_output_fingerprint: str
    gold_file_fingerprint: str
    benchmark_spec_fingerprint: str
    architecture_version: str
    protocol_id: str
    protocol_cache_version: str
    assessment_prompt_version: str
    assessment_cache_version: str


class BenchmarkResult(StrictModel):
    result_schema_version: Literal["litsync-screening-benchmark-result-v1"] = RESULT_SCHEMA_VERSION
    benchmark_id: str
    benchmark_version: str
    job_id: str
    benchmark_spec_fingerprint: str
    gold_file_fingerprint: str
    resolved_gold_source_row_ids: list[str]
    unsure_gold_source_row_ids: list[str]
    provenance: RunProvenance
    metrics: dict[str, MetricValue]
    confusion_matrix: dict[str, dict[str, int]]
    false_keep_source_row_ids: list[str]
    false_reject_source_row_ids: list[str]
    row_outcomes: list[dict[str, Any]]
    gate: GateOutcome


class BenchmarkComparison(StrictModel):
    comparison_schema_version: Literal[
        "litsync-screening-benchmark-comparison-v1"
    ] = COMPARISON_SCHEMA_VERSION
    benchmark_id: str
    benchmark_version: str
    benchmark_spec_fingerprint: str
    job_ids: list[str]
    valid: bool
    reasons: list[str]
    metric_deltas: dict[str, float | int | None]
    movement_matrix: dict[str, dict[str, int]]
    transitions: list[dict[str, Any]]
    newly_introduced_false_rejects: list[dict[str, Any]]
    corrected_false_rejects: list[dict[str, Any]]
    newly_introduced_false_keeps: list[dict[str, Any]]
    corrected_false_keeps: list[dict[str, Any]]
    pairwise_comparisons: list[dict[str, Any]]
