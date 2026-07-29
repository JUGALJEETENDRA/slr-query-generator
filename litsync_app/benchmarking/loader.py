from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from .contracts import (
    BenchmarkSpec,
    GoldLabel,
    ProvenanceClass,
    RunProvenance,
    ScreeningDecision,
    validate_identifier,
)
from .errors import GoldValidationError, RunArtifactError, SpecValidationError
from .provenance import (
    canonical_fingerprint,
    file_fingerprint,
    screening_output_fingerprint,
    source_row_id,
    text_value,
)


FORBIDDEN_DIAGNOSTIC_FIELDS = {
    "prompt", "question", "research_question", "title", "abstract",
    "response", "raw_response", "response_text", "content",
}


@dataclass(frozen=True)
class LoadedSpec:
    spec: BenchmarkSpec
    path: Path


@dataclass(frozen=True)
class LoadedGold:
    rows: dict[str, dict[str, str]]
    resolved_ids: list[str]
    unsure_ids: list[str]
    file_fingerprint: str


@dataclass(frozen=True)
class LoadedRun:
    job_id: str
    rows: dict[str, dict[str, str]]
    ordered_ids: list[str]
    summary: dict[str, Any]
    diagnostics: list[dict[str, Any]]
    prisma: dict[str, Any]
    provenance: RunProvenance
    paths: dict[str, Path]


def _spec_fingerprint(payload: dict[str, Any]) -> str:
    content = dict(payload)
    content.pop("benchmark_spec_fingerprint", None)
    return canonical_fingerprint(content)


def load_spec(path: str | Path) -> LoadedSpec:
    target = Path(path).resolve()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecValidationError(f"benchmark spec is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise SpecValidationError("benchmark spec must be a JSON object")
    calculated = _spec_fingerprint(payload)
    supplied = str(payload.get("benchmark_spec_fingerprint") or "")
    if supplied != calculated:
        raise SpecValidationError(
            "benchmark_spec_fingerprint does not match canonical spec content"
        )
    try:
        spec = BenchmarkSpec.model_validate(payload)
    except ValidationError as exc:
        raise SpecValidationError(str(exc)) from exc
    return LoadedSpec(spec=spec, path=target)


def load_gold(loaded_spec: LoadedSpec) -> LoadedGold:
    spec = loaded_spec.spec
    path = Path(spec.gold_label_file)
    if path.is_absolute():
        raise GoldValidationError("gold label file must be relative to the benchmark spec")
    spec_root = loaded_spec.path.parent.resolve()
    path = (spec_root / path).resolve()
    try:
        path.relative_to(spec_root)
    except ValueError as exc:
        raise GoldValidationError(
            "gold label file must remain within the benchmark spec directory"
        ) from exc
    try:
        fingerprint = file_fingerprint(path)
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    except OSError as exc:
        raise GoldValidationError(f"gold label file is unreadable: {exc}") from exc
    if fingerprint != spec.gold_file_fingerprint:
        raise GoldValidationError("gold_file_fingerprint does not match the frozen file")
    required = {"Source_Row_Index", "Gold_Decision"}
    if not required.issubset(frame.columns):
        raise GoldValidationError(
            "gold file requires Source_Row_Index and Gold_Decision"
        )
    rows: dict[str, dict[str, str]] = {}
    for record in frame.to_dict(orient="records"):
        source_id = source_row_id(record.get("Source_Row_Index"))
        if not source_id.strip():
            raise GoldValidationError("gold source-row IDs must be non-empty")
        if source_id in rows:
            raise GoldValidationError(f"duplicate gold source-row ID: {source_id}")
        label = text_value(record.get("Gold_Decision")).strip().upper()
        try:
            GoldLabel(label)
        except ValueError as exc:
            raise GoldValidationError(f"invalid gold label for row {source_id}: {label}") from exc
        rows[source_id] = {
            str(key): text_value(value)
            for key, value in record.items()
        }
        rows[source_id]["Gold_Decision"] = label
        rows[source_id]["Source_Row_Index"] = source_id
    expected = set(spec.gold_selected_source_row_ids)
    actual = set(rows)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise GoldValidationError(
            f"gold selection mismatch; missing={missing}, extra={extra}"
        )
    resolved = sorted(
        source_id
        for source_id, row in rows.items()
        if row["Gold_Decision"] in {"KEEP", "REJECT"}
    )
    unsure = sorted(set(rows) - set(resolved))
    return LoadedGold(rows, resolved, unsure, fingerprint)


def _safe_job_id(job_id: str) -> str:
    try:
        selected = validate_identifier(str(job_id or ""), label="screening job ID")
    except ValueError as exc:
        raise RunArtifactError("invalid screening job ID") from exc
    return selected


def _artifact_paths(root: Path, job_id: str) -> dict[str, Path]:
    paths = {
        "csv": root / "runs" / f"screened-{job_id}.csv",
        "summary": root / "cache" / "gemini_web_v24" / "diagnostics" / f"{job_id}.summary.json",
        "diagnostics": root / "cache" / "gemini_web_v24" / "diagnostics" / f"{job_id}.jsonl",
        "prisma": root / "prisma" / f"{job_id}.json",
    }
    for path in paths.values():
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise RunArtifactError("screening artifact path escapes the artifacts root") from exc
    return paths


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunArtifactError(f"{label} artifact is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise RunArtifactError(f"{label} artifact must be a JSON object")
    return value


def _read_diagnostics(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RunArtifactError(f"diagnostic stream is unreadable: {exc}") from exc
    events = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunArtifactError(f"invalid diagnostic JSON on line {number}") from exc
        if not isinstance(event, dict):
            raise RunArtifactError(f"diagnostic line {number} is not an object")
        unsafe = set(event) & FORBIDDEN_DIAGNOSTIC_FIELDS
        if unsafe:
            raise RunArtifactError(f"unsafe diagnostic fields: {sorted(unsafe)}")
        events.append(event)
    return events


def _normalized_protocol(value: Any) -> str:
    return text_value(value)


def _string_list(
    payload: dict[str, Any],
    field: str,
    reasons: list[str],
) -> list[str]:
    value = payload.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        reasons.append(f"summary field {field} must be a list of strings")
        return []
    return list(value)


def _integer_field(
    payload: dict[str, Any],
    field: str,
    reasons: list[str],
    *,
    default: int = -1,
) -> int:
    value = payload.get(field)
    if isinstance(value, bool):
        reasons.append(f"summary field {field} must be an integer")
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        reasons.append(f"summary field {field} must be an integer")
        return default


def _valid_reason_code(value: str) -> bool:
    return (
        bool(value)
        and value.isascii()
        and value[0].islower()
        and all(character.islower() or character.isdigit() or character == "_" for character in value)
    )


def _require_finite_number(
    payload: dict[str, Any],
    field: str,
    reasons: list[str],
) -> None:
    value = payload.get(field)
    try:
        number = float(value)
    except (TypeError, ValueError):
        reasons.append(f"summary field {field} must be numeric")
        return
    if not math.isfinite(number):
        reasons.append(f"summary field {field} must be finite")
def _version_identity_failures(
    rows: dict[str, dict[str, str]],
    summary: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    prompt_versions = {
        text_value(row.get("Prompt_Version")).strip()
        for row in rows.values()
    }
    protocol_ids = {
        text_value(row.get("Protocol_ID")).strip()
        for row in rows.values()
    }
    if prompt_versions != {str(summary.get("assessment_prompt_version") or "")}:
        reasons.append("row prompt versions differ from the run summary")
    if protocol_ids != {str(summary.get("protocol_id") or "")}:
        reasons.append("row protocol IDs differ from the run summary")
    return reasons


def load_run(
    loaded_spec: LoadedSpec,
    gold: LoadedGold,
    job_id: str,
    artifacts_root: str | Path = "outputs",
) -> LoadedRun:
    spec = loaded_spec.spec
    selected_job = _safe_job_id(job_id)
    root = Path(artifacts_root).resolve()
    paths = _artifact_paths(root, selected_job)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RunArtifactError(f"missing run artifacts: {', '.join(missing)}")
    summary = _read_json(paths["summary"], "summary")
    prisma = _read_json(paths["prisma"], "PRISMA")
    diagnostics = _read_diagnostics(paths["diagnostics"])
    try:
        frame = pd.read_csv(
            paths["csv"],
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    except OSError as exc:
        raise RunArtifactError(f"screening CSV is unreadable: {exc}") from exc
    required = {
        "Source_Row_Index", "Decision", "Title", "Abstract", "Protocol_ID",
        "Prompt_Version", "Validation_Status", "Evidence_JSON",
        "Execution_Origin", "Direct_Handling_Reason",
    }
    if not required.issubset(frame.columns):
        raise RunArtifactError(
            "screening CSV lacks required columns: "
            + ", ".join(sorted(required - set(frame.columns)))
        )
    rows: dict[str, dict[str, str]] = {}
    ordered_ids: list[str] = []
    evidence_errors: list[str] = []
    for record in frame.to_dict(orient="records"):
        source_id = source_row_id(record.get("Source_Row_Index"))
        if not source_id.strip() or source_id in rows:
            raise RunArtifactError("run source-row IDs must be unique and non-empty")
        decision = text_value(record.get("Decision")).strip().upper()
        try:
            ScreeningDecision(decision)
        except ValueError as exc:
            raise RunArtifactError(f"invalid decision for row {source_id}") from exc
        row = {str(key): text_value(value) for key, value in record.items()}
        row["Source_Row_Index"] = source_id
        row["Decision"] = decision
        try:
            evidence = json.loads(row.get("Evidence_JSON") or "[]")
            if not isinstance(evidence, list):
                raise ValueError
            if any(not isinstance(span, dict) for span in evidence):
                raise ValueError
        except (json.JSONDecodeError, TypeError, ValueError):
            evidence_errors.append(f"invalid evidence JSON for row {source_id}")
        rows[source_id] = row
        ordered_ids.append(source_id)

    reasons: list[str] = []
    reasons.extend(evidence_errors)
    expected_run_ids = spec.run_selected_source_row_ids
    if ordered_ids != expected_run_ids:
        reasons.append("completed run population differs from run_selected_source_row_ids")
    for source_id, gold_row in gold.rows.items():
        run_row = rows.get(source_id)
        if run_row is None:
            reasons.append(f"gold row {source_id} is missing from the completed run")
            continue
        for column in ("Title", "Abstract"):
            expected = gold_row.get(column, "")
            if expected and expected != run_row.get(column, ""):
                reasons.append(f"immutable {column.lower()} mismatch for gold row {source_id}")

    if str(summary.get("job_id") or "") != selected_job:
        reasons.append("summary job ID does not match requested job")
    if str(prisma.get("job_id") or "") != selected_job:
        reasons.append("PRISMA job ID does not match requested job")
    diagnostic_name = Path(str(summary.get("diagnostics_path") or "")).name
    if diagnostic_name != paths["diagnostics"].name:
        reasons.append("summary diagnostic path belongs to a different job")
    if summary.get("run_status") != "complete":
        reasons.append("run summary does not record complete status")
    for field in (
        "runtime_seconds", "papers_per_minute", "retry_count",
        "primary_batches_submitted", "primary_papers_requested",
        "verification_batches_submitted", "verification_papers_requested",
        "technical_fallback_count", "verification_validated_agreements",
        "verification_validated_disagreements",
        "verification_semantic_validation_failures",
        "degraded_subgroup_replay_count",
        "degraded_subgroup_replay_success_count",
    ):
        _require_finite_number(summary, field, reasons)
    for field in ("detector_outcomes", "route_counts", "verification_outcomes"):
        if not isinstance(summary.get(field), dict):
            reasons.append(f"summary field {field} must be an object")
    if _integer_field(summary, "run_selected_count", reasons) != len(rows):
        reasons.append("summary selected-row count differs from CSV")
    if _string_list(summary, "run_selected_source_row_ids", reasons) != ordered_ids:
        reasons.append("summary selected-row IDs differ from CSV")
    screening_plan = prisma.get("screening_plan")
    if not isinstance(screening_plan, dict):
        reasons.append("PRISMA screening_plan must be an object")
        screening_plan = {}
    try:
        plan_count = int(screening_plan.get("records_selected"))
    except (TypeError, ValueError):
        reasons.append("PRISMA records_selected must be an integer")
        plan_count = -1
    if plan_count != len(rows):
        reasons.append("PRISMA selected-row count differs from CSV")
    screening_state = prisma.get("screening_state")
    if not isinstance(screening_state, dict):
        reasons.append("PRISMA screening_state must be an object")
        screening_state = {}
    prisma_counts = screening_state.get("counts", {})
    if not isinstance(prisma_counts, dict):
        reasons.append("PRISMA screening counts must be an object")
        prisma_counts = {}
    csv_counts = {
        name.lower(): sum(row["Decision"] == name for row in rows.values())
        for name in ("KEEP", "MAYBE", "REJECT")
    }
    if prisma_counts:
        try:
            counts_disagree = any(
                int(prisma_counts.get(key, -1)) != value
                for key, value in csv_counts.items()
            )
        except (TypeError, ValueError):
            reasons.append("PRISMA decision counts must be integers")
        else:
            if counts_disagree:
                reasons.append("PRISMA decision counts differ from CSV")

    protocol_inputs = prisma.get("protocol_inputs", {})
    if not isinstance(protocol_inputs, dict):
        reasons.append("PRISMA protocol_inputs must be an object")
        protocol_inputs = {}
    expected_protocol_inputs = {
        "research_question": spec.research_question,
        "research_context": spec.research_context,
        "inclusion_criteria": spec.inclusion_criteria,
        "exclusion_criteria": spec.exclusion_criteria,
    }
    for field, expected in expected_protocol_inputs.items():
        if _normalized_protocol(protocol_inputs.get(field)) != _normalized_protocol(expected):
            reasons.append(f"PRISMA protocol input differs: {field}")

    summary_dataset = str(summary.get("source_dataset_fingerprint") or "")
    summary_input = str(summary.get("screening_input_fingerprint") or "")
    prisma_input = str(prisma.get("input_fingerprint") or "")
    calculated_output = screening_output_fingerprint(
        rows[source_id] for source_id in ordered_ids
    )
    summary_output = str(summary.get("screening_output_fingerprint") or "")
    if summary_dataset != spec.source_dataset_fingerprint:
        reasons.append("source dataset fingerprint is missing or mismatched")
    if len({summary_input, prisma_input, spec.screening_input_fingerprint}) != 1:
        reasons.append("exact-byte screening input fingerprints do not agree")
    if summary_output != calculated_output:
        reasons.append("screening output fingerprint is missing or mismatched")
    reasons.extend(_version_identity_failures(rows, summary))

    valid_origins = {
        "resumed",
        "assessment_cache_hit",
        "fresh_primary",
        "directly_handled_without_primary",
    }
    row_origins: dict[str, list[str]] = {origin: [] for origin in valid_origins}
    row_direct_reasons: dict[str, str] = {}
    for source_id in ordered_ids:
        row = rows[source_id]
        origin = row.get("Execution_Origin", "")
        direct_reason = row.get("Direct_Handling_Reason", "")
        if origin not in valid_origins:
            reasons.append(f"invalid Execution_Origin for row {source_id}")
            continue
        row_origins[origin].append(source_id)
        if origin == "directly_handled_without_primary":
            if not _valid_reason_code(direct_reason):
                reasons.append(f"direct row {source_id} lacks Direct_Handling_Reason")
            else:
                row_direct_reasons[source_id] = direct_reason
        elif direct_reason:
            reasons.append(f"non-direct row {source_id} has Direct_Handling_Reason")

    resumed = _string_list(summary, "resumed_source_row_ids", reasons)
    cached = _string_list(
        summary, "assessment_cache_hit_source_row_ids", reasons
    )
    fresh = _string_list(summary, "fresh_primary_source_row_ids", reasons)
    direct = _string_list(
        summary, "directly_handled_without_primary_source_row_ids", reasons
    )
    missing_abstract = _string_list(
        summary, "missing_abstract_source_row_ids", reasons
    )
    expected_groups = {
        "resumed": resumed,
        "assessment_cache_hit": cached,
        "fresh_primary": fresh,
        "directly_handled_without_primary": direct,
    }
    for origin, summary_ids in expected_groups.items():
        if summary_ids != row_origins[origin]:
            reasons.append(f"{origin} summary IDs differ from persisted row origins")
    groups = resumed + cached + fresh + direct
    if (
        len(groups) != len(set(groups))
        or set(groups) != set(ordered_ids)
        or any(len(values) != len(set(values)) for values in expected_groups.values())
    ):
        reasons.append("execution-origin provenance is incomplete, overlapping, or duplicated")
    count_fields = {
        "resumed_count": resumed,
        "assessment_cache_hits_loaded": cached,
        "fresh_primary_papers": fresh,
        "directly_handled_without_primary_count": direct,
    }
    for field, values in count_fields.items():
        if _integer_field(summary, field, reasons) != len(values):
            reasons.append(f"{field} does not match its row IDs")
    if _integer_field(summary, "primary_papers_requested", reasons) != len(fresh):
        reasons.append("primary_papers_requested does not match fresh-primary rows")
    raw_direct_reasons = summary.get("direct_handling_reasons")
    if not isinstance(raw_direct_reasons, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in (raw_direct_reasons.items() if isinstance(raw_direct_reasons, dict) else [])
    ):
        reasons.append("direct_handling_reasons must be a string mapping")
        raw_direct_reasons = {}
    if raw_direct_reasons != row_direct_reasons:
        reasons.append("direct_handling_reasons differ from persisted final rows")
    row_missing_abstract = [
        source_id
        for source_id in ordered_ids
        if not rows[source_id].get("Abstract", "").strip()
    ]
    if missing_abstract != row_missing_abstract:
        reasons.append("missing-abstract summary IDs differ from persisted final rows")
    if _integer_field(summary, "missing_abstract_count", reasons) != len(
        missing_abstract
    ):
        reasons.append("missing_abstract_count does not match its row IDs")

    if reasons:
        classification = ProvenanceClass.INVALID_PROVENANCE
    elif resumed:
        classification = (
            ProvenanceClass.FULLY_RESUMED
            if len(resumed) == len(ordered_ids)
            else ProvenanceClass.PARTIALLY_RESUMED
        )
    elif cached:
        classification = ProvenanceClass.WARM_CACHE
    else:
        classification = ProvenanceClass.COLD
    provenance = RunProvenance(
        classification=classification,
        valid=not reasons,
        reasons=reasons,
        job_id=selected_job,
        run_selected_source_row_ids=ordered_ids,
        resumed_source_row_ids=resumed,
        cache_hit_source_row_ids=cached,
        fresh_primary_source_row_ids=fresh,
        directly_handled_without_primary_source_row_ids=direct,
        direct_handling_reasons=row_direct_reasons,
        missing_abstract_source_row_ids=missing_abstract,
        source_dataset_fingerprint=summary_dataset,
        screening_input_fingerprint=summary_input,
        screening_output_fingerprint=calculated_output,
        gold_file_fingerprint=gold.file_fingerprint,
        benchmark_spec_fingerprint=spec.benchmark_spec_fingerprint,
        architecture_version=str(summary.get("architecture_version") or ""),
        protocol_id=str(summary.get("protocol_id") or ""),
        protocol_cache_version=str(summary.get("protocol_cache_version") or ""),
        assessment_prompt_version=str(summary.get("assessment_prompt_version") or ""),
        assessment_cache_version=str(summary.get("assessment_cache_version") or ""),
    )
    return LoadedRun(
        selected_job, rows, ordered_ids, summary, diagnostics, prisma, provenance, paths
    )
