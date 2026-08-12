from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


EXPORT_SCHEMA_VERSION = "litsync-screening-export-v1"
EXPORT_FILENAMES = {
    "all": "screened_all.csv",
    "keep": "screened_keep.csv",
    "maybe": "screened_maybe.csv",
    "reject": "screened_reject.csv",
    "review_queue": "screened_maybe_review_queue.csv",
    "summary": "screening_summary.json",
}
VALID_DECISIONS = {"KEEP", "MAYBE", "REJECT"}
JOB_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")

CANONICAL_COLUMNS = [
    "Job_ID",
    "Protocol_ID",
    "Review_Protocol_ID",
    "Research_Question",
    "Research_Context",
    "Inclusion_Criteria",
    "Exclusion_Criteria",
    "Decision",
    "Confidence",
    "Reason",
    "Evidence_Quote",
    "Primary_Decision",
    "Primary_Confidence",
    "Primary_Reason",
    "Primary_Evidence_Quote",
    "Verifier_Decision",
    "Verifier_Confidence",
    "Verifier_Reason",
    "Verifier_Evidence_Quote",
    "Agreement_Status",
    "Validation_Status",
    "Failure_Class",
    "Route_Used",
    "Execution_Origin",
    "Architecture_Version",
    "Prompt_Version",
    "Source_Row_Index",
    "Canonical_DOI",
    "Canonical_Source_URL",
    "Original_Model_Decision",
    "Manual_Decision",
    "Manual_Review_Status",
    "Manual_Review_Notes",
    "Final_Decision_Source",
]

# These fields are created by LitSync screening rather than supplied source
# metadata. Remaining columns keep their original order at the front of exports.
GENERATED_SCREENING_COLUMNS = {
    *CANONICAL_COLUMNS,
    "Original_Decision",
    "Decision_Source",
    "Exclusion_Reason",
    "Manual_Review_At",
    "Primary_Assessment_JSON",
    "Verifier_Assessment_JSON",
    "Evidence_JSON",
    "Criteria_JSON",
    "Uncertainty_JSON",
    "Escalated",
    "Validation_Errors",
    "Schema_Version",
    "Model_Tier",
    "Resource_Profile",
    "Model",
    "Processing_Seconds",
    "Original_Processing_Seconds",
    "Cache_Hit",
    "Runtime_Downgrades",
    "Layer_Trace_JSON",
    "Layer_Metrics_JSON",
    "Decision_Risk",
    "Triage_Basis",
    "RQ_Frame_ID",
    "RQ_Frame_Version",
    "RQ_Frame_Source",
    "RQ_Frame_Status",
    "RQ_Frame_Validation_Failures",
    "RQ_Group_Coverage_JSON",
    "Local_Profile",
    "Protocol_Model",
    "Deep_Model",
    "Edge_Model",
}

DOI_ALIASES = [
    "DOI",
    "Article DOI",
    "Document DOI",
    "Digital Object Identifier",
    "DOI Link",
    "DOI URL",
]
URL_ALIASES = [
    "URL",
    "Link",
    "Article URL",
    "Source URL",
    "Landing Page",
    "Landing Page URL",
    "Document URL",
    "Full Text URL",
    "Record URL",
    "Paper URL",
    "Web URL",
]
PLACEHOLDERS = {"", "-", "n/a", "na", "none", "null", "unavailable"}
SENSITIVE_COLUMN_PATTERN = re.compile(
    r"(?:password|passwd|api[\s_-]*key|authorization|bearer|cookie|session[\s_-]*token|access[\s_-]*token|refresh[\s_-]*token)",
    re.IGNORECASE,
)


class ScreeningExportError(ValueError):
    pass


_LOCKS_GUARD = threading.Lock()
_JOB_LOCKS: dict[str, threading.Lock] = {}


def validate_job_id(job_id: str) -> str:
    selected = str(job_id or "").strip()
    if not JOB_ID_PATTERN.fullmatch(selected):
        raise ScreeningExportError("Invalid screening job ID.")
    return selected


def resolve_persisted_output(output_root: str | Path, job_id: str) -> Path:
    selected = validate_job_id(job_id)
    root = Path(output_root).resolve()
    runs_root = (root / "runs").resolve()
    target = (runs_root / f"screened-{selected}.csv").resolve()
    try:
        target.relative_to(runs_root)
    except ValueError as exc:
        raise ScreeningExportError("Invalid screening job ID.") from exc
    if not target.is_file():
        raise FileNotFoundError(
            f"Persisted screening output for job '{selected}' was not found."
        )
    return target


def export_file_path(output_root: str | Path, job_id: str, export_name: str) -> Path:
    selected = validate_job_id(job_id)
    if export_name not in EXPORT_FILENAMES:
        raise ScreeningExportError("Unsupported screening export name.")
    root = (Path(output_root).resolve() / "exports" / selected).resolve()
    target = (root / EXPORT_FILENAMES[export_name]).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ScreeningExportError("Invalid screening export path.") from exc
    return target


def _normalized_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _alias_columns(columns: list[str], aliases: list[str]) -> list[str]:
    normalized: dict[str, list[str]] = {}
    for column in columns:
        normalized.setdefault(_normalized_column_name(column), []).append(column)
    selected: list[str] = []
    for alias in aliases:
        for column in normalized.get(_normalized_column_name(alias), []):
            if column not in selected:
                selected.append(column)
    return selected


def _first_value(row: pd.Series, columns: list[str]) -> str:
    for column in columns:
        value = str(row.get(column, "") or "").strip()
        if value.casefold() not in PLACEHOLDERS:
            return value
    return ""


def _canonical_doi(value: str) -> str:
    candidate = str(value or "").strip()
    if candidate.casefold() in PLACEHOLDERS:
        return ""
    candidate = re.sub(
        r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()
    if candidate.casefold() in PLACEHOLDERS:
        return ""
    return candidate


def _canonical_url(value: str) -> str:
    candidate = str(value or "").strip()
    if candidate.casefold() in PLACEHOLDERS:
        return ""
    if re.fullmatch(r"https?://(?:dx\.)?doi\.org/?", candidate, re.IGNORECASE):
        return ""
    return candidate


def _atomic_csv(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8-sig",
            lineterminator="\n",
        )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(value: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _job_metadata(output_root: Path, job_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    prisma = _read_json(output_root / "prisma" / f"{job_id}.json")
    latest = _read_json(output_root / "latest_screening.json")
    summary = latest.get("summary", {}) if latest.get("job_id") == job_id else {}
    return prisma, summary if isinstance(summary, dict) else {}


def _only_value(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        return ""
    values = [str(value).strip() for value in frame[column] if str(value).strip()]
    unique = list(dict.fromkeys(values))
    if len(unique) > 1:
        raise ScreeningExportError(f"Persisted output contains multiple {column} values.")
    return unique[0] if unique else ""


def _validate_rows(frame: pd.DataFrame, job_id: str) -> None:
    if frame.empty:
        raise ScreeningExportError("Persisted screening output is empty.")
    required = {"Decision", "Source_Row_Index"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ScreeningExportError(
            "Persisted screening output is missing required columns: " + ", ".join(missing)
        )
    decisions = frame["Decision"].astype(str).str.strip().str.upper()
    invalid = sorted(set(decisions).difference(VALID_DECISIONS))
    if invalid:
        raise ScreeningExportError("Persisted output contains invalid screening decisions.")
    identifiers = frame["Source_Row_Index"].astype(str).str.strip()
    if identifiers.eq("").any() or identifiers.duplicated().any():
        raise ScreeningExportError("Source_Row_Index values must be nonempty and unique.")
    if "Job_ID" in frame.columns:
        stored_jobs = {
            str(value).strip() for value in frame["Job_ID"] if str(value).strip()
        }
        if stored_jobs and stored_jobs != {job_id}:
            raise ScreeningExportError("Persisted output does not belong to the requested job.")
    validation = frame.get("Validation_Status", pd.Series("", index=frame.index)).astype(str).str.casefold()
    origin = frame.get("Execution_Origin", pd.Series("", index=frame.index)).astype(str).str.casefold()
    final_source = frame.get("Final_Decision_Source", pd.Series("", index=frame.index)).astype(str).str.casefold()
    decision_source = frame.get("Decision_Source", pd.Series("", index=frame.index)).astype(str).str.casefold()
    human_reviewed = final_source.eq("human_review") | decision_source.isin({"manual_review", "human_review"})
    unsafe = decisions.isin({"KEEP", "REJECT"}) & (
        validation.eq("safe_fallback") | origin.eq("technical_fallback")
    ) & ~human_reviewed
    if unsafe.any():
        raise ScreeningExportError(
            "Unsafe definitive fallback decisions prevent export; review the persisted job."
        )
    protocol_id = _only_value(frame, "Protocol_ID")
    review_protocol_id = _only_value(frame, "Review_Protocol_ID")
    if protocol_id and review_protocol_id and protocol_id != review_protocol_id:
        raise ScreeningExportError(
            "Protocol_ID and Review_Protocol_ID do not match for this job."
        )


def _source_columns(frame: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in frame.columns
        if str(column) not in GENERATED_SCREENING_COLUMNS
        and not SENSITIVE_COLUMN_PATTERN.search(str(column))
    ]


def _assessment_field(raw: Any, field: str) -> str:
    try:
        parsed = json.loads(str(raw or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    if not isinstance(parsed, dict):
        return ""
    value = parsed.get(field, "")
    return str(value).strip() if value is not None else ""


def _effective_decision_source(row: pd.Series) -> str:
    explicit = str(row.get("Final_Decision_Source", "") or "").strip()
    if explicit:
        return explicit
    source = str(row.get("Decision_Source", "") or "").casefold()
    if source in {"manual_review", "human_review"} or str(row.get("Manual_Decision", "")).strip():
        return "human_review"
    if str(row.get("Decision", "")).upper() == "MAYBE" and (
        str(row.get("Validation_Status", "")).casefold() == "safe_fallback"
        or str(row.get("Execution_Origin", "")).casefold() == "technical_fallback"
    ):
        return "model_safe_maybe"
    return "model_validated"


def _review_priority(row: pd.Series) -> tuple[int, str, str, str]:
    failure = str(row.get("Failure_Class", "") or "").casefold()
    validation = str(row.get("Validation_Status", "") or "").casefold()
    agreement = str(row.get("Agreement_Status", "") or "").casefold()
    route = str(row.get("Route_Used", "") or "").casefold()
    has_provisional = bool(
        str(row.get("Primary_Decision", "")).strip()
        or str(row.get("Verifier_Decision", "")).strip()
    )
    if "validation_contradiction" in failure or "contradiction" in validation:
        return 1, "validation_contradiction", "Validation contradiction requires adjudication.", "Review title and abstract against all criteria."
    if "browser_or_transport_failure" in failure and has_provisional:
        return 2, "transport_failure_with_provisional_assessment", "Transport failed after a provisional assessment.", "Retry model assessment when browser transport is available."
    if failure or validation == "safe_fallback" or route == "safe_fallback":
        return 3, "technical_safe_fallback", "Technical validation did not produce a reusable decision.", "Review title and abstract against all criteria."
    if "disagree" in agreement or "disagreement" in agreement:
        return 4, "independent_review_disagreement", "Independent assessments disagree.", "Resolve disagreement between independent assessments."
    if route == "missing_abstract" or "missing abstract" in str(row.get("Reason", "")).casefold():
        return 6, "missing_or_insufficient_evidence", "The supplied source evidence is incomplete.", "Obtain the missing abstract or full text."
    return 5, "semantic_maybe", "A valid assessment remained semantically uncertain.", "Check whether exclusion evidence is substantive or incidental."


def _prepare_frame(
    frame: pd.DataFrame,
    *,
    job_id: str,
    prisma: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    result = frame.copy()
    result["Decision"] = result["Decision"].astype(str).str.strip().str.upper()
    protocol_inputs = prisma.get("protocol_inputs", {})
    if not isinstance(protocol_inputs, dict):
        protocol_inputs = {}
    protocol_id = _only_value(result, "Protocol_ID")
    review_protocol_id = _only_value(result, "Review_Protocol_ID") or protocol_id
    values = {
        "Job_ID": job_id,
        "Protocol_ID": protocol_id,
        "Review_Protocol_ID": review_protocol_id,
        "Research_Question": str(protocol_inputs.get("research_question") or ""),
        "Research_Context": str(protocol_inputs.get("research_context") or ""),
        "Inclusion_Criteria": str(protocol_inputs.get("inclusion_criteria") or ""),
        "Exclusion_Criteria": str(protocol_inputs.get("exclusion_criteria") or ""),
    }
    for column, value in values.items():
        result[column] = value

    doi_columns = _alias_columns(list(result.columns), DOI_ALIASES)
    url_columns = _alias_columns(list(result.columns), URL_ALIASES)
    result["Canonical_DOI"] = result.apply(
        lambda row: _canonical_doi(_first_value(row, doi_columns)), axis=1
    )
    result["Canonical_Source_URL"] = result.apply(
        lambda row: _canonical_url(_first_value(row, url_columns))
        or (f"https://doi.org/{row['Canonical_DOI']}" if row["Canonical_DOI"] else ""),
        axis=1,
    )
    primary_json = result.get("Primary_Assessment_JSON", pd.Series("", index=result.index))
    verifier_json = result.get("Verifier_Assessment_JSON", pd.Series("", index=result.index))
    for prefix, json_values in (("Primary", primary_json), ("Verifier", verifier_json)):
        reason_column = f"{prefix}_Reason"
        evidence_column = f"{prefix}_Evidence_Quote"
        if reason_column not in result.columns:
            result[reason_column] = json_values.map(lambda value: _assessment_field(value, "reason"))
        if evidence_column not in result.columns:
            result[evidence_column] = json_values.map(
                lambda value: _assessment_field(value, "evidence_quote")
            )
    result["Original_Model_Decision"] = result.apply(
        lambda row: str(
            row.get("Original_Model_Decision")
            or row.get("Original_Decision")
            or row.get("Primary_Decision")
            or row.get("Decision")
            or ""
        ).strip(),
        axis=1,
    )
    result["Manual_Decision"] = result.apply(
        lambda row: str(row.get("Manual_Decision") or (
            row.get("Decision")
            if str(row.get("Decision_Source", "")).casefold() in {"manual_review", "human_review"}
            else ""
        )).strip(),
        axis=1,
    )
    result["Manual_Review_Status"] = result.apply(
        lambda row: str(row.get("Manual_Review_Status") or (
            "reviewed" if row["Manual_Decision"] else (
                "pending" if row["Decision"] == "MAYBE" else "not_required"
            )
        )), axis=1,
    )
    result["Manual_Review_Notes"] = result.apply(
        lambda row: str(
            row.get("Manual_Review_Notes") or row.get("Exclusion_Reason") or ""
        ), axis=1,
    )
    result["Final_Decision_Source"] = result.apply(_effective_decision_source, axis=1)

    source_columns = _source_columns(frame)
    remaining_generated = [
        str(column) for column in frame.columns
        if str(column) not in source_columns
        and str(column) not in CANONICAL_COLUMNS
        and not SENSITIVE_COLUMN_PATTERN.search(str(column))
    ]
    final_columns = source_columns + CANONICAL_COLUMNS + remaining_generated
    for column in final_columns:
        if column not in result.columns:
            result[column] = ""
    result = result.loc[:, list(dict.fromkeys(final_columns))]
    missing_protocol_fields = [
        column for column in (
            "Protocol_ID", "Review_Protocol_ID", "Research_Question",
            "Research_Context", "Inclusion_Criteria", "Exclusion_Criteria",
        ) if not str(result.iloc[0].get(column, "")).strip()
    ]
    return result, source_columns, missing_protocol_fields


def _queue_frame(maybe: pd.DataFrame) -> pd.DataFrame:
    queue = maybe.copy()
    details = queue.apply(_review_priority, axis=1)
    queue["Review_Priority"] = [item[0] for item in details]
    queue["Review_Priority_Group"] = [item[1] for item in details]
    queue["Review_Reason"] = [item[2] for item in details]
    queue["Suggested_Reviewer_Action"] = [item[3] for item in details]
    queue["_disagreement"] = queue["Agreement_Status"].astype(str).str.casefold().str.contains("disagree").astype(int)
    queue["_confidence"] = pd.to_numeric(queue["Confidence"], errors="coerce").fillna(-1)
    queue["_source_numeric"] = pd.to_numeric(queue["Source_Row_Index"], errors="coerce")
    queue["_source_text"] = queue["Source_Row_Index"].astype(str)
    queue = queue.sort_values(
        ["Review_Priority", "_disagreement", "_confidence", "_source_numeric", "_source_text"],
        ascending=[True, False, False, True, True],
        kind="stable",
        na_position="last",
    )
    return queue.drop(columns=["_disagreement", "_confidence", "_source_numeric", "_source_text"])


def _summary(
    frame: pd.DataFrame,
    *,
    job_id: str,
    output_root: Path,
    source_output: Path,
    prisma: dict[str, Any],
    job_summary: dict[str, Any],
    missing_protocol_fields: list[str],
    file_counts: dict[str, int],
) -> dict[str, Any]:
    decisions = frame["Decision"]
    validation = frame["Validation_Status"].astype(str).str.casefold()
    failures = frame["Failure_Class"].astype(str).str.casefold()
    origin = frame["Execution_Origin"].astype(str).str.casefold()
    final_source = frame["Final_Decision_Source"].astype(str).str.casefold()
    protocol_inputs = prisma.get("protocol_inputs", {})
    if not isinstance(protocol_inputs, dict):
        protocol_inputs = {}
    relative_source = source_output.relative_to(output_root).as_posix()
    summary = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "job_id": job_id,
        "protocol_id": str(frame.iloc[0]["Protocol_ID"]),
        "review_protocol_id": str(frame.iloc[0]["Review_Protocol_ID"]),
        "architecture_version": str(frame.iloc[0]["Architecture_Version"]),
        "prompt_version": str(frame.iloc[0]["Prompt_Version"]),
        "research_question": str(protocol_inputs.get("research_question") or ""),
        "research_context": str(protocol_inputs.get("research_context") or ""),
        "inclusion_criteria": str(protocol_inputs.get("inclusion_criteria") or ""),
        "exclusion_criteria": str(protocol_inputs.get("exclusion_criteria") or ""),
        "total_count": int(len(frame)),
        "keep_count": int(decisions.eq("KEEP").sum()),
        "maybe_count": int(decisions.eq("MAYBE").sum()),
        "reject_count": int(decisions.eq("REJECT").sum()),
        "validated_count": int(validation.eq("validated").sum()),
        "safe_fallback_count": int(validation.eq("safe_fallback").sum()),
        "transport_failure_count": int(failures.str.contains("browser_or_transport_failure").sum()),
        "validation_contradiction_count": int(failures.str.contains("validation_contradiction").sum()),
        "unsafe_definitive_fallback_count": int((decisions.isin({"KEEP", "REJECT"}) & (validation.eq("safe_fallback") | origin.eq("technical_fallback")) & ~final_source.eq("human_review")).sum()),
        "missing_abstract_count": int(
            job_summary.get("missing_abstract_count")
            or frame["Route_Used"].astype(str).str.casefold().eq("missing_abstract").sum()
        ),
        "runtime_seconds": job_summary.get("runtime_seconds"),
        "primary_batch_size": job_summary.get("primary_batch_size"),
        "primary_batches_submitted": job_summary.get("primary_batches_submitted"),
        "primary_batches_completed": job_summary.get("primary_batches_completed"),
        "verification_batches_submitted": job_summary.get("verification_batches_submitted"),
        "verification_batches_completed": job_summary.get("verification_batches_completed"),
        "primary_papers_requested": job_summary.get("primary_papers_requested"),
        "verification_papers_requested": job_summary.get("verification_papers_requested"),
        "retry_count": job_summary.get("retry_count"),
        "resumed_count": job_summary.get("resumed_count"),
        "fresh_primary_count": job_summary.get("fresh_primary_count"),
        "browser_context_started": job_summary.get("browser_context_started"),
        "pages_opened": job_summary.get("pages_opened"),
        "pages_closed": job_summary.get("pages_closed"),
        "peak_simultaneous_tabs": job_summary.get("peak_simultaneous_tabs"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_output_path": relative_source,
        "missing_protocol_fields": missing_protocol_fields,
        "generated_from_latest_persisted_decisions": True,
        "export_files": {
            name: {"filename": EXPORT_FILENAMES[name], "row_count": int(count)}
            for name, count in file_counts.items()
            if name != "summary"
        },
    }
    return summary


def generate_screening_exports(output_root: str | Path, job_id: str) -> dict[str, Any]:
    selected = validate_job_id(job_id)
    root = Path(output_root).resolve()
    with _LOCKS_GUARD:
        lock = _JOB_LOCKS.setdefault(selected, threading.Lock())
    with lock:
        source_output = resolve_persisted_output(root, selected)
        try:
            raw = pd.read_csv(
                source_output, dtype=str, keep_default_na=False, encoding="utf-8-sig"
            )
        except (OSError, UnicodeError, pd.errors.ParserError) as exc:
            raise ScreeningExportError("Persisted screening output could not be read.") from exc
        _validate_rows(raw, selected)
        prisma, job_summary = _job_metadata(root, selected)
        frame, source_columns, missing_protocol_fields = _prepare_frame(
            raw, job_id=selected, prisma=prisma
        )
        _validate_rows(frame, selected)
        partitions: dict[str, pd.DataFrame] = {
            "all": frame,
            "keep": frame.loc[frame["Decision"] == "KEEP"],
            "reject": frame.loc[frame["Decision"] == "REJECT"],
        }
        partitions["maybe"] = frame.loc[frame["Decision"] == "MAYBE"]
        partitions["review_queue"] = _queue_frame(partitions["maybe"])
        file_counts = {name: len(value) for name, value in partitions.items()}
        for name, value in partitions.items():
            _atomic_csv(value, export_file_path(root, selected, name))
        summary = _summary(
            frame,
            job_id=selected,
            output_root=root,
            source_output=source_output,
            prisma=prisma,
            job_summary=job_summary,
            missing_protocol_fields=missing_protocol_fields,
            file_counts=file_counts,
        )
        _atomic_json(summary, export_file_path(root, selected, "summary"))
        return {
            "job_id": selected,
            "generated": True,
            "counts": file_counts,
            "export_names": [*partitions, "summary"],
            "source_columns": source_columns,
            "summary": summary,
        }
