from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd


GOLD_LABELS = {"KEEP", "REJECT", "UNSURE"}
SAMPLE_VERSION = "gold-pilot-v2-bound"
BINDING_VERSION = "gold-screening-binding-v1"
CSV_BINDING_VERSION = "gold-csv-job-id-v1"
DEFAULT_SAMPLE_SIZE = 60
BOOTSTRAP_ITERATIONS = 2000
DEFAULT_SAMPLING_STRATA = {"KEEP": 0.40, "REJECT": 0.40, "MAYBE": 0.20}
UNBOUND_CSV_MESSAGE = (
    "This validation CSV is not bound to a screening job. "
    "Regenerate it from the intended completed screening run."
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _source_id(value: Any) -> str:
    text = _text(value)
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text


def _excel_safe(value: Any) -> str:
    text = _text(value)
    return "'" + text if text.startswith(("=", "+", "-", "@")) else text


def screening_content_fingerprint(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("Persisted screening output is empty.")
    excluded = {"job_id", "screening_job_id"}
    columns = sorted({
        str(key)
        for row in rows
        for key in row
        if str(key).lower() not in excluded
    })
    canonical_rows = [
        [_text(row.get(column)) for column in columns]
        for row in sorted(
            rows,
            key=lambda item: _source_id(item.get("Source_Row_Index")),
        )
    ]
    payload = {
        "binding_version": BINDING_VERSION,
        "columns": columns,
        "rows": canonical_rows,
    }
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _seed(protocol_id: str) -> int:
    digest = hashlib.sha256(f"{SAMPLE_VERSION}:{protocol_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _normalize_sampling_strata(value: dict[str, float] | None) -> dict[str, float]:
    supplied = value or DEFAULT_SAMPLING_STRATA
    if set(supplied) != {"KEEP", "REJECT", "MAYBE"}:
        raise ValueError("sampling_strata must contain KEEP, REJECT, and MAYBE")
    parsed = {decision: float(weight) for decision, weight in supplied.items()}
    if any(weight < 0 for weight in parsed.values()) or sum(parsed.values()) <= 0:
        raise ValueError("sampling_strata weights must be non-negative with a positive total")
    total = sum(parsed.values())
    return {decision: weight / total for decision, weight in parsed.items()}


def _allocate_quotas(
    rows: list[dict[str, Any]], target: int, sampling_strata: dict[str, float] | None = None
) -> dict[str, int]:
    strata = _normalize_sampling_strata(sampling_strata)
    available = {
        decision: sum(_text(row.get("Decision")).upper() == decision for row in rows)
        for decision in ("KEEP", "REJECT", "MAYBE")
    }
    quotas = {decision: int(target * strata[decision]) for decision in strata}
    for decision in sorted(strata, key=lambda item: target * strata[item] - quotas[item], reverse=True):
        if sum(quotas.values()) >= target:
            break
        quotas[decision] += 1
    selected = {decision: min(quotas[decision], available[decision]) for decision in quotas}
    remaining = target - sum(selected.values())
    for decision in sorted(available, key=lambda item: available[item] - selected[item], reverse=True):
        extra = min(remaining, available[decision] - selected[decision])
        selected[decision] += extra
        remaining -= extra
        if remaining == 0:
            break
    return selected


def _full_run_safety(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = total_evidence = invalid_evidence = invalid_definitive = 0
    validated = repairs = 0
    for row in rows:
        decision = _text(row.get("Decision")).upper()
        status = _text(row.get("Validation_Status")).lower()
        validated += int(status == "validated")
        repairs += int(_text(row.get("Escalated")).lower() in {"1", "true"})
        invalid_definitive += int(decision in {"KEEP", "REJECT"} and status != "validated")
        try:
            evidence = json.loads(_text(row.get("Evidence_JSON")) or "[]")
            if not isinstance(evidence, list):
                raise ValueError("evidence is not a list")
        except (json.JSONDecodeError, TypeError, ValueError):
            invalid_evidence += 1
            evidence = []
        title = _text(row.get("Title"))
        abstract = _text(row.get("Abstract"))
        for span in evidence:
            total_evidence += 1
            source = title if span.get("source") == "title" else abstract
            quote = _text(span.get("quote"))
            exact += int(bool(quote) and quote in source)
    total = len(rows)
    return {
        "total_rows": total,
        "structurally_validated_rate": round(validated / total, 4) if total else 0.0,
        "repair_call_rate": round(repairs / total, 4) if total else 0.0,
        "exact_evidence_rate": round(exact / total_evidence, 4) if total_evidence else 1.0,
        "invalid_evidence_payloads": invalid_evidence,
        "invalid_definitive_count": invalid_definitive,
    }


def create_blinded_sample(
    rows: list[dict[str, Any]],
    research_question: str,
    output_dir: str | Path,
    job_id: str,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    manifest_root: str | Path | None = None,
    sampling_strata: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No screening results are available. Finish screening first.")
    question = _text(research_question)
    if not question:
        raise ValueError("The research question is required for human labeling.")
    bound_job_id = _text(job_id)
    if not bound_job_id:
        raise ValueError("Select a completed screening job before creating Gold Validation.")
    protocol_ids = {_text(row.get("Protocol_ID")) for row in rows if _text(row.get("Protocol_ID"))}
    if len(protocol_ids) != 1:
        raise ValueError("Screening results must belong to exactly one review protocol.")
    protocol_id = next(iter(protocol_ids))
    prompt_versions = {
        _text(row.get("Prompt_Version"))
        for row in rows
        if _text(row.get("Prompt_Version"))
    }
    if len(prompt_versions) != 1:
        raise ValueError(
            "Screening results must belong to exactly one assessment prompt version."
        )
    assessment_prompt_version = next(iter(prompt_versions))
    content_fingerprint = screening_content_fingerprint(rows)
    decisions = {_text(row.get("Decision")).upper() for row in rows}
    unknown_decisions = sorted(decisions - {"KEEP", "REJECT", "MAYBE"})
    if unknown_decisions:
        raise ValueError("Unsupported screening decisions: " + ", ".join(unknown_decisions))
    source_ids = [_source_id(row.get("Source_Row_Index")) for row in rows]
    if any(not value for value in source_ids) or len(source_ids) != len(set(source_ids)):
        raise ValueError("Screening results require unique non-empty Source_Row_Index values.")
    target = min(max(1, int(sample_size)), len(rows))
    normalized_strata = _normalize_sampling_strata(sampling_strata)
    quotas = _allocate_quotas(rows, target, normalized_strata)
    rng = random.Random(_seed(protocol_id))

    selected: list[dict[str, Any]] = []
    for decision in ("KEEP", "REJECT", "MAYBE"):
        stratum = sorted(
            (row for row in rows if _text(row.get("Decision")).upper() == decision),
            key=lambda row: _source_id(row.get("Source_Row_Index")),
        )
        selected.extend(rng.sample(stratum, quotas[decision]))
    rng.shuffle(selected)

    selected_ids = sorted(_source_id(row.get("Source_Row_Index")) for row in selected)
    sample_configuration = {
        "requested_sample_size": int(sample_size),
        "effective_sample_size": target,
        "sampling_strata": normalized_strata,
    }
    set_payload = json.dumps({
        "sample_version": SAMPLE_VERSION,
        "binding_version": BINDING_VERSION,
        "csv_binding_version": CSV_BINDING_VERSION,
        "job_id": bound_job_id,
        "protocol_id": protocol_id,
        "assessment_prompt_version": assessment_prompt_version,
        "screening_content_fingerprint": content_fingerprint,
        "research_question": question,
        "sample_configuration": sample_configuration,
        "selected_source_row_ids": selected_ids,
    }, ensure_ascii=False, sort_keys=True)
    set_id = hashlib.sha256(set_payload.encode("utf-8")).hexdigest()[:16]
    directory = Path(output_dir) / "gold_validation"
    directory.mkdir(parents=True, exist_ok=True)
    label_path = directory / f"{set_id}_labels.csv"
    private_directory = Path(manifest_root or output_dir) / "gold_validation"
    private_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = private_directory / f"{set_id}_manifest.json"

    population = {
        decision: sum(_text(row.get("Decision")).upper() == decision for row in rows)
        for decision in ("KEEP", "REJECT", "MAYBE")
    }
    sampled = {decision: quotas[decision] for decision in quotas}
    manifest_rows = []
    blinded_rows = []
    for row in selected:
        decision = _text(row.get("Decision")).upper()
        source_id = _source_id(row.get("Source_Row_Index"))
        weight = population[decision] / sampled[decision] if sampled[decision] else 0.0
        title = _text(row.get("Title"))
        abstract = _text(row.get("Abstract"))
        immutable_blinded = {
            "Screening_Job_ID": bound_job_id,
            "Validation_Set_ID": set_id,
            "Source_Row_Index": source_id,
            "Research_Question": _excel_safe(question),
            "Title": _excel_safe(title),
            "Abstract": _excel_safe(abstract),
            "Year": _text(row.get("Year")),
            "DOI": _text(row.get("DOI")),
        }
        manifest_rows.append({
            "source_row_index": source_id,
            "model_decision": decision,
            "sampling_weight": weight,
            "title": title,
            "abstract": abstract,
            "label_title": _excel_safe(title),
            "label_abstract": _excel_safe(abstract),
            "validation_status": _text(row.get("Validation_Status")),
            "escalated": _text(row.get("Escalated")).lower() in {"1", "true"},
            "immutable_blinded": immutable_blinded,
        })
        blinded_rows.append({
            **immutable_blinded,
            "Gold_Decision": "",
            "Reviewer_Notes": "",
        })

    pd.DataFrame(blinded_rows).to_csv(label_path, index=False, encoding="utf-8-sig")
    manifest = {
        "sample_version": SAMPLE_VERSION,
        "binding_version": BINDING_VERSION,
        "csv_binding_version": CSV_BINDING_VERSION,
        "validation_set_id": set_id,
        "job_id": bound_job_id,
        "protocol_id": protocol_id,
        "assessment_prompt_version": assessment_prompt_version,
        "screening_content_fingerprint": content_fingerprint,
        "research_question": question,
        "sample_configuration": sample_configuration,
        "population_counts": population,
        "sample_counts": sampled,
        "sampling_strata": normalized_strata,
        "sampling_note": "Diagnostic sampling only; strata never modify screening decisions.",
        "sample_size": len(selected),
        "rows": manifest_rows,
        "full_run_safety": _full_run_safety(rows),
    }
    serialized_manifest = json.dumps(manifest, ensure_ascii=False, indent=2)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(
                "An existing validation manifest has the same ID but different binding content."
            )
    else:
        manifest_path.write_text(serialized_manifest, encoding="utf-8")
    return {
        "validation_set_id": set_id,
        "job_id": bound_job_id,
        "screening_content_fingerprint": content_fingerprint,
        "sample_size": len(selected),
        "sample_counts": sampled,
        "population_counts": population,
        "label_path": str(label_path),
        "manifest_path": str(manifest_path),
    }


def _quality_metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    relevant_weight = retrieved_weight = false_reject_weight = 0.0
    keep_weight = correct_keep_weight = 0.0
    for record in records:
        human = record["human_decision"]
        model = record["model_decision"]
        weight = float(record["sampling_weight"])
        if human == "KEEP":
            relevant_weight += weight
            retrieved_weight += weight * int(model in {"KEEP", "MAYBE"})
            false_reject_weight += weight * int(model == "REJECT")
        if model == "KEEP":
            keep_weight += weight
            correct_keep_weight += weight * int(human == "KEEP")
    return {
        "relevant_recall_keep_or_maybe": retrieved_weight / relevant_weight if relevant_weight else None,
        "false_reject_rate": false_reject_weight / relevant_weight if relevant_weight else None,
        "definitive_keep_precision": correct_keep_weight / keep_weight if keep_weight else None,
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _bootstrap_intervals(
    records: list[dict[str, Any]], set_id: str, iterations: int = BOOTSTRAP_ITERATIONS
) -> dict[str, dict[str, float | None]]:
    strata = {
        decision: [record for record in records if record["model_decision"] == decision]
        for decision in ("KEEP", "REJECT", "MAYBE")
    }
    rng = random.Random(_seed(set_id))
    samples = {name: [] for name in _quality_metrics(records)}
    for _ in range(iterations):
        draw = []
        for stratum in strata.values():
            if stratum:
                draw.extend(rng.choice(stratum) for _ in range(len(stratum)))
        for name, value in _quality_metrics(draw).items():
            if value is not None:
                samples[name].append(value)
    return {
        name: {
            "lower": round(_percentile(values, 0.025), 4) if values else None,
            "upper": round(_percentile(values, 0.975), 4) if values else None,
        }
        for name, values in samples.items()
    }


def evaluate_completed_labels(
    completed_csv: str | Path,
    manifest_root: str | Path,
    job_id: str,
    screening_rows: list[dict[str, Any]],
    report_output_root: str | Path | None = None,
) -> dict[str, Any]:
    requested_job_id = _text(job_id)
    if not requested_job_id:
        raise ValueError(
            "Select a completed screening job before evaluating Gold Validation."
        )
    frame = pd.read_csv(completed_csv, dtype=str, keep_default_na=False)
    if "Screening_Job_ID" not in frame.columns:
        raise ValueError(UNBOUND_CSV_MESSAGE)
    uploaded_job_ids = [_text(value) for value in frame["Screening_Job_ID"]]
    unique_job_ids = set(uploaded_job_ids)
    if not uploaded_job_ids or "" in unique_job_ids or len(unique_job_ids) != 1:
        raise ValueError(UNBOUND_CSV_MESSAGE)
    uploaded_job_id = next(iter(unique_job_ids))
    if uploaded_job_id != requested_job_id:
        raise ValueError(
            f"This validation CSV belongs to screening job '{uploaded_job_id}', "
            f"not '{requested_job_id}'."
        )
    required = {"Validation_Set_ID", "Source_Row_Index", "Title", "Abstract", "Gold_Decision"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError("Completed validation CSV is missing columns: " + ", ".join(missing_columns))
    set_ids = {_text(value) for value in frame["Validation_Set_ID"] if _text(value)}
    if len(set_ids) != 1:
        raise ValueError("The completed CSV must contain exactly one Validation_Set_ID.")
    set_id = next(iter(set_ids))
    if not re.fullmatch(r"[0-9a-f]{16}", set_id):
        raise ValueError("Invalid Validation_Set_ID format.")
    if any(_text(value) != set_id for value in frame["Validation_Set_ID"]):
        raise ValueError("Every uploaded row must contain the same Validation_Set_ID.")
    manifest_path = Path(manifest_root) / "gold_validation" / f"{set_id}_manifest.json"
    if not manifest_path.exists():
        raise ValueError("Unknown validation set. Generate the sample on this LitSync installation first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_binding = {
        "binding_version", "csv_binding_version", "job_id", "assessment_prompt_version",
        "screening_content_fingerprint",
    }
    if (
        manifest.get("binding_version") != BINDING_VERSION
        or manifest.get("csv_binding_version") != CSV_BINDING_VERSION
        or not required_binding.issubset(manifest)
    ):
        raise ValueError(UNBOUND_CSV_MESSAGE)
    bound_job_id = _text(manifest.get("job_id"))
    if not bound_job_id or bound_job_id != uploaded_job_id:
        raise ValueError(UNBOUND_CSV_MESSAGE)
    protocol_ids = {
        _text(row.get("Protocol_ID"))
        for row in screening_rows
        if _text(row.get("Protocol_ID"))
    }
    prompt_versions = {
        _text(row.get("Prompt_Version"))
        for row in screening_rows
        if _text(row.get("Prompt_Version"))
    }
    if (
        protocol_ids != {_text(manifest.get("protocol_id"))}
        or prompt_versions != {_text(manifest.get("assessment_prompt_version"))}
    ):
        raise ValueError(
            "The validation set protocol or assessment prompt version does not "
            f"match screening job '{requested_job_id}'. Regenerate the validation CSV."
        )
    current_fingerprint = screening_content_fingerprint(screening_rows)
    if current_fingerprint != manifest.get("screening_content_fingerprint"):
        raise ValueError(
            f"Screening job '{requested_job_id}' has changed since this validation "
            "set was created. Regenerate the validation CSV."
        )
    expected = {row["source_row_index"]: row for row in manifest["rows"]}

    source_ids = [_source_id(value) for value in frame["Source_Row_Index"]]
    duplicates = sorted({value for value in source_ids if source_ids.count(value) > 1})
    if duplicates:
        raise ValueError("Duplicate Source_Row_Index values: " + ", ".join(duplicates))
    unknown = sorted(set(source_ids) - set(expected))
    if unknown:
        raise ValueError(
            "Added source rows do not belong to this validation set: "
            + ", ".join(unknown)
        )
    removed = sorted(set(expected) - set(source_ids))
    if removed:
        raise ValueError(
            "Removed source rows are missing from this validation set: "
            + ", ".join(removed)
        )

    uploaded: dict[str, dict[str, Any]] = {}
    invalid_labels = []
    for (_, row), source_id in zip(frame.iterrows(), source_ids):
        expected_row = expected[source_id]
        immutable = expected_row.get("immutable_blinded")
        if not isinstance(immutable, dict):
            raise ValueError(UNBOUND_CSV_MESSAGE)
        altered = [
            column for column, expected_value in immutable.items()
            if _text(row.get(column)) != _text(expected_value)
        ]
        if altered:
            raise ValueError(
                f"Immutable blinded content was altered for source row {source_id}: "
                + ", ".join(altered)
            )
        label = _text(row["Gold_Decision"]).upper()
        if label and label not in GOLD_LABELS:
            invalid_labels.append(f"{source_id}:{label}")
        uploaded[source_id] = {
            "label": label,
            "notes": _text(row.get("Reviewer_Notes", "")),
        }
    if invalid_labels:
        raise ValueError("Invalid Gold_Decision values: " + ", ".join(invalid_labels))

    records = []
    missing_rows = []
    blank_rows = []
    unsure_rows = []
    for source_id, expected_row in expected.items():
        supplied = uploaded.get(source_id)
        if supplied is None:
            missing_rows.append(source_id)
            continue
        label = supplied["label"]
        if not label:
            blank_rows.append(source_id)
            continue
        if label == "UNSURE":
            unsure_rows.append({
                "source_row_index": source_id,
                "title": expected_row["title"],
                "notes": supplied["notes"],
            })
            continue
        records.append({
            **expected_row,
            "human_decision": label,
            "reviewer_notes": supplied["notes"],
        })

    metrics = _quality_metrics(records)
    confusion = {
        model: {
            human: sum(
                record["model_decision"] == model and record["human_decision"] == human
                for record in records
            )
            for human in ("KEEP", "REJECT")
        }
        for model in ("KEEP", "MAYBE", "REJECT")
    }
    false_keeps = [
        {"source_row_index": record["source_row_index"], "title": record["title"]}
        for record in records
        if record["model_decision"] == "KEEP" and record["human_decision"] == "REJECT"
    ]
    false_rejects = [
        {"source_row_index": record["source_row_index"], "title": record["title"]}
        for record in records
        if record["model_decision"] == "REJECT" and record["human_decision"] == "KEEP"
    ]
    rounded_metrics = {
        name: round(value, 4) if value is not None else None for name, value in metrics.items()
    }
    comparisons = {
        "recall_target_0_95": metrics["relevant_recall_keep_or_maybe"] is not None
        and metrics["relevant_recall_keep_or_maybe"] >= 0.95,
        "false_reject_target_0_05": metrics["false_reject_rate"] is not None
        and metrics["false_reject_rate"] <= 0.05,
        "keep_precision_target_0_85": metrics["definitive_keep_precision"] is not None
        and metrics["definitive_keep_precision"] >= 0.85,
    }
    report = {
        "status": "provisional_single_reviewer",
        "label": "Provisional—single reviewer",
        "validation_set_id": set_id,
        "binding_version": BINDING_VERSION,
        "job_id": requested_job_id,
        "protocol_id": manifest["protocol_id"],
        "assessment_prompt_version": manifest["assessment_prompt_version"],
        "screening_content_fingerprint": manifest["screening_content_fingerprint"],
        "sample_size": int(manifest["sample_size"]),
        "resolved_labels": len(records),
        "unsure_count": len(unsure_rows),
        "blank_label_count": len(blank_rows),
        "missing_row_count": len(missing_rows),
        "resolved_coverage": round(len(records) / manifest["sample_size"], 4),
        "metrics": rounded_metrics,
        "confidence_intervals_95": _bootstrap_intervals(records, set_id),
        "informational_target_comparisons": comparisons,
        "confusion_matrix": confusion,
        "false_keeps": false_keeps,
        "false_rejects": false_rejects,
        "unsure_rows": unsure_rows,
        "blank_source_rows": blank_rows,
        "missing_source_rows": missing_rows,
        "full_run_safety": manifest["full_run_safety"],
        "warning": "Diagnostic only. This single-reviewer pilot is not a production release gate.",
    }
    report_directory = Path(report_output_root or manifest_root) / "gold_validation"
    report_directory.mkdir(parents=True, exist_ok=True)
    report_path = report_directory / f"{set_id}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
