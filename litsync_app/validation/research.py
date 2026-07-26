from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import random
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


SCHEMA_VERSION = "research-validation-v1"
FROZEN_GEMINI_VERSION = "gemini-web-batched-v2.4"
VALID_HUMAN_LABELS = {"KEEP", "MAYBE", "REJECT", "ABSTAIN"}
FINAL_GOLD_LABELS = {"KEEP", "MAYBE", "REJECT"}
VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}
FORBIDDEN_SCREENING_FIELDS = {
    "gold_decision", "human_decision", "expected_decision", "expected_label",
    "reviewer_id", "reviewer_rationale", "reviewer_notes", "adjudication_rationale",
    "model_decision", "sampling_stratum", "control_group", "label",
}
ROOT_CAUSES = {
    "browser_transport_failure", "evidence_grounding_failure", "verification_failure",
    "model_nondeterminism", "semantic_decision_error", "possible_protocol_error",
    "gold_adjudication_error",
}

APPROVED_DIAGNOSTIC_FIELDS = {
    "event", "submission_number", "stage", "retry_number", "outcome",
    "recovery_action", "attempt_duration_ms", "response_selector",
    "response_container_count", "response_state", "generation_detected",
    "timeout_stage", "fallback_reason",
}
FORBIDDEN_DIAGNOSTIC_FIELDS = {
    "prompt", "question", "research_question", "title", "abstract",
    "response", "raw_response", "response_text", "content", "content_hash",
}


def validate_diagnostics(path: str | Path) -> dict[str, Any]:
    """Reject diagnostic streams containing research content."""
    events = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        unsafe = (set(event) - APPROVED_DIAGNOSTIC_FIELDS) | (
            set(event) & FORBIDDEN_DIAGNOSTIC_FIELDS
        )
        if unsafe:
            raise ValueError(f"unsafe Gemini Web diagnostic fields: {sorted(unsafe)}")
        events.append(event)
    return {"event_count": len(events), "approved_fields_only": True}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _source_key(value: Any) -> str:
    text = _text(value)
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _stable_seed(*values: str) -> int:
    return int(_digest([SCHEMA_VERSION, *values])[:16], 16)


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def _json_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value or "[]"))
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _study_paths(private_root: str | Path, output_root: str | Path, study_id: str) -> dict[str, Path]:
    private = Path(private_root) / "research_validation" / study_id
    public = Path(output_root) / "research_validation" / study_id
    return {
        "private": private,
        "public": public,
        "manifest": private / "study.json",
        "papers": private / "screening_papers.csv",
        "linkage": private / "review_linkage.json",
        "gold": private / "gold.json",
        "registry": Path(private_root) / "research_validation" / "registry.jsonl",
    }


def _load_manifest(private_root: str | Path, study_id: str) -> tuple[dict[str, Any], dict[str, Path]]:
    paths = _study_paths(private_root, "outputs", study_id)
    if not paths["manifest"].exists():
        raise ValueError(f"Unknown research-validation study: {study_id}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    output_root = manifest["storage"]["output_root"]
    paths = _study_paths(private_root, output_root, study_id)
    return manifest, paths


def _save_manifest(manifest: dict[str, Any], paths: dict[str, Path]) -> None:
    manifest["updated_at"] = _now()
    _atomic_json(paths["manifest"], manifest)


def _validate_input_columns(frame: pd.DataFrame) -> None:
    leaked = sorted(
        column for column in frame.columns if _normalize_name(column) in FORBIDDEN_SCREENING_FIELDS
    )
    if leaked:
        raise ValueError("Screening corpus contains forbidden evaluation fields: " + ", ".join(leaked))


def _paper_fingerprint(row: dict[str, Any]) -> str:
    return _digest({
        "source_row_index": _source_key(row.get("Source_Row_Index")),
        "title": _text(row.get("Title")), "abstract": _text(row.get("Abstract")),
        "year": _text(row.get("Year")), "doi": _text(row.get("DOI")),
    })


def _preregistration_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"], "study_id": manifest["study_id"],
        "corpus_fingerprint": manifest["corpus_fingerprint"], "source": manifest["source"],
        "review": manifest["review"], "reviewers": manifest["reviewers"],
        "sampling": {
            key: manifest["sampling"][key] for key in (
                "method", "pilot_size", "core_sample_size", "risk_sample_limit",
                "core_source_ids", "decision_ratio_targets",
            )
        },
        "policy": manifest["policy"],
        "required_version": manifest["screening"]["required_version"],
    }


def _validate_preregistration(manifest: dict[str, Any], paths: dict[str, Path]) -> None:
    expected = manifest.get("preregistration", {})
    if expected.get("manifest_fingerprint") != _digest(_preregistration_payload(manifest)):
        raise ValueError("Private study preregistration was altered")
    if not paths["papers"].exists() or expected.get("screening_input_sha256") != hashlib.sha256(paths["papers"].read_bytes()).hexdigest():
        raise ValueError("Preregistered screening corpus was altered")


def initialize_study(
    *, corpus_path: str | Path, research_question: str, title_column: str,
    abstract_column: str, reviewer_ids: list[str], private_root: str | Path = "private",
    output_root: str | Path = "outputs", research_context: str = "",
    inclusion_criteria: str = "", exclusion_criteria: str = "",
    year_column: str = "", doi_column: str = "", pilot_size: int = 100,
    core_sample_size: int = 60, risk_sample_size: int = 30,
    manual_review_capacity: float = 0.30,
) -> dict[str, Any]:
    reviewer_ids = [_text(reviewer) for reviewer in reviewer_ids]
    if len(reviewer_ids) != 2 or not all(reviewer_ids) or len(set(map(str.casefold, reviewer_ids))) != 2:
        raise ValueError("Exactly two distinct reviewer identifiers are required")
    if not 0 < manual_review_capacity <= 1:
        raise ValueError("manual_review_capacity must be between 0 and 1")
    source = Path(corpus_path)
    frame = pd.read_csv(source)
    _validate_input_columns(frame)
    missing = [column for column in (title_column, abstract_column) if column not in frame.columns]
    if missing:
        raise ValueError("Corpus is missing columns: " + ", ".join(missing))
    rows = []
    for source_index, row in frame.iterrows():
        title, abstract = _text(row[title_column]), _text(row[abstract_column])
        if not title or not abstract:
            continue
        rows.append({
            "Corpus_Source_Index": str(source_index), "Title": title, "Abstract": abstract,
            "Year": _text(row[year_column]) if year_column and year_column in frame.columns else "",
            "DOI": _text(row[doi_column]) if doi_column and doi_column in frame.columns else "",
        })
    if not rows:
        raise ValueError("Corpus contains no complete title-and-abstract rows")
    question = _text(research_question)
    if not question:
        raise ValueError("research_question is required")
    canonical = [{key: row[key] for key in ("Corpus_Source_Index", "Title", "Abstract", "Year", "DOI")} for row in rows]
    corpus_fingerprint = _digest(canonical)
    study_id = _digest({
        "corpus": corpus_fingerprint, "question": question, "context": research_context,
        "inclusion": inclusion_criteria, "exclusion": exclusion_criteria,
        "reviewers": sorted(reviewer_ids, key=str.casefold), "schema": SCHEMA_VERSION,
    })[:16]
    paths = _study_paths(private_root, output_root, study_id)
    if paths["manifest"].exists():
        existing = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        feasibility = _trust_feasibility(existing)
        return {
            "study_id": study_id, "status": existing["status"], "existing": True,
            "trust_feasibility": feasibility,
            "preregistration_warnings": [feasibility["warning"]] if feasibility["warning"] else [],
        }
    target = min(max(1, int(pilot_size)), len(canonical))
    rng = random.Random(_stable_seed(corpus_fingerprint, question, "pilot"))
    pilot = [canonical[index] for index in sorted(rng.sample(range(len(canonical)), target))]
    papers = pd.DataFrame(pilot)
    papers.insert(0, "Source_Row_Index", [str(index) for index in range(len(papers))])
    core_target = min(max(1, int(core_sample_size)), len(papers))
    core_rng = random.Random(_stable_seed(corpus_fingerprint, question, "core"))
    core_ids = sorted(str(index) for index in core_rng.sample(range(len(papers)), core_target))
    _atomic_csv(paths["papers"], papers[["Title", "Abstract", "Year", "DOI"]])
    manifest = {
        "schema_version": SCHEMA_VERSION, "study_id": study_id, "status": "INITIALIZED",
        "created_at": _now(), "updated_at": _now(), "corpus_fingerprint": corpus_fingerprint,
        "source": {"path": str(source.resolve()), "eligible_rows": len(canonical), "pilot_rows": len(papers)},
        "review": {
            "research_question": question, "research_context": _text(research_context),
            "inclusion_criteria": _text(inclusion_criteria),
            "exclusion_criteria": _text(exclusion_criteria),
        },
        "reviewers": list(reviewer_ids),
        "sampling": {
            "method": "uniform_probability_before_screening",
            "pilot_size": len(papers), "core_sample_size": len(core_ids),
            "risk_sample_limit": min(max(0, int(risk_sample_size)), max(0, len(papers) - len(core_ids))),
            "core_source_ids": core_ids, "risk_source_ids": [],
            "decision_ratio_targets": False,
        },
        "policy": {
            "manual_review_capacity": float(manual_review_capacity), "minimum_core_labels": min(60, len(core_ids)),
            "minimum_reviewer_coverage": 0.95, "minimum_kappa": 0.70,
            "minimum_relevant_recall": 0.95, "maximum_false_reject_rate": 0.05,
            "minimum_keep_precision": 0.85, "minimum_repeatability": 0.90,
            "maximum_transport_fallback_rate": 0.05,
        },
        "screening": {"engine": "gemini_web_v24", "required_version": FROZEN_GEMINI_VERSION, "runs": []},
        "reviews": {}, "adjudication": {"status": "NOT_READY"},
        "storage": {"private_root": str(Path(private_root)), "output_root": str(Path(output_root))},
        "leakage_checks": {"input_columns_clean": True, "gold_locked_before_comparison": False},
    }
    manifest["preregistration"] = {
        "locked_at": _now(),
        "screening_input_sha256": hashlib.sha256(paths["papers"].read_bytes()).hexdigest(),
        "manifest_fingerprint": _digest(_preregistration_payload(manifest)),
    }
    _save_manifest(manifest, paths)
    feasibility = _trust_feasibility(manifest)
    return {
        "study_id": study_id, "status": manifest["status"], "existing": False,
        "pilot_size": len(papers), "core_sample_size": len(core_ids),
        "manifest_path": str(paths["manifest"]), "papers_path": str(paths["papers"]),
        "trust_feasibility": feasibility,
        "preregistration_warnings": [feasibility["warning"]] if feasibility["warning"] else [],
    }


def _run_record(summary: dict[str, Any], output_path: Path, label: str) -> dict[str, Any]:
    diagnostics = Path(str(summary["diagnostics_path"]))
    validate_diagnostics(diagnostics)
    return {
        "label": label, "job_id": str(summary.get("job_id") or output_path.stem),
        "output_path": str(output_path), "diagnostics_path": str(diagnostics),
        "diagnostics_summary_path": str(diagnostics.with_suffix(".summary.json")),
        "architecture_version": summary.get("architecture_version"),
        "protocol_id": summary.get("protocol_id"), "resumed_count": summary.get("resumed_count"),
        "runtime_seconds": summary.get("runtime_seconds"), "retry_count": summary.get("retry_count"),
        "timeout_fallback_count": summary.get("timeout_fallback_count"),
        "summary": summary, "completed_at": _now(),
    }


def run_study(
    study_id: str, *, private_root: str | Path = "private",
    screen: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if os.getenv("GEMINI_WEB_CAPTURE_RAW_DEBUG", "").strip().casefold() in {"1", "true", "yes"}:
        raise RuntimeError("Research validation requires raw Gemini capture to remain disabled")
    manifest, paths = _load_manifest(private_root, study_id)
    _validate_preregistration(manifest, paths)
    if manifest["status"] not in {"INITIALIZED", "SCREENING", "SCREENED"}:
        raise ValueError(f"Study cannot screen from state {manifest['status']}")
    if screen is None:
        from litsync_app.screening.bulk import screen_csv
        screen_function = screen_csv
    else:
        screen_function = screen
    manifest["status"] = "SCREENING"
    _save_manifest(manifest, paths)
    for directory in (
        paths["public"] / "cache" / "gemini_web_v24" / "diagnostics",
        paths["public"] / "cache" / "gemini_web_v24" / "checkpoints",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    runs = list(manifest["screening"].get("runs", []))
    completed_labels = {run["label"] for run in runs}
    for label in ("repeat_a", "repeat_b"):
        if label in completed_labels:
            continue
        job_id = f"rv-{study_id[:8]}-{label[-1]}-{uuid.uuid4().hex[:8]}"
        output_path = (paths["public"] / "runs" / f"{label}.csv").resolve()
        summary = screen_function(
            csv_path=str(paths["papers"].resolve()), research_question=manifest["review"]["research_question"],
            research_context=manifest["review"]["research_context"],
            inclusion_criteria=manifest["review"]["inclusion_criteria"],
            exclusion_criteria=manifest["review"]["exclusion_criteria"],
            output_path=str(output_path), progress_job_id=job_id,
            screening_engine="gemini_web_v24", resume=False,
        )
        if summary.get("architecture_version") != FROZEN_GEMINI_VERSION:
            raise RuntimeError("Research validation must use Gemini Web v2.4")
        if int(summary.get("resumed_count", -1)) != 0:
            raise RuntimeError("Research validation runs must not resume paper decisions")
        runs.append(_run_record(summary, output_path, label))
        manifest["screening"]["runs"] = runs
        _save_manifest(manifest, paths)
    protocols = {str(run.get("protocol_id") or "") for run in runs}
    if len(protocols) != 1 or "" in protocols:
        manifest["status"] = "PROTOCOL_DRIFT"
        _save_manifest(manifest, paths)
        raise RuntimeError("Repeat runs did not use one immutable protocol")
    manifest["status"] = "SCREENED"
    manifest["screening"]["protocol_id"] = next(iter(protocols))
    _select_risk_sample(manifest, paths)
    _save_manifest(manifest, paths)
    return study_status(study_id, private_root=private_root)


def _run_rows(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        _source_key(row.get("Source_Row_Index")): row
        for row in pd.read_csv(run["output_path"]).to_dict(orient="records")
    }


def _risk_reasons(first: dict[str, Any], second: dict[str, Any]) -> list[str]:
    reasons = []
    decision_a, decision_b = _text(first.get("Decision")).upper(), _text(second.get("Decision")).upper()
    if decision_a != decision_b:
        reasons.append("repeatability_disagreement")
    if {decision_a, decision_b} == {"KEEP", "REJECT"}:
        reasons.append("direct_decision_contradiction")
    if any(_text(row.get("Failure_Class")) == "transport_timeout" for row in (first, second)):
        reasons.append("transport_failure")
    if any(_text(row.get("Validation_Status")) != "validated" for row in (first, second)):
        reasons.append("invalid_structure_or_evidence")
    statuses = {_text(row.get("Verification_Status")) for row in (first, second)}
    if statuses & {"failed", "uncertain", "disagreed"}:
        reasons.append("verification_failure_or_disagreement")
    if any(_text(row.get("Critic_Route")) for row in (first, second)):
        reasons.append("critic_routed")
    criteria = [item for row in (first, second) for item in _json_list(row.get("Criteria_JSON"))]
    if any(item.get("scope_support") in {"INCIDENTAL", "INSUFFICIENT"} for item in criteria if isinstance(item, dict)):
        reasons.append("non_substantive_support")
    return sorted(set(reasons))


def _select_risk_sample(manifest: dict[str, Any], paths: dict[str, Path]) -> None:
    runs = manifest["screening"]["runs"]
    first, second = _run_rows(runs[0]), _run_rows(runs[1])
    core = set(manifest["sampling"]["core_source_ids"])
    candidates = []
    for source_id in sorted(set(first) & set(second), key=lambda value: int(value)):
        if source_id in core:
            continue
        reasons = _risk_reasons(first[source_id], second[source_id])
        if reasons:
            candidates.append({
                "source_id": source_id, "reasons": reasons,
                "tie": _digest([manifest["study_id"], "risk", source_id]),
            })
    uncovered = set(reason for candidate in candidates for reason in candidate["reasons"])
    selected = []
    remaining = list(candidates)
    limit = int(manifest["sampling"]["risk_sample_limit"])
    while remaining and len(selected) < limit:
        remaining.sort(
            key=lambda item: (-len(set(item["reasons"]) & uncovered), -len(item["reasons"]), item["tie"])
        )
        chosen = remaining.pop(0)
        selected.append(chosen)
        uncovered -= set(chosen["reasons"])
    manifest["sampling"]["risk_source_ids"] = [item["source_id"] for item in selected]
    manifest["sampling"]["risk_selection_private"] = selected


def _review_source_rows(manifest: dict[str, Any], paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    papers = pd.read_csv(paths["papers"], dtype=str, keep_default_na=False).to_dict(orient="records")
    return {str(index): {"Source_Row_Index": str(index), **row} for index, row in enumerate(papers)}


def _load_protocol(paths: dict[str, Path], protocol_id: str) -> dict[str, Any]:
    protocol_root = paths["public"] / "cache" / "gemini_web_v24" / "protocols"
    for path in protocol_root.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("protocol_id")) == str(protocol_id):
            return payload
    raise ValueError("The immutable compiled protocol artifact is missing")


def export_review_packs(study_id: str, *, private_root: str | Path = "private") -> dict[str, Any]:
    manifest, paths = _load_manifest(private_root, study_id)
    _validate_preregistration(manifest, paths)
    if manifest["status"] not in {"SCREENED", "REVIEWING", "ADJUDICATING", "GOLD_LOCKED", "REPORTED"}:
        raise ValueError("Both screening repeats must finish before review packs are exported")
    source_rows = _review_source_rows(manifest, paths)
    selected = sorted(set(
        manifest["sampling"]["core_source_ids"] + manifest["sampling"]["risk_source_ids"]
    ), key=lambda value: int(value))
    linkage = {"study_id": study_id, "reviewers": {}, "paper_fingerprints": {}}
    outputs = {}
    for reviewer in manifest["reviewers"]:
        ordered = list(selected)
        random.Random(_stable_seed(study_id, reviewer, "review-pack")).shuffle(ordered)
        records = []
        reviewer_links = {}
        for source_id in ordered:
            source = source_rows[source_id]
            opaque = _digest([study_id, reviewer, source_id])[:16]
            fingerprint = _paper_fingerprint(source)
            reviewer_links[opaque] = source_id
            linkage["paper_fingerprints"][source_id] = fingerprint
            records.append({
                "Study_ID": study_id, "Reviewer_ID": reviewer, "Review_Row_ID": opaque,
                "Research_Question": manifest["review"]["research_question"],
                "Research_Context": manifest["review"]["research_context"],
                "Inclusion_Criteria": manifest["review"]["inclusion_criteria"],
                "Exclusion_Criteria": manifest["review"]["exclusion_criteria"],
                "Title": source["Title"], "Abstract": source["Abstract"],
                "Year": source.get("Year", ""), "DOI": source.get("DOI", ""),
                "Human_Decision": "", "Reviewer_Rationale": "", "Reviewer_Confidence": "",
            })
        linkage["reviewers"][reviewer] = reviewer_links
        path = paths["public"] / "review-packs" / f"{_normalize_name(reviewer)}.csv"
        _atomic_csv(path, pd.DataFrame(records))
        outputs[reviewer] = str(path)
    _atomic_json(paths["linkage"], linkage)
    manifest["status"] = "REVIEWING"
    manifest["reviews"]["pack_paths"] = outputs
    manifest["reviews"]["selected_rows"] = len(selected)
    _save_manifest(manifest, paths)
    return {"study_id": study_id, "status": manifest["status"], "review_packs": outputs}


def import_review(
    study_id: str, reviewer_id: str, completed_path: str | Path, *,
    private_root: str | Path = "private",
) -> dict[str, Any]:
    manifest, paths = _load_manifest(private_root, study_id)
    _validate_preregistration(manifest, paths)
    if reviewer_id not in manifest["reviewers"]:
        raise ValueError("Reviewer is not registered for this study")
    if not paths["linkage"].exists():
        raise ValueError("Export blinded review packs before importing labels")
    linkage = json.loads(paths["linkage"].read_text(encoding="utf-8"))
    expected = linkage["reviewers"][reviewer_id]
    frame = pd.read_csv(completed_path, dtype=str, keep_default_na=False)
    required = {
        "Study_ID", "Reviewer_ID", "Review_Row_ID", "Title", "Abstract",
        "Human_Decision", "Reviewer_Rationale", "Reviewer_Confidence",
    }
    if required - set(frame.columns):
        raise ValueError("Completed review pack is missing required columns")
    if set(frame["Review_Row_ID"]) != set(expected):
        raise ValueError("Completed review rows do not match the exported reviewer pack")
    source_rows = _review_source_rows(manifest, paths)
    records = {}
    for _, row in frame.iterrows():
        opaque = _text(row["Review_Row_ID"])
        source_id = expected[opaque]
        if _text(row["Study_ID"]) != study_id or _text(row["Reviewer_ID"]) != reviewer_id:
            raise ValueError("Study or reviewer identity was altered")
        source = source_rows[source_id]
        if _text(row["Title"]) != source["Title"] or _text(row["Abstract"]) != source["Abstract"]:
            raise ValueError(f"Paper text was altered for review row {opaque}")
        decision = _text(row["Human_Decision"]).upper()
        confidence = _text(row["Reviewer_Confidence"]).upper()
        rationale = _text(row["Reviewer_Rationale"])
        if decision not in VALID_HUMAN_LABELS:
            raise ValueError(f"Invalid human decision for review row {opaque}: {decision or 'blank'}")
        if confidence not in VALID_CONFIDENCE:
            raise ValueError(f"Invalid reviewer confidence for review row {opaque}")
        if not rationale:
            raise ValueError(f"Reviewer rationale is required for review row {opaque}")
        records[source_id] = {
            "decision": decision, "rationale": rationale, "confidence": confidence,
            "submitted_at": _now(),
        }
    review_path = paths["private"] / "reviews" / f"{_normalize_name(reviewer_id)}.json"
    _atomic_json(review_path, {"study_id": study_id, "reviewer_id": reviewer_id, "records": records})
    manifest["reviews"].setdefault("submitted", {})[reviewer_id] = str(review_path)
    if len(manifest["reviews"]["submitted"]) == 2:
        manifest["status"] = "ADJUDICATING"
        manifest["adjudication"]["status"] = "READY"
    _save_manifest(manifest, paths)
    return {"study_id": study_id, "reviewer_id": reviewer_id, "rows": len(records), "status": manifest["status"]}


def _load_reviews(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reviews = {}
    for reviewer, path in manifest.get("reviews", {}).get("submitted", {}).items():
        reviews[reviewer] = json.loads(Path(path).read_text(encoding="utf-8"))["records"]
    return reviews


def export_adjudication(study_id: str, *, private_root: str | Path = "private") -> dict[str, Any]:
    manifest, paths = _load_manifest(private_root, study_id)
    _validate_preregistration(manifest, paths)
    reviews = _load_reviews(manifest)
    if len(reviews) != 2:
        raise ValueError("Both independent review packs must be imported first")
    reviewers = manifest["reviewers"]
    source_rows = _review_source_rows(manifest, paths)
    disagreements = []
    for source_id in sorted(set(reviews[reviewers[0]]) | set(reviews[reviewers[1]]), key=lambda value: int(value)):
        first, second = reviews[reviewers[0]][source_id], reviews[reviewers[1]][source_id]
        if first["decision"] == second["decision"] and first["decision"] != "ABSTAIN":
            continue
        source = source_rows[source_id]
        disagreements.append({
            "Study_ID": study_id, "Adjudication_Row_ID": _digest([study_id, "adjudicate", source_id])[:16],
            "Research_Question": manifest["review"]["research_question"],
            "Title": source["Title"], "Abstract": source["Abstract"],
            "Reviewer_A_Decision": first["decision"], "Reviewer_A_Rationale": first["rationale"],
            "Reviewer_B_Decision": second["decision"], "Reviewer_B_Rationale": second["rationale"],
            "Final_Gold_Decision": "", "Adjudication_Rationale": "",
        })
    path = paths["public"] / "adjudication" / "adjudication.csv"
    columns = [
        "Study_ID", "Adjudication_Row_ID", "Research_Question", "Title", "Abstract",
        "Reviewer_A_Decision", "Reviewer_A_Rationale", "Reviewer_B_Decision",
        "Reviewer_B_Rationale", "Final_Gold_Decision", "Adjudication_Rationale",
    ]
    _atomic_csv(path, pd.DataFrame(disagreements, columns=columns))
    linkage = {
        row["Adjudication_Row_ID"]: source_id
        for source_id in source_rows
        for row in disagreements
        if row["Adjudication_Row_ID"] == _digest([study_id, "adjudicate", source_id])[:16]
    }
    _atomic_json(paths["private"] / "adjudication_linkage.json", linkage)
    manifest["adjudication"].update({
        "status": "EXPORTED", "path": str(path), "disagreement_count": len(disagreements),
    })
    _save_manifest(manifest, paths)
    return {"study_id": study_id, "disagreement_count": len(disagreements), "adjudication_path": str(path)}


def import_adjudication(
    study_id: str, completed_path: str | Path, *, private_root: str | Path = "private",
) -> dict[str, Any]:
    manifest, paths = _load_manifest(private_root, study_id)
    _validate_preregistration(manifest, paths)
    payload = _build_gold_payload(manifest, paths, completed_path)
    candidate_fingerprint = payload["gold_fingerprint"]
    lock_evidence = (
        paths["gold"].exists()
        or manifest.get("adjudication", {}).get("status") == "LOCKED"
        or manifest.get("status") in {"GOLD_LOCKED", "REPORTED"}
    )
    if lock_evidence:
        existing_fingerprint = _validate_existing_gold_lock(manifest, paths)
        if candidate_fingerprint != existing_fingerprint:
            raise ValueError(
                "Gold is immutable: the imported adjudication conflicts with the locked gold fingerprint"
            )
        return {
            "study_id": study_id, "status": manifest["status"],
            "resolved_rows": len(payload["records"]), "idempotent": True,
            "gold_fingerprint": existing_fingerprint,
        }
    if manifest.get("status") != "ADJUDICATING":
        raise ValueError("Gold may be locked only from the adjudication-ready state")
    _atomic_json(paths["gold"], payload)
    manifest["status"] = "GOLD_LOCKED"
    manifest["adjudication"].update({
        "status": "LOCKED", "gold_path": str(paths["gold"]),
        "gold_fingerprint": candidate_fingerprint, "resolved_rows": len(payload["records"]),
    })
    manifest["leakage_checks"]["gold_locked_before_comparison"] = True
    _save_manifest(manifest, paths)
    return {
        "study_id": study_id, "status": manifest["status"],
        "resolved_rows": len(payload["records"]), "idempotent": False,
        "gold_fingerprint": candidate_fingerprint,
    }


def _build_gold_payload(
    manifest: dict[str, Any], paths: dict[str, Path], completed_path: str | Path,
) -> dict[str, Any]:
    reviews = _load_reviews(manifest)
    if len(reviews) != 2:
        raise ValueError("Both reviews are required")
    reviewers = manifest["reviewers"]
    linkage_path = paths["private"] / "adjudication_linkage.json"
    linkage = json.loads(linkage_path.read_text(encoding="utf-8")) if linkage_path.exists() else {}
    adjudicated = {}
    if linkage:
        frame = pd.read_csv(completed_path, dtype=str, keep_default_na=False)
        if set(frame["Adjudication_Row_ID"]) != set(linkage):
            raise ValueError("Adjudication rows do not match the exported disagreements")
        for _, row in frame.iterrows():
            source_id = linkage[_text(row["Adjudication_Row_ID"])]
            decision = _text(row["Final_Gold_Decision"]).upper()
            rationale = _text(row["Adjudication_Rationale"])
            if decision not in FINAL_GOLD_LABELS or not rationale:
                raise ValueError("Every disagreement requires a final gold decision and rationale")
            adjudicated[source_id] = {"decision": decision, "rationale": rationale, "basis": "adjudicated"}
    gold = {}
    for source_id in sorted(reviews[reviewers[0]], key=lambda value: int(value)):
        first, second = reviews[reviewers[0]][source_id], reviews[reviewers[1]][source_id]
        if source_id in adjudicated:
            gold[source_id] = adjudicated[source_id]
        elif first["decision"] == second["decision"] and first["decision"] in FINAL_GOLD_LABELS:
            gold[source_id] = {
                "decision": first["decision"],
                "rationale": f"Reviewer A: {first['rationale']} Reviewer B: {second['rationale']}",
                "basis": "reviewer_agreement",
            }
        else:
            raise ValueError(f"Source row {source_id} remains unresolved")
    source_rows = _review_source_rows(manifest, paths)
    payload = {
        "schema_version": SCHEMA_VERSION, "study_id": manifest["study_id"], "locked_at": _now(),
        "records": {
            source_id: {**record, "paper_fingerprint": _paper_fingerprint(source_rows[source_id])}
            for source_id, record in gold.items()
        },
    }
    payload["gold_fingerprint"] = _digest(payload["records"])
    return payload


def _validate_existing_gold_lock(
    manifest: dict[str, Any], paths: dict[str, Path],
) -> str:
    if not paths["gold"].exists():
        raise ValueError("Immutable gold lock is inconsistent: gold.json is missing")
    try:
        gold_payload = json.loads(paths["gold"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Immutable gold lock is inconsistent: gold.json is unreadable") from exc
    records = gold_payload.get("records")
    if not isinstance(records, dict):
        raise ValueError("Immutable gold lock is inconsistent: gold records are missing")
    recomputed = _digest(records)
    gold_fingerprint = _text(gold_payload.get("gold_fingerprint"))
    manifest_fingerprint = _text(manifest.get("adjudication", {}).get("gold_fingerprint"))
    if not gold_fingerprint or not manifest_fingerprint:
        raise ValueError("Immutable gold lock is inconsistent: fingerprint metadata is incomplete")
    if len({recomputed, gold_fingerprint, manifest_fingerprint}) != 1:
        raise ValueError("Immutable gold lock is inconsistent: gold and manifest fingerprints disagree")
    report_metadata = manifest.get("report") or {}
    if manifest.get("status") == "REPORTED" and not report_metadata:
        raise ValueError("Immutable gold lock is inconsistent: reported study metadata is missing")
    if report_metadata:
        report_path = Path(str(report_metadata.get("path") or ""))
        if not report_path.exists():
            raise ValueError("Immutable gold lock is inconsistent: report file is missing")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Immutable gold lock is inconsistent: report file is unreadable") from exc
        report_fingerprint = _text(report.get("inputs", {}).get("gold_fingerprint"))
        if report_fingerprint != recomputed:
            raise ValueError("Immutable gold lock is inconsistent: report fingerprint disagrees")
    return recomputed


def _cohen_kappa(first: list[str], second: list[str]) -> tuple[float | None, float]:
    if not first or len(first) != len(second):
        return None, 0.0
    labels = ("KEEP", "MAYBE", "REJECT", "ABSTAIN")
    agreement = sum(a == b for a, b in zip(first, second)) / len(first)
    expected = sum(
        (first.count(label) / len(first)) * (second.count(label) / len(second))
        for label in labels
    )
    kappa = (agreement - expected) / (1 - expected) if expected < 1 else 1.0
    return round(kappa, 4), round(agreement, 4)


def _wilson(successes: int, total: int, z: float = 1.96) -> dict[str, float | None]:
    if total <= 0:
        return {"estimate": None, "lower": None, "upper": None}
    estimate = successes / total
    denominator = 1 + z * z / total
    center = (estimate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((estimate * (1 - estimate) + z * z / (4 * total)) / total) / denominator
    return {
        "estimate": round(estimate, 4), "lower": round(max(0, center - margin), 4),
        "upper": round(min(1, center + margin), 4),
    }


def _minimum_perfect_sample_for_lower_bound(threshold: float) -> int:
    total = 1
    while _wilson(total, total)["lower"] < threshold:
        total += 1
    return total


def _trust_feasibility(manifest: dict[str, Any]) -> dict[str, Any]:
    threshold = float(manifest["policy"]["minimum_relevant_recall"])
    core_size = int(manifest["sampling"]["core_sample_size"])
    minimum = _minimum_perfect_sample_for_lower_bound(threshold)
    maximum_lower = _wilson(core_size, core_size)["lower"]
    feasible = core_size >= minimum
    return {
        "recall_threshold": threshold,
        "core_sample_size": core_size,
        "maximum_achievable_recall_lower_bound": maximum_lower,
        "minimum_perfect_relevant_rows_required": minimum,
        "certifiable_with_current_core_size": feasible,
        "warning": (
            ""
            if feasible else
            f"A {core_size}-paper core cannot place the Wilson lower bound at {threshold:.2f}, "
            f"even if every sampled paper is relevant and every relevant paper is retrieved; "
            f"at least {minimum} relevant gold papers are required."
        ),
    }


def _repeatability(first: dict[str, dict[str, Any]], second: dict[str, dict[str, Any]]) -> dict[str, Any]:
    shared = sorted(set(first) & set(second), key=lambda value: int(value))
    transitions = {a: {b: 0 for b in ("KEEP", "MAYBE", "REJECT")} for a in ("KEEP", "MAYBE", "REJECT")}
    disagreements, direct, repeated_maybe = [], [], []
    for source_id in shared:
        a, b = _text(first[source_id].get("Decision")).upper(), _text(second[source_id].get("Decision")).upper()
        if a in transitions and b in transitions[a]:
            transitions[a][b] += 1
        if a != b:
            disagreements.append(source_id)
        if {a, b} == {"KEEP", "REJECT"}:
            direct.append(source_id)
        if a == b == "MAYBE":
            repeated_maybe.append(source_id)
    return {
        "shared_rows": len(shared),
        "exact_agreement_rate": round((len(shared) - len(disagreements)) / len(shared), 4) if shared else 0.0,
        "transition_matrix": transitions, "disagreement_rows": disagreements,
        "keep_reject_contradictions": direct, "repeated_maybe_rows": repeated_maybe,
    }


def _root_cause(
    first: dict[str, Any], second: dict[str, Any], gold: str,
) -> tuple[str, str]:
    rows = (first, second)
    if any(_text(row.get("Failure_Class")) == "transport_timeout" for row in rows):
        return "browser_transport_failure", "A screening pass used the transport-failure fallback."
    if any(_text(row.get("Validation_Status")) != "validated" for row in rows):
        return "evidence_grounding_failure", "A structured assessment or cited evidence failed validation."
    if any(_text(row.get("Verification_Status")) in {"failed", "uncertain", "disagreed"} for row in rows):
        return "verification_failure", "Independent verification did not produce a usable agreement."
    decisions = {_text(row.get("Decision")).upper() for row in rows}
    if len(decisions) > 1:
        return "model_nondeterminism", "Identical paper and protocol produced different decisions."
    decision = next(iter(decisions), "")
    if (gold == "KEEP" and decision == "REJECT") or (gold == "REJECT" and decision == "KEEP"):
        return "semantic_decision_error", "Validated repeat decisions contradict the locked human gold label."
    return "possible_protocol_error", "The decision remains unresolved or overcommitted and requires protocol review."


def _model_metrics(
    rows: dict[str, dict[str, Any]], gold: dict[str, dict[str, Any]], core: set[str],
) -> dict[str, Any]:
    resolved = sorted(core & set(gold) & set(rows), key=lambda value: int(value))
    confusion = {label: {decision: 0 for decision in ("KEEP", "MAYBE", "REJECT")} for label in ("KEEP", "MAYBE", "REJECT")}
    for source_id in resolved:
        human, model = gold[source_id]["decision"], _text(rows[source_id].get("Decision")).upper()
        confusion[human][model] += 1
    relevant = sum(confusion["KEEP"].values())
    relevant_retrieved = confusion["KEEP"]["KEEP"] + confusion["KEEP"]["MAYBE"]
    false_rejects = confusion["KEEP"]["REJECT"]
    model_keeps = sum(confusion[label]["KEEP"] for label in confusion)
    correct_keeps = confusion["KEEP"]["KEEP"]
    resolved_definitive = sum(
        confusion[label][decision]
        for label in ("KEEP", "REJECT") for decision in ("KEEP", "REJECT")
    )
    correct_definitive = confusion["KEEP"]["KEEP"] + confusion["REJECT"]["REJECT"]
    return {
        "resolved_core_rows": len(resolved), "confusion": confusion,
        "relevant_recall_keep_or_maybe": _wilson(relevant_retrieved, relevant),
        "false_reject_rate": _wilson(false_rejects, relevant),
        "definitive_keep_precision": _wilson(correct_keeps, model_keeps),
        "definitive_accuracy": _wilson(correct_definitive, resolved_definitive),
        "gold_maybe_overcommitment_count": confusion["MAYBE"]["KEEP"] + confusion["MAYBE"]["REJECT"],
    }


def _conservative_quality(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    combined = {"resolved_core_rows": min(first["resolved_core_rows"], second["resolved_core_rows"])}
    for name, direction in (
        ("relevant_recall_keep_or_maybe", "minimum"),
        ("false_reject_rate", "maximum"),
        ("definitive_keep_precision", "minimum"),
        ("definitive_accuracy", "minimum"),
    ):
        candidates = [first[name], second[name]]
        available = [item for item in candidates if item["estimate"] is not None]
        if not available:
            combined[name] = {"estimate": None, "lower": None, "upper": None}
        else:
            chooser = min if direction == "minimum" else max
            combined[name] = chooser(available, key=lambda item: item["estimate"])
    combined["gold_maybe_overcommitment_count"] = max(
        first["gold_maybe_overcommitment_count"], second["gold_maybe_overcommitment_count"]
    )
    return combined


def _operational_metrics(run: dict[str, Any], rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    decisions = Counter(_text(row.get("Decision")).upper() for row in rows.values())
    invalid = sum(_text(row.get("Validation_Status")) != "validated" for row in rows.values())
    fallback = sum(_text(row.get("Failure_Class")) == "transport_timeout" for row in rows.values())
    critics = sum(bool(_text(row.get("Critic_Route"))) for row in rows.values())
    verification_fallback = sum(_text(row.get("Verification_Status")) in {"failed", "uncertain", "disagreed"} for row in rows.values())
    support = Counter(
        item.get("scope_support", "MISSING")
        for row in rows.values() for item in _json_list(row.get("Criteria_JSON")) if isinstance(item, dict)
    )
    diagnostics = []
    diagnostic_path = Path(str(run.get("diagnostics_path") or ""))
    if diagnostic_path.exists():
        diagnostics = [json.loads(line) for line in diagnostic_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    submissions = [int(event.get("submission_number") or 0) for event in diagnostics]
    midpoint = max(submissions, default=0) / 2
    failure_events = [event for event in diagnostics if event.get("outcome") not in {"success", "completed", None, ""}]
    late_failures = sum(int(event.get("submission_number") or 0) > midpoint for event in failure_events)
    runtime = run.get("runtime_seconds")
    return {
        "total_rows": total, "decision_counts": dict(decisions),
        "manual_review_burden": round(decisions.get("MAYBE", 0) / total, 4) if total else 0.0,
        "critic_rate": round(critics / total, 4) if total else 0.0,
        "verification_fallback_rate": round(verification_fallback / total, 4) if total else 0.0,
        "invalid_row_count": invalid, "transport_fallback_count": fallback,
        "transport_fallback_rate": round(fallback / total, 4) if total else 0.0,
        "scope_support_counts": dict(support), "runtime_seconds": runtime,
        "papers_per_minute": round(total / (float(runtime) / 60), 3) if runtime else None,
        "retry_count": run.get("retry_count"),
        "detector_outcomes": run.get("summary", {}).get("detector_outcomes", {}),
        "recovery_actions": run.get("summary", {}).get("recovery_actions", {}),
        "critic_route_counts": run.get("summary", {}).get("critic_route_counts", {}),
        "verification_outcomes": run.get("summary", {}).get("verification_outcomes", {}),
        "diagnostic_event_count": len(diagnostics),
        "generation_detected_events": sum(bool(event.get("generation_detected")) for event in diagnostics),
        "timeout_stage_counts": dict(Counter(
            _text(event.get("timeout_stage")) for event in diagnostics if _text(event.get("timeout_stage"))
        )),
        "late_session_failure_events": late_failures,
        "late_session_degradation_observed": bool(late_failures),
    }


def _combined_operations(runs: list[dict[str, Any]], rows: list[dict[str, dict[str, Any]]]) -> dict[str, Any]:
    per_run = [_operational_metrics(run, run_rows) for run, run_rows in zip(runs, rows)]
    return {
        "runs": per_run,
        "total_runtime_seconds": round(sum(float(item.get("runtime_seconds") or 0) for item in per_run), 3),
        "total_retries": sum(int(item.get("retry_count") or 0) for item in per_run),
        "total_transport_fallbacks": sum(item["transport_fallback_count"] for item in per_run),
        "maximum_transport_fallback_rate": max((item["transport_fallback_rate"] for item in per_run), default=0.0),
        "maximum_manual_review_burden": max((item["manual_review_burden"] for item in per_run), default=0.0),
        "any_invalid_rows": any(item["invalid_row_count"] for item in per_run),
    }


def _operational_signatures(operations: dict[str, Any]) -> list[str]:
    runs = operations.get("runs", [])
    signatures = set()
    if any(run.get("late_session_degradation_observed") for run in runs):
        signatures.add("late_session_degradation")
    if any(
        int(run.get("recovery_actions", {}).get("browser_recycle_after_no_container_timeout") or 0) > 0
        for run in runs
    ):
        signatures.add("no_container_timeout_recovery")
    if (
        int(operations.get("total_transport_fallbacks") or 0) > 0
        or any(
            int(run.get("recovery_actions", {}).get("browser_recycle_after_exhausted_retry") or 0) > 0
            for run in runs
        )
    ):
        signatures.add("exhausted_transport_fallback")
    return sorted(signatures)


def _human_metrics(manifest: dict[str, Any], gold: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reviews = _load_reviews(manifest)
    if len(reviews) != 2:
        return {"reviewer_count": len(reviews), "coverage": 0.0, "agreement": None, "cohen_kappa": None}
    first, second = (reviews[reviewer] for reviewer in manifest["reviewers"])
    selected = set(manifest["sampling"]["core_source_ids"] + manifest["sampling"]["risk_source_ids"])
    shared = sorted(selected & set(first) & set(second), key=lambda value: int(value))
    first_labels = [first[source_id]["decision"] for source_id in shared]
    second_labels = [second[source_id]["decision"] for source_id in shared]
    kappa, agreement = _cohen_kappa(first_labels, second_labels)
    disagreement_categories = Counter(
        "->".join(sorted((a, b))) for a, b in zip(first_labels, second_labels) if a != b
    )
    return {
        "reviewer_count": 2, "selected_rows": len(selected), "shared_review_rows": len(shared),
        "coverage": round(len(gold) / len(selected), 4) if selected else 0.0,
        "raw_agreement": agreement, "cohen_kappa": kappa,
        "abstention_count": sum(label == "ABSTAIN" for label in first_labels + second_labels),
        "pre_adjudication_disagreement_count": sum(a != b for a, b in zip(first_labels, second_labels)),
        "disagreement_categories": dict(disagreement_categories),
        "reviewer_confidence_counts": {
            reviewer: dict(Counter(record["confidence"] for record in reviews[reviewer].values()))
            for reviewer in manifest["reviewers"]
        },
    }


def _root_cause_rows(
    manifest: dict[str, Any], first: dict[str, dict[str, Any]], second: dict[str, dict[str, Any]],
    gold: dict[str, dict[str, Any]], source_rows: dict[str, dict[str, Any]],
    protocol_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for source_id in sorted(gold, key=lambda value: int(value)):
        if source_id not in first or source_id not in second:
            continue
        decisions = [_text(first[source_id].get("Decision")).upper(), _text(second[source_id].get("Decision")).upper()]
        human = gold[source_id]["decision"]
        mismatch = any(
            (human == "KEEP" and decision == "REJECT")
            or (human == "REJECT" and decision == "KEEP")
            for decision in decisions
        )
        unstable = decisions[0] != decisions[1]
        verification = any(
            _text(row.get("Verification_Status")) in {"failed", "uncertain", "disagreed"}
            for row in (first[source_id], second[source_id])
        )
        if not (mismatch or unstable or verification):
            continue
        category, explanation = _root_cause(first[source_id], second[source_id], human)
        support = sorted({
            str(item.get("scope_support") or "MISSING")
            for row in (first[source_id], second[source_id])
            for item in _json_list(row.get("Criteria_JSON")) if isinstance(item, dict)
        })
        route = sorted({_text(row.get("Critic_Route")) or "none" for row in (first[source_id], second[source_id])})
        verification_status = sorted({_text(row.get("Verification_Status")) or "not_required" for row in (first[source_id], second[source_id])})
        criterion_roles = sorted({
            f"{protocol_by_id.get(str(item.get('criterion_id')), {}).get('kind', 'unknown')}:"
            f"{'required' if protocol_by_id.get(str(item.get('criterion_id')), {}).get('required') else 'optional'}"
            for row in (first[source_id], second[source_id])
            for item in _json_list(row.get("Criteria_JSON")) if isinstance(item, dict)
        })
        signature = ":".join([
            category, "->".join(decisions), "+".join(criterion_roles),
            "+".join(support), "+".join(route), "+".join(verification_status),
        ])
        records.append({
            "source_row_index": source_id, "title": source_rows[source_id]["Title"],
            "gold_decision": human, "gold_rationale": gold[source_id]["rationale"],
            "repeat_a_decision": decisions[0], "repeat_b_decision": decisions[1],
            "repeat_a_reason": _text(first[source_id].get("Reason")),
            "repeat_b_reason": _text(second[source_id].get("Reason")),
            "repeat_a_confidence": _text(first[source_id].get("Confidence")),
            "repeat_b_confidence": _text(second[source_id].get("Confidence")),
            "repeat_a_decision_risk": _text(first[source_id].get("Decision_Risk")),
            "repeat_b_decision_risk": _text(second[source_id].get("Decision_Risk")),
            "repeat_a_validation": _text(first[source_id].get("Validation_Status")),
            "repeat_b_validation": _text(second[source_id].get("Validation_Status")),
            "repeat_a_criteria": _json_list(first[source_id].get("Criteria_JSON")),
            "repeat_b_criteria": _json_list(second[source_id].get("Criteria_JSON")),
            "repeat_a_evidence": _json_list(first[source_id].get("Evidence_JSON")),
            "repeat_b_evidence": _json_list(second[source_id].get("Evidence_JSON")),
            "scope_support": support, "critic_routes": route,
            "criterion_roles": criterion_roles,
            "verification_statuses": verification_status,
            "preliminary_root_cause": category, "root_cause_explanation": explanation,
            "normalized_signature": signature, "confirmed_root_cause": "",
            "researcher_notes": "",
        })
    return records


def import_root_cause_confirmation(
    study_id: str, completed_path: str | Path, *, private_root: str | Path = "private",
) -> dict[str, Any]:
    manifest, paths = _load_manifest(private_root, study_id)
    _validate_preregistration(manifest, paths)
    report_path = paths["public"] / "report.json"
    if not report_path.exists() or not paths["gold"].exists():
        raise ValueError("Generate the post-gold report before confirming root causes")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {str(row["source_row_index"]) for row in report.get("paper_level_root_causes", [])}
    frame = pd.read_csv(completed_path, dtype=str, keep_default_na=False)
    required = {"source_row_index", "confirmed_root_cause", "researcher_notes"}
    duplicate_ids = frame.loc[
        frame["source_row_index"].duplicated(keep=False),
        "source_row_index",
    ].tolist()
    if duplicate_ids:
        raise ValueError(
            f"Duplicate root-cause confirmation rows are not allowed: {sorted(set(duplicate_ids))}"
        )
    if required - set(frame.columns) or set(frame["source_row_index"]) != expected:
        raise ValueError("Root-cause confirmation rows do not match the generated audit")
    confirmations = {}
    for _, row in frame.iterrows():
        source_id = _source_key(row["source_row_index"])
        category, notes = _text(row["confirmed_root_cause"]), _text(row["researcher_notes"])
        if category not in ROOT_CAUSES or not notes:
            raise ValueError("Every root cause requires an approved category and researcher notes")
        confirmations[source_id] = {"category": category, "notes": notes, "confirmed_at": _now()}
    path = paths["private"] / "root_cause_confirmations.json"
    _atomic_json(path, {"study_id": study_id, "records": confirmations})
    return {"study_id": study_id, "confirmed_rows": len(confirmations), "report": generate_report(study_id, private_root=private_root)}


def _trust_verdict(
    manifest: dict[str, Any], human: dict[str, Any], quality: dict[str, Any],
    repeatability: dict[str, Any], operational: dict[str, Any], diagnostic_unsafe: int,
) -> tuple[str, list[str], list[str]]:
    policy = manifest["policy"]
    hard_failures, uncertainties = [], []
    if not manifest["leakage_checks"].get("gold_locked_before_comparison"):
        hard_failures.append("Gold labels were not locked before comparison")
    if human.get("reviewer_count") != 2 or quality["resolved_core_rows"] < policy["minimum_core_labels"]:
        return "INSUFFICIENT_EVIDENCE", hard_failures, ["Insufficient resolved dual-review core labels"]
    if human.get("coverage", 0) < policy["minimum_reviewer_coverage"]:
        return "INSUFFICIENT_EVIDENCE", hard_failures, ["Human-label coverage is below the preregistered minimum"]
    if human.get("cohen_kappa") is None or human["cohen_kappa"] < policy["minimum_kappa"]:
        return "INSUFFICIENT_EVIDENCE", hard_failures, ["Reviewer agreement is below the preregistered minimum"]
    recall = quality["relevant_recall_keep_or_maybe"]["estimate"]
    false_reject = quality["false_reject_rate"]["estimate"]
    precision = quality["definitive_keep_precision"]["estimate"]
    if recall is not None and recall < policy["minimum_relevant_recall"]:
        hard_failures.append("Relevant-paper recall is below threshold")
    if false_reject is not None and false_reject > policy["maximum_false_reject_rate"]:
        hard_failures.append("False-REJECT rate is above threshold")
    if precision is not None and precision < policy["minimum_keep_precision"]:
        hard_failures.append("Definitive-KEEP precision is below threshold")
    if diagnostic_unsafe:
        hard_failures.append("The diagnostic supplement contains an unsafe definitive error")
    if repeatability["exact_agreement_rate"] < policy["minimum_repeatability"]:
        hard_failures.append("Repeatability is below threshold")
    if repeatability["keep_reject_contradictions"]:
        hard_failures.append("Repeat runs contain KEEP-to-REJECT contradictions")
    if operational["any_invalid_rows"]:
        hard_failures.append("The run contains structurally invalid rows")
    if operational["maximum_transport_fallback_rate"] > policy["maximum_transport_fallback_rate"]:
        hard_failures.append("Transport fallback rate is above threshold")
    if hard_failures:
        return "REJECT", hard_failures, uncertainties
    for metric in ("relevant_recall_keep_or_maybe", "definitive_keep_precision"):
        interval = quality[metric]
        threshold = policy["minimum_relevant_recall"] if metric.startswith("relevant") else policy["minimum_keep_precision"]
        if interval["lower"] is not None and interval["lower"] < threshold:
            uncertainties.append(f"The 95% interval for {metric} crosses its trust threshold")
    if operational["maximum_manual_review_burden"] > policy["manual_review_capacity"]:
        uncertainties.append("Expected manual-review burden exceeds declared researcher capacity")
    return ("CONDITIONAL" if uncertainties else "TRUST"), hard_failures, uncertainties


def _report_html(report: dict[str, Any]) -> str:
    failures = "".join(f"<li>{html.escape(item)}</li>" for item in report["trust"]["failures"])
    uncertainties = "".join(f"<li>{html.escape(item)}</li>" for item in report["trust"]["uncertainties"])
    roots = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in (
            "source_row_index", "title", "gold_decision", "repeat_a_decision",
            "repeat_b_decision", "display_root_cause",
        )) + "</tr>"
        for row in report["paper_level_root_causes"]
    )
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>LitSync validation</title>
<style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:.45rem;text-align:left}}.verdict{{font-size:1.5rem;font-weight:700}}</style></head><body>
<h1>LitSync Research Validation</h1><p class='verdict'>Verdict: {html.escape(report['trust']['verdict'])}</p>
<h2>Review</h2><p>{html.escape(report['review']['research_question'])}</p>
<h2>Failures</h2><ul>{failures or '<li>None observed</li>'}</ul>
<h2>Uncertainties</h2><ul>{uncertainties or '<li>None reported</li>'}</ul>
<h2>Metrics</h2><pre>{html.escape(json.dumps(report['metrics'], indent=2))}</pre>
<h2>Paper-level root causes</h2><table><tr><th>Row</th><th>Title</th><th>Gold</th><th>Run A</th><th>Run B</th><th>Cause</th></tr>{roots}</table>
<h2>Most important limitation</h2><p>{html.escape(report['most_important_limitation'])}</p></body></html>"""


def generate_report(study_id: str, *, private_root: str | Path = "private") -> dict[str, Any]:
    manifest, paths = _load_manifest(private_root, study_id)
    _validate_preregistration(manifest, paths)
    if not paths["gold"].exists():
        safe_runs = [
            {
                "label": run["label"], "architecture_version": run.get("architecture_version"),
                "protocol_id": run.get("protocol_id"), "resumed_count": run.get("resumed_count"),
                "runtime_seconds": run.get("runtime_seconds"), "retry_count": run.get("retry_count"),
                "timeout_fallback_count": run.get("timeout_fallback_count"),
            }
            for run in manifest["screening"].get("runs", [])
        ]
        report = {
            "schema_version": SCHEMA_VERSION, "study_id": study_id,
            "trust": {"verdict": "INSUFFICIENT_EVIDENCE", "failures": [],
                      "uncertainties": ["Dual-review labels and adjudication are incomplete"]},
            "status": manifest["status"], "review": manifest["review"],
            "pre_gold_operational_evidence": {
                "screening_runs": safe_runs,
                "protocol_identity_consistent": len({run.get("protocol_id") for run in safe_runs}) <= 1,
                "core_sample_size": manifest["sampling"]["core_sample_size"],
                "diagnostic_supplement_size": len(manifest["sampling"].get("risk_source_ids", [])),
                "review_pack_rows": manifest.get("reviews", {}).get("selected_rows", 0),
            },
            "blinding": {
                "model_decisions_withheld": True, "core_selected_before_screening": True,
                "reviewer_orders_independent": True, "gold_not_fabricated": True,
            },
            "requirements_remaining": ["Import both completed reviewer packs", "Lock adjudicated gold labels"],
            "most_important_limitation": "No trustworthy model-quality conclusion is possible without independent human gold labels.",
        }
        path = paths["public"] / "report.json"
        _atomic_json(path, report)
        (paths["public"] / "report.html").write_text(_report_html({
            **report, "metrics": {}, "paper_level_root_causes": [],
        }), encoding="utf-8")
        return {**report, "report_path": str(path)}
    gold_payload = json.loads(paths["gold"].read_text(encoding="utf-8"))
    gold = gold_payload["records"]
    runs = manifest["screening"]["runs"]
    first, second = _run_rows(runs[0]), _run_rows(runs[1])
    core, risk = set(manifest["sampling"]["core_source_ids"]), set(manifest["sampling"]["risk_source_ids"])
    human = _human_metrics(manifest, gold)
    quality_a = _model_metrics(first, gold, core)
    quality_b = _model_metrics(second, gold, core)
    quality = _conservative_quality(quality_a, quality_b)
    repeatability = _repeatability(first, second)
    operational = _combined_operations(runs, [first, second])
    source_rows = _review_source_rows(manifest, paths)
    protocol = _load_protocol(paths, manifest["screening"]["protocol_id"])
    protocol_by_id = {str(item.get("id")): item for item in protocol.get("criteria", [])}
    roots = _root_cause_rows(manifest, first, second, gold, source_rows, protocol_by_id)
    confirmation_path = paths["private"] / "root_cause_confirmations.json"
    confirmations = (
        json.loads(confirmation_path.read_text(encoding="utf-8"))["records"]
        if confirmation_path.exists() else {}
    )
    for row in roots:
        confirmation = confirmations.get(str(row["source_row_index"]))
        if confirmation:
            row["confirmed_root_cause"] = confirmation["category"]
            row["researcher_notes"] = confirmation["notes"]
            row["normalized_signature"] = confirmation["category"] + ":" + row["normalized_signature"].split(":", 1)[1]
        row["display_root_cause"] = row["confirmed_root_cause"] or row["preliminary_root_cause"]
    diagnostic_unsafe = sum(
        source_id in risk and (
            (gold[source_id]["decision"] == "KEEP" and any(_text(row[source_id].get("Decision")).upper() == "REJECT" for row in (first, second)))
            or (gold[source_id]["decision"] == "REJECT" and any(_text(row[source_id].get("Decision")).upper() == "KEEP" for row in (first, second)))
        )
        for source_id in gold if source_id in first and source_id in second
    )
    verdict, failures, uncertainties = _trust_verdict(
        manifest, human, quality, repeatability, operational, diagnostic_unsafe,
    )
    signatures = Counter(row["normalized_signature"] for row in roots)
    gold_adjudication_errors = [
        row for row in roots if row.get("confirmed_root_cause") == "gold_adjudication_error"
    ]
    report = {
        "schema_version": SCHEMA_VERSION, "study_id": study_id, "generated_at": _now(),
        "status": "complete", "review": manifest["review"],
        "compiled_protocol": {
            "protocol_id": protocol.get("protocol_id"), "objective": protocol.get("objective"),
            "scope_interpretation": protocol.get("scope_interpretation"),
            "criteria": protocol.get("criteria", []),
            "ambiguities": protocol.get("ambiguities", []),
            "semantic_boundaries": protocol.get("semantic_boundaries", []),
        },
        "inputs": {
            "corpus_fingerprint": manifest["corpus_fingerprint"],
            "study_design_fingerprint": manifest["preregistration"]["manifest_fingerprint"],
            "gold_fingerprint": gold_payload["gold_fingerprint"],
            "protocol_id": manifest["screening"]["protocol_id"],
            **manifest["source"],
        },
        "blinding": {
            "core_selected_before_screening": True, "reviewers_prediction_blind": True,
            "gold_locked_before_comparison": True, "decision_ratio_targets": False,
            "raw_response_capture": False,
        },
        "metrics": {
            "human_review": human, "screening_quality": quality,
            "screening_quality_by_repeat": {"repeat_a": quality_a, "repeat_b": quality_b},
            "repeatability": repeatability, "operations": operational,
            "diagnostic_unsafe_error_count": diagnostic_unsafe,
        },
        "trust_feasibility": _trust_feasibility(manifest),
        "paired_records": [
            {
                "source_row_index": source_id, "gold_decision": gold[source_id]["decision"],
                "repeat_a_decision": _text(first[source_id].get("Decision")).upper(),
                "repeat_b_decision": _text(second[source_id].get("Decision")).upper(),
                "sample_role": "core" if source_id in core else "diagnostic",
            }
            for source_id in sorted(set(gold) & set(first) & set(second), key=lambda value: int(value))
        ],
        "recurring_failure_signatures": [
            {"signature": signature, "paper_count": count}
            for signature, count in signatures.most_common() if count >= 2
        ],
        "failure_signature_counts": dict(signatures),
        "paper_level_root_causes": roots,
        "root_cause_confirmation": {
            "required_rows": len(roots), "confirmed_rows": len(confirmations),
            "complete": len(confirmations) == len(roots),
        },
        "interpretation": {
            "confirmed_gold_adjudication_error_count": len(gold_adjudication_errors),
            "locked_gold_metrics_preserved": True,
            "gold_adjudication_note": (
                "Confirmed gold-adjudication errors qualify interpretation but do not rewrite immutable gold "
                "or retroactively alter confusion metrics."
                if gold_adjudication_errors else ""
            ),
        },
        "trust": {"verdict": verdict, "failures": failures, "uncertainties": uncertainties},
        "most_important_limitation": (
            "Title-and-abstract gold cannot determine full-text eligibility, and the finite human sample may leave "
            "rare unsafe errors statistically unresolved."
        ),
    }
    report["cross_domain"] = _append_registry(paths["registry"], manifest, report)
    _atomic_json(paths["public"] / "report.json", report)
    (paths["public"] / "report.html").write_text(_report_html(report), encoding="utf-8")
    _atomic_csv(paths["public"] / "paper-root-causes.csv", pd.DataFrame(roots))
    _atomic_csv(paths["public"] / "root-cause-confirmation.csv", pd.DataFrame(roots))
    manifest["status"] = "REPORTED"
    manifest["report"] = {"verdict": verdict, "path": str(paths["public"] / "report.json")}
    _save_manifest(manifest, paths)
    return {**report, "report_path": str(paths["public"] / "report.json")}


def _append_registry(path: Path, manifest: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    existing = [item for item in existing if item.get("study_id") != manifest["study_id"]]
    prior_signatures = defaultdict(set)
    prior_operational_signatures = defaultdict(set)
    for item in existing:
        if item.get("evidence_quality") == "dual_review_adjudicated":
            for signature in item.get("failure_signatures", []):
                prior_signatures[signature].add(item.get("domain_id"))
            historical_operations = item.get("operational_signatures") or _operational_signatures(
                item.get("summary_metrics", {}).get("operations", {})
            )
            for signature in historical_operations:
                prior_operational_signatures[signature].add(item.get("domain_id"))
    signatures = sorted({
        item["normalized_signature"] for item in report.get("paper_level_root_causes", [])
        if item.get("confirmed_root_cause")
    })
    operational_signatures = _operational_signatures(report["metrics"]["operations"])
    domain_id = _digest(manifest["review"]["research_question"])[:12]
    cross_domain = sorted(
        signature for signature in signatures
        if any(prior_domain and prior_domain != domain_id for prior_domain in prior_signatures.get(signature, set()))
    )
    cross_domain_operations = sorted(
        signature for signature in operational_signatures
        if any(
            prior_domain and prior_domain != domain_id
            for prior_domain in prior_operational_signatures.get(signature, set())
        )
    )
    record = {
        "study_id": manifest["study_id"], "registered_at": _now(),
        "workflow_version": FROZEN_GEMINI_VERSION, "protocol_id": manifest["screening"].get("protocol_id"),
        "domain_id": domain_id,
        "evidence_quality": "dual_review_adjudicated", "verdict": report["trust"]["verdict"],
        "failure_signatures": signatures, "cross_domain_signatures": cross_domain,
        "operational_signatures": operational_signatures,
        "cross_domain_operational_signatures": cross_domain_operations,
        "summary_metrics": report["metrics"],
    }
    existing.append(record)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in existing), encoding="utf-8")
    temporary.replace(path)
    return {
        "same_signature_in_independently_reviewed_domains": cross_domain,
        "operational_signatures": operational_signatures,
        "same_operational_signature_in_independently_reviewed_domains": cross_domain_operations,
        "cross_domain_semantic_weakness_confirmed": bool(cross_domain),
        "cross_domain_operational_weakness_confirmed": bool(cross_domain_operations),
        "cross_domain_weakness_confirmed": bool(cross_domain or cross_domain_operations),
        "historical_single_reviewer_controls_are_context_only": True,
    }


def compare_reports(baseline_path: str | Path, candidate_path: str | Path) -> dict[str, Any]:
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    if baseline.get("inputs", {}).get("corpus_fingerprint") != candidate.get("inputs", {}).get("corpus_fingerprint"):
        raise ValueError("Version comparison requires the same immutable corpus")
    if baseline.get("study_id") != candidate.get("study_id"):
        raise ValueError("Version comparison requires the same preregistered study")
    for key in ("study_design_fingerprint", "gold_fingerprint", "protocol_id"):
        if baseline.get("inputs", {}).get(key) != candidate.get("inputs", {}).get(key):
            raise ValueError(f"Version comparison requires the same {key}")
    baseline_rows = {str(row["source_row_index"]): row for row in baseline.get("paired_records", [])}
    candidate_rows = {str(row["source_row_index"]): row for row in candidate.get("paired_records", [])}
    if not baseline_rows or not candidate_rows:
        raise ValueError("Version comparison requires paired paper-level decisions")
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError("Version comparison requires identical locked human-gold rows")
    transitions = {a: {b: 0 for b in ("KEEP", "MAYBE", "REJECT")} for a in ("KEEP", "MAYBE", "REJECT")}
    changed = []
    for source_id in sorted(baseline_rows, key=lambda value: int(value)):
        old = baseline_rows[source_id]["repeat_a_decision"]
        new = candidate_rows[source_id]["repeat_a_decision"]
        if old in transitions and new in transitions[old]:
            transitions[old][new] += 1
        if old != new:
            changed.append({
                "source_row_index": source_id, "gold_decision": baseline_rows[source_id]["gold_decision"],
                "baseline_decision": old, "candidate_decision": new,
            })
    return {
        "study_id": baseline["study_id"],
        "baseline_verdict": baseline["trust"]["verdict"],
        "candidate_verdict": candidate["trust"]["verdict"],
        "baseline_metrics": baseline["metrics"], "candidate_metrics": candidate["metrics"],
        "paired_transition_matrix": transitions, "changed_papers": changed,
        "fair_comparison": True,
        "note": "Comparison uses identical corpus, preregistration, human gold, and sampling design.",
    }


def study_status(study_id: str, *, private_root: str | Path = "private") -> dict[str, Any]:
    manifest, paths = _load_manifest(private_root, study_id)
    _validate_preregistration(manifest, paths)
    return {
        "study_id": study_id, "status": manifest["status"],
        "screening_runs": len(manifest["screening"].get("runs", [])),
        "protocol_id": manifest["screening"].get("protocol_id"),
        "review_packs": manifest.get("reviews", {}).get("pack_paths", {}),
        "reviews_submitted": sorted(manifest.get("reviews", {}).get("submitted", {})),
        "adjudication": manifest.get("adjudication", {}), "report": manifest.get("report", {}),
        "trust_feasibility": _trust_feasibility(manifest),
        "public_root": str(paths["public"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LitSync domain-neutral research validation")
    parser.add_argument("--private-root", default="private")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--corpus", required=True); init.add_argument("--question", required=True)
    init.add_argument("--title-column", required=True); init.add_argument("--abstract-column", required=True)
    init.add_argument("--year-column", default=""); init.add_argument("--doi-column", default="")
    init.add_argument("--context", default=""); init.add_argument("--inclusion", default=""); init.add_argument("--exclusion", default="")
    init.add_argument("--reviewer", action="append", required=True); init.add_argument("--output-root", default="outputs")
    init.add_argument("--pilot-size", type=int, default=100); init.add_argument("--core-size", type=int, default=60)
    init.add_argument("--risk-size", type=int, default=30); init.add_argument("--manual-capacity", type=float, default=.30)
    for name in ("run", "export-review", "export-adjudication", "report", "status"):
        command = commands.add_parser(name); command.add_argument("--study", required=True)
    review = commands.add_parser("import-review"); review.add_argument("--study", required=True)
    review.add_argument("--reviewer", required=True); review.add_argument("--file", required=True)
    adjudicate = commands.add_parser("import-adjudication"); adjudicate.add_argument("--study", required=True)
    adjudicate.add_argument("--file", required=True)
    causes = commands.add_parser("import-root-causes"); causes.add_argument("--study", required=True)
    causes.add_argument("--file", required=True)
    compare = commands.add_parser("compare"); compare.add_argument("--baseline", required=True); compare.add_argument("--candidate", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    private_root = args.private_root
    if args.command == "init":
        result = initialize_study(
            corpus_path=args.corpus, research_question=args.question, title_column=args.title_column,
            abstract_column=args.abstract_column, year_column=args.year_column, doi_column=args.doi_column,
            reviewer_ids=args.reviewer, private_root=private_root, output_root=args.output_root,
            research_context=args.context, inclusion_criteria=args.inclusion,
            exclusion_criteria=args.exclusion, pilot_size=args.pilot_size,
            core_sample_size=args.core_size, risk_sample_size=args.risk_size,
            manual_review_capacity=args.manual_capacity,
        )
    elif args.command == "run": result = run_study(args.study, private_root=private_root)
    elif args.command == "export-review": result = export_review_packs(args.study, private_root=private_root)
    elif args.command == "import-review": result = import_review(args.study, args.reviewer, args.file, private_root=private_root)
    elif args.command == "export-adjudication": result = export_adjudication(args.study, private_root=private_root)
    elif args.command == "import-adjudication": result = import_adjudication(args.study, args.file, private_root=private_root)
    elif args.command == "import-root-causes": result = import_root_cause_confirmation(args.study, args.file, private_root=private_root)
    elif args.command == "report": result = generate_report(args.study, private_root=private_root)
    elif args.command == "status": result = study_status(args.study, private_root=private_root)
    else: result = compare_reports(args.baseline, args.candidate)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
