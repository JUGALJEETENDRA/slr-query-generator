from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any

from .contracts import BenchmarkComparison, BenchmarkResult, validate_identifier
from .errors import PublicationError
from .provenance import canonical_fingerprint, canonical_json_bytes


PUBLICATION_SCHEMA_VERSION = "litsync-benchmark-publication-v1"
COMPLETE_MARKER_NAME = "_COMPLETE.json"
REQUIRED_PUBLICATION_ARTIFACTS = {
    "evaluation": {
        "result": "benchmark-result.json",
        "html": "benchmark-report.html",
        "errors": "benchmark-errors.csv",
    },
    "comparison": {
        "comparison": "benchmark-comparison.json",
        "html": "benchmark-report.html",
        "errors": "benchmark-errors.csv",
    },
}


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _staging_root(output_dir: str | Path) -> Path:
    output = Path(output_dir).resolve()
    root = output.parent / f".benchmark-stage-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_artifact(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise PublicationError("completion marker contains an invalid artifact path")
    path = PurePath(value)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or ":" in value
    ):
        raise PublicationError("completion marker contains an unsafe artifact path")
    return value


def _artifact_manifest(
    staged_paths: dict[str, Path],
    publication_kind: str,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    required = REQUIRED_PUBLICATION_ARTIFACTS.get(publication_kind)
    if required is None:
        raise PublicationError("unsupported publication kind")
    if set(staged_paths) != set(required):
        raise PublicationError("staged publication artifact set is incomplete")
    relative_paths: list[str] = []
    artifacts: dict[str, dict[str, Any]] = {}
    for key, filename in required.items():
        path = Path(staged_paths[key])
        if not path.is_file() or path.name != filename:
            raise PublicationError(f"required publication artifact is missing: {filename}")
        if path.parent != next(iter(staged_paths.values())).parent:
            raise PublicationError("staged publication artifacts must share one directory")
        relative_paths.append(filename)
        artifacts[filename] = {
            "sha256": _sha256(path),
            "byte_size": path.stat().st_size,
        }
    return relative_paths, artifacts


def create_completion_marker(
    staged_paths: dict[str, Path],
    output_dir: str | Path,
    *,
    publication_kind: str,
    benchmark_id: str,
    benchmark_version: str,
    benchmark_spec_fingerprint: str,
    job_ids: list[str],
    verdict: str,
) -> dict[str, Any]:
    relative_paths, artifacts = _artifact_manifest(staged_paths, publication_kind)
    output = Path(output_dir).resolve()
    marker = {
        "publication_schema_version": PUBLICATION_SCHEMA_VERSION,
        "publication_kind": publication_kind,
        "benchmark_id": benchmark_id,
        "benchmark_version": benchmark_version,
        "benchmark_spec_fingerprint": benchmark_spec_fingerprint,
        "job_ids": list(job_ids),
        "verdict": verdict,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_relative_paths": relative_paths,
        "artifacts": artifacts,
    }
    marker["publication_id"] = canonical_fingerprint({
        "marker_path": str(output / COMPLETE_MARKER_NAME),
        "publication_kind": publication_kind,
        "benchmark_id": benchmark_id,
        "benchmark_version": benchmark_version,
        "benchmark_spec_fingerprint": benchmark_spec_fingerprint,
        "job_ids": list(job_ids),
        "verdict": verdict,
        "artifact_relative_paths": relative_paths,
        "artifacts": artifacts,
    })
    stage = next(iter(staged_paths.values())).parent
    _atomic_bytes(stage / COMPLETE_MARKER_NAME, canonical_json_bytes(marker))
    return marker


def validate_completion_directory(
    output_dir: str | Path,
    *,
    expected_kind: str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    output = Path(output_dir).resolve()
    marker_path = output / COMPLETE_MARKER_NAME
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationError("publication completion marker is missing or unreadable") from exc
    if not isinstance(marker, dict):
        raise PublicationError("publication completion marker must be an object")
    required_marker_fields = {
        "publication_schema_version", "publication_kind", "publication_id",
        "benchmark_id", "benchmark_version", "benchmark_spec_fingerprint",
        "job_ids", "verdict", "created_at", "artifact_relative_paths", "artifacts",
    }
    if set(marker) != required_marker_fields:
        raise PublicationError("publication completion marker fields are invalid")
    if marker["publication_schema_version"] != PUBLICATION_SCHEMA_VERSION:
        raise PublicationError("unsupported publication completion schema")
    try:
        validate_identifier(marker["benchmark_id"], label="benchmark ID")
        validate_identifier(marker["benchmark_version"], label="benchmark version")
    except (TypeError, ValueError) as exc:
        raise PublicationError("completion marker benchmark identity is invalid") from exc
    if (
        not isinstance(marker["benchmark_spec_fingerprint"], str)
        or len(marker["benchmark_spec_fingerprint"]) != 64
        or any(character not in "0123456789abcdef" for character in marker["benchmark_spec_fingerprint"])
        or not isinstance(marker["verdict"], str)
        or not marker["verdict"]
        or not isinstance(marker["created_at"], str)
    ):
        raise PublicationError("completion marker immutable metadata is invalid")
    try:
        created_at = datetime.fromisoformat(marker["created_at"])
    except ValueError as exc:
        raise PublicationError("completion marker created_at is invalid") from exc
    if created_at.tzinfo is None:
        raise PublicationError("completion marker created_at must include a timezone")
    kind = marker["publication_kind"]
    if kind not in REQUIRED_PUBLICATION_ARTIFACTS or (
        expected_kind is not None and kind != expected_kind
    ):
        raise PublicationError("publication kind does not match")
    if not isinstance(marker["job_ids"], list) or not marker["job_ids"]:
        raise PublicationError("completion marker job IDs are invalid")
    try:
        for job_id in marker["job_ids"]:
            validate_identifier(job_id, label="job ID")
    except (TypeError, ValueError) as exc:
        raise PublicationError("completion marker job IDs are invalid") from exc
    paths = marker["artifact_relative_paths"]
    if not isinstance(paths, list):
        raise PublicationError("completion marker artifact paths must be a list")
    safe_paths = [_safe_relative_artifact(value) for value in paths]
    if len(safe_paths) != len(set(safe_paths)):
        raise PublicationError("completion marker contains duplicate artifact paths")
    expected_names = list(REQUIRED_PUBLICATION_ARTIFACTS[kind].values())
    if safe_paths != expected_names:
        raise PublicationError("completion marker does not record the required artifacts")
    artifacts = marker["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_names):
        raise PublicationError("completion marker artifact manifest is invalid")
    actual_entries = {path.name for path in output.iterdir()}
    if actual_entries != set(expected_names) | {COMPLETE_MARKER_NAME}:
        raise PublicationError("publication directory contains mixed or unrecorded files")
    published: dict[str, Path] = {}
    for key, filename in REQUIRED_PUBLICATION_ARTIFACTS[kind].items():
        path = output / filename
        entry = artifacts.get(filename)
        if (
            not path.is_file()
            or not isinstance(entry, dict)
            or set(entry) != {"sha256", "byte_size"}
            or not isinstance(entry["sha256"], str)
            or len(entry["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in entry["sha256"])
            or not isinstance(entry["byte_size"], int)
            or entry["byte_size"] < 0
            or path.stat().st_size != entry["byte_size"]
            or _sha256(path) != entry["sha256"]
        ):
            raise PublicationError(f"publication artifact integrity failed: {filename}")
        published[key] = path
    calculated_id = canonical_fingerprint({
        "marker_path": str(marker_path),
        "publication_kind": kind,
        "benchmark_id": marker["benchmark_id"],
        "benchmark_version": marker["benchmark_version"],
        "benchmark_spec_fingerprint": marker["benchmark_spec_fingerprint"],
        "job_ids": marker["job_ids"],
        "verdict": marker["verdict"],
        "artifact_relative_paths": safe_paths,
        "artifacts": artifacts,
    })
    if marker["publication_id"] != calculated_id:
        raise PublicationError("publication ID does not match immutable publication content")
    return marker, published


def marker_hash(output_dir: str | Path) -> str:
    return _sha256(Path(output_dir).resolve() / COMPLETE_MARKER_NAME)


def _same_publication(existing: dict[str, Any], staged: dict[str, Any]) -> bool:
    comparable = {
        "publication_schema_version", "publication_kind", "publication_id",
        "benchmark_id", "benchmark_version", "benchmark_spec_fingerprint",
        "job_ids", "verdict", "artifact_relative_paths", "artifacts",
    }
    return all(existing.get(field) == staged.get(field) for field in comparable)


def publish_staged_report(
    staged_paths: dict[str, Path],
    output_dir: str | Path,
    *,
    publication_kind: str,
    benchmark_id: str,
    benchmark_version: str,
    benchmark_spec_fingerprint: str,
    job_ids: list[str],
    verdict: str,
    prepared_marker: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output = Path(output_dir).resolve()
    staged_marker = prepared_marker or create_completion_marker(
        staged_paths,
        output,
        publication_kind=publication_kind,
        benchmark_id=benchmark_id,
        benchmark_version=benchmark_version,
        benchmark_spec_fingerprint=benchmark_spec_fingerprint,
        job_ids=job_ids,
        verdict=verdict,
    )
    stage = next(iter(staged_paths.values())).parent
    if output.exists():
        if not output.is_dir():
            raise PublicationError("publication destination is not a directory")
        if not any(output.iterdir()):
            output.rmdir()
        else:
            existing, published = validate_completion_directory(
                output, expected_kind=publication_kind
            )
            if not _same_publication(existing, staged_marker):
                raise PublicationError(
                    "publication destination contains a different completed publication"
                )
            shutil.rmtree(stage)
            return published
    os.replace(stage, output)
    _, published = validate_completion_directory(output, expected_kind=publication_kind)
    return published


def _safe_csv_cell(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.lstrip()
    if stripped and stripped[0] in "=+-@":
        return "'" + value
    return value


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {field: _safe_csv_cell(row.get(field, "")) for field in fields}
            for row in rows
        ])


def _metric_table(result: BenchmarkResult) -> str:
    rows = []
    for name, metric in sorted(result.metrics.items()):
        interval = (
            f"{metric.confidence_interval.lower:.3f}-{metric.confidence_interval.upper:.3f}"
            if metric.confidence_interval else "-"
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td><td>{html.escape(str(metric.value))}</td>"
            f"<td>{html.escape(str(metric.numerator))}/{html.escape(str(metric.denominator))}</td>"
            f"<td>{html.escape(interval)}</td><td>{html.escape(metric.population_scope)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _result_html(result: BenchmarkResult) -> str:
    failures = "".join(
        f"<li><strong>{html.escape(item.rule)}</strong>: {html.escape(item.reason)}</li>"
        for item in result.gate.failures
    ) or "<li>None</li>"
    confusion = html.escape(json.dumps(result.confusion_matrix, indent=2))
    errors = "".join(
        f"<tr><td>{html.escape(str(row['source_row_id']))}</td>"
        f"<td>{html.escape(str(row['gold_label']))}</td>"
        f"<td>{html.escape(str(row['decision']))}</td>"
        f"<td>{html.escape(str(row['reuse_status']))}</td>"
        f"<td>{html.escape(str(row.get('title', '')))}</td></tr>"
        for row in result.row_outcomes
        if row["source_row_id"] in (
            set(result.false_keep_source_row_ids)
            | set(result.false_reject_source_row_ids)
        )
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>LitSync Benchmark</title>
<style>body{{font:15px system-ui;margin:2rem;color:#17202a}}.verdict{{font-size:2rem;font-weight:700}}
.warning{{background:#fff3cd;padding:1rem}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #ccd1d1;padding:.45rem;text-align:left}}pre{{background:#f4f6f7;padding:1rem}}</style>
</head><body><h1>{html.escape(result.benchmark_id)} {html.escape(result.benchmark_version)}</h1>
<div class="verdict">{html.escape(result.gate.verdict.value)}</div>
<p>Job: {html.escape(result.job_id)} · Provenance: {html.escape(result.provenance.classification.value)}</p>
<div class="warning"><strong>Gate findings</strong><ul>{failures}</ul></div>
<h2>Populations</h2><p>Run: {len(result.provenance.run_selected_source_row_ids)} ·
Resolved gold: {len(result.resolved_gold_source_row_ids)} · UNSURE: {len(result.unsure_gold_source_row_ids)}</p>
<h2>Metrics</h2><table><tr><th>Metric</th><th>Value</th><th>Numerator/denominator</th>
<th>Wilson 95%</th><th>Population</th></tr>{_metric_table(result)}</table>
<h2>Confusion matrix</h2><pre>{confusion}</pre>
<h2>False decisions</h2><table><tr><th>Row</th><th>Gold</th><th>Decision</th><th>Reuse</th><th>Title</th></tr>{errors}</table>
<h2>Version and fingerprint provenance</h2><pre>{html.escape(json.dumps(result.provenance.model_dump(mode='json'), indent=2))}</pre>
</body></html>"""


def _comparison_html(comparison: BenchmarkComparison) -> str:
    warnings = "".join(
        f"<li>{html.escape(reason)}</li>" for reason in comparison.reasons
    ) or "<li>None</li>"
    transitions = "".join(
        f"<tr><td>{html.escape(str(row['source_row_id']))}</td>"
        f"<td>{html.escape(str(row['baseline_decision']))}</td>"
        f"<td>{html.escape(str(row['candidate_decision']))}</td>"
        f"<td>{html.escape(str(row['baseline_reuse_status']))}</td>"
        f"<td>{html.escape(str(row['candidate_reuse_status']))}</td>"
        f"<td>{html.escape(str(row['claim_status']))}</td></tr>"
        for row in comparison.transitions
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>LitSync Comparison</title>
<style>body{{font:15px system-ui;margin:2rem}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:.45rem}}pre{{background:#f5f5f5;padding:1rem}}</style></head>
<body><h1>Benchmark comparison</h1><p>Valid: {comparison.valid}</p><ul>{warnings}</ul>
<h2>Metric deltas</h2><pre>{html.escape(json.dumps(comparison.metric_deltas, indent=2))}</pre>
<h2>Transitions</h2><table><tr><th>Row</th><th>Baseline</th><th>Candidate</th>
<th>Baseline reuse</th><th>Candidate reuse</th><th>Claim status</th></tr>{transitions}</table>
</body></html>"""


def stage_result_report(
    result: BenchmarkResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = _staging_root(output_dir)
    result_path = root / "benchmark-result.json"
    html_path = root / "benchmark-report.html"
    errors_path = root / "benchmark-errors.csv"
    _atomic_bytes(result_path, canonical_json_bytes(result.model_dump(mode="json")))
    _atomic_bytes(html_path, _result_html(result).encode("utf-8"))
    rows = [
        row for row in result.row_outcomes
        if row["source_row_id"] in (
            set(result.false_keep_source_row_ids)
            | set(result.false_reject_source_row_ids)
            | set(result.unsure_gold_source_row_ids)
        )
    ]
    _write_csv(
        errors_path,
        ["source_row_id", "gold_label", "decision", "reuse_status", "title"],
        rows,
    )
    return {"result": result_path, "html": html_path, "errors": errors_path}


def stage_comparison_report(
    comparison: BenchmarkComparison,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = _staging_root(output_dir)
    comparison_path = root / "benchmark-comparison.json"
    html_path = root / "benchmark-report.html"
    errors_path = root / "benchmark-errors.csv"
    _atomic_bytes(
        comparison_path,
        canonical_json_bytes(comparison.model_dump(mode="json")),
    )
    _atomic_bytes(html_path, _comparison_html(comparison).encode("utf-8"))
    error_rows = (
        comparison.newly_introduced_false_rejects
        + comparison.corrected_false_rejects
        + comparison.newly_introduced_false_keeps
        + comparison.corrected_false_keeps
    )
    _write_csv(
        errors_path,
        [
            "source_row_id", "gold_label", "baseline_decision",
            "candidate_decision", "baseline_reuse_status",
            "candidate_reuse_status", "change", "claim_status",
        ],
        error_rows,
    )
    return {"comparison": comparison_path, "html": html_path, "errors": errors_path}


def write_result_report(result: BenchmarkResult, output_dir: str | Path) -> dict[str, Path]:
    staged = stage_result_report(result, output_dir)
    return publish_staged_report(
        staged,
        output_dir,
        publication_kind="evaluation",
        benchmark_id=result.benchmark_id,
        benchmark_version=result.benchmark_version,
        benchmark_spec_fingerprint=result.benchmark_spec_fingerprint,
        job_ids=[result.job_id],
        verdict=result.gate.verdict.value,
    )


def write_comparison_report(
    comparison: BenchmarkComparison,
    output_dir: str | Path,
) -> dict[str, Path]:
    staged = stage_comparison_report(comparison, output_dir)
    return publish_staged_report(
        staged,
        output_dir,
        publication_kind="comparison",
        benchmark_id=comparison.benchmark_id,
        benchmark_version=comparison.benchmark_version,
        benchmark_spec_fingerprint=comparison.benchmark_spec_fingerprint,
        job_ids=comparison.job_ids,
        verdict="COMPARISON_VALID" if comparison.valid else "INVALID",
    )
