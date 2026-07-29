from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .contracts import BenchmarkSpec, BenchmarkVerdict, validate_identifier
from .errors import PublicationError, SpecImmutabilityError
from .provenance import canonical_json_bytes
from .report import (
    COMPLETE_MARKER_NAME,
    create_completion_marker,
    marker_hash,
    publish_staged_report,
    validate_completion_directory,
)


REGISTRY_SCHEMA_VERSION = "litsync-benchmark-registry-v2"


def _registry_path(root: Path, spec: BenchmarkSpec) -> Path:
    benchmark_id = validate_identifier(spec.benchmark_id, label="benchmark ID")
    benchmark_version = validate_identifier(
        spec.benchmark_version, label="benchmark version"
    )
    resolved_root = root.resolve()
    path = (resolved_root / benchmark_id / f"{benchmark_version}.json").resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise SpecImmutabilityError("benchmark registry path escapes its root") from exc
    return path


def _read_registry(path: Path, spec: BenchmarkSpec) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecImmutabilityError("benchmark registry entry is unreadable") from exc
    if not isinstance(payload, dict):
        raise SpecImmutabilityError("benchmark registry entry must be an object")
    if payload.get("registry_schema_version") != REGISTRY_SCHEMA_VERSION:
        raise SpecImmutabilityError("unsupported benchmark registry schema")
    expected = {
        "registry_schema_version", "benchmark_id", "benchmark_version",
        "benchmark_spec_fingerprint", "completed_publications",
        "pending_publication",
    }
    if set(payload) != expected:
        raise SpecImmutabilityError("benchmark registry fields are invalid")
    if (
        payload.get("benchmark_id") != spec.benchmark_id
        or payload.get("benchmark_version") != spec.benchmark_version
        or payload.get("benchmark_spec_fingerprint")
        != spec.benchmark_spec_fingerprint
    ):
        raise SpecImmutabilityError(
            "benchmark ID/version is already locked to a different specification"
        )
    if not isinstance(payload.get("completed_publications"), list):
        raise SpecImmutabilityError("benchmark registry history is invalid")
    pending = payload.get("pending_publication")
    if pending is not None and not isinstance(pending, dict):
        raise SpecImmutabilityError("benchmark pending publication is invalid")
    return payload


def _empty_registry(spec: BenchmarkSpec) -> dict[str, Any]:
    return {
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "benchmark_id": spec.benchmark_id,
        "benchmark_version": spec.benchmark_version,
        "benchmark_spec_fingerprint": spec.benchmark_spec_fingerprint,
        "completed_publications": [],
        "pending_publication": None,
    }


def _write_registry(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    os.replace(temporary, path)


def _validated_reference(reference: Any) -> tuple[dict[str, Any], dict[str, Path]]:
    required = {
        "publication_id", "publication_kind", "job_ids", "verdict",
        "completion_marker_path", "completion_marker_sha256",
    }
    if not isinstance(reference, dict) or set(reference) != required:
        raise SpecImmutabilityError("benchmark registry publication reference is invalid")
    if (
        not isinstance(reference["publication_id"], str)
        or not isinstance(reference["publication_kind"], str)
        or not isinstance(reference["job_ids"], list)
        or any(not isinstance(value, str) for value in reference["job_ids"])
        or not isinstance(reference["verdict"], str)
        or not isinstance(reference["completion_marker_path"], str)
        or not isinstance(reference["completion_marker_sha256"], str)
    ):
        raise SpecImmutabilityError("benchmark registry publication reference is malformed")
    marker_path = Path(reference["completion_marker_path"])
    if marker_path.name != COMPLETE_MARKER_NAME:
        raise SpecImmutabilityError("benchmark registry marker path is invalid")
    try:
        marker, paths = validate_completion_directory(
            marker_path.parent,
            expected_kind=reference["publication_kind"],
        )
    except PublicationError as exc:
        raise SpecImmutabilityError(
            "completed benchmark registry publication is corrupt"
        ) from exc
    if (
        marker["publication_id"] != reference["publication_id"]
        or marker["job_ids"] != reference["job_ids"]
        or marker["verdict"] != reference["verdict"]
        or marker_hash(marker_path.parent) != reference["completion_marker_sha256"]
    ):
        raise SpecImmutabilityError(
            "completed benchmark registry publication does not match its marker"
        )
    return marker, paths


def _reference_identity(reference: Any) -> tuple[Any, ...] | None:
    required = {
        "publication_id", "publication_kind", "job_ids", "verdict",
        "completion_marker_path", "completion_marker_sha256",
    }
    if not isinstance(reference, dict) or set(reference) != required:
        return None
    if not isinstance(reference.get("job_ids"), list):
        return None
    return (
        reference.get("publication_id"),
        reference.get("publication_kind"),
        reference.get("job_ids"),
        reference.get("verdict"),
        reference.get("completion_marker_path"),
    )


def check_registry(
    spec: BenchmarkSpec,
    registry_dir: str | Path,
    *,
    allow_pending_retry: bool = False,
) -> None:
    path = _registry_path(Path(registry_dir), spec)
    payload = _read_registry(path, spec)
    if payload is None:
        return
    seen: set[str] = set()
    for reference in payload["completed_publications"]:
        marker, _ = _validated_reference(reference)
        if (
            marker["benchmark_id"] != spec.benchmark_id
            or marker["benchmark_version"] != spec.benchmark_version
            or marker["benchmark_spec_fingerprint"]
            != spec.benchmark_spec_fingerprint
        ):
            raise SpecImmutabilityError("registered publication has a different spec")
        if reference["publication_id"] in seen:
            raise SpecImmutabilityError("benchmark registry history contains duplicates")
        seen.add(reference["publication_id"])
    if payload["pending_publication"] is not None and not allow_pending_retry:
        raise SpecImmutabilityError(
            "benchmark registry contains an incomplete pending publication"
        )


def _reference(
    marker: dict[str, Any],
    output_dir: Path,
    marker_sha256: str,
) -> dict[str, Any]:
    return {
        "publication_id": marker["publication_id"],
        "publication_kind": marker["publication_kind"],
        "job_ids": marker["job_ids"],
        "verdict": marker["verdict"],
        "completion_marker_path": str(output_dir / COMPLETE_MARKER_NAME),
        "completion_marker_sha256": marker_sha256,
    }


def _acquire_lock(path: Path) -> tuple[int, Path]:
    lock = path.with_suffix(path.suffix + ".lock")
    try:
        descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise PublicationError(
            f"benchmark registry is busy or has a stale lock: {lock}"
        ) from exc
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    return descriptor, lock


def publish_completed_evaluation(
    spec: BenchmarkSpec,
    verdict: BenchmarkVerdict | str,
    staged_paths: dict[str, Path],
    output_dir: str | Path,
    registry_dir: str | Path,
    *,
    result_key: str,
    publication_kind: str = "evaluation",
    job_ids: list[str] | None = None,
) -> dict[str, Path]:
    verdict_value = verdict.value if isinstance(verdict, BenchmarkVerdict) else str(verdict)
    if verdict_value == BenchmarkVerdict.INVALID.value:
        raise PublicationError("invalid publications cannot be registered")
    output = Path(output_dir).resolve()
    jobs = list(job_ids or [])
    if not jobs:
        raise PublicationError("registered publication requires job IDs")
    marker = create_completion_marker(
        staged_paths,
        output,
        publication_kind=publication_kind,
        benchmark_id=spec.benchmark_id,
        benchmark_version=spec.benchmark_version,
        benchmark_spec_fingerprint=spec.benchmark_spec_fingerprint,
        job_ids=jobs,
        verdict=verdict_value,
    )
    stage = next(iter(staged_paths.values())).parent
    staged_marker_hash = marker_hash(stage)
    reference = _reference(marker, output, staged_marker_hash)

    root = Path(registry_dir).resolve()
    registry_path = _registry_path(root, spec)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, lock = _acquire_lock(registry_path)
    try:
        os.close(descriptor)
        payload = _read_registry(registry_path, spec) or _empty_registry(spec)
        for completed in payload["completed_publications"]:
            completed_marker, completed_paths = _validated_reference(completed)
            if completed["publication_id"] == reference["publication_id"]:
                if completed_marker["artifacts"] != marker["artifacts"]:
                    raise PublicationError("registered publication identity collision")
                shutil.rmtree(stage)
                return completed_paths

        pending = payload["pending_publication"]
        if pending is not None:
            if _reference_identity(pending) != _reference_identity(reference):
                raise PublicationError(
                    "a different incomplete publication requires explicit recovery"
                )
            if output.exists() and not output.is_dir():
                raise PublicationError("pending publication output is not a directory")
            if output.exists() and any(output.iterdir()):
                try:
                    recovered_marker, recovered_paths = validate_completion_directory(
                        output, expected_kind=publication_kind
                    )
                except PublicationError as exc:
                    raise PublicationError(
                        "matching pending publication has invalid incomplete output"
                    ) from exc
                if recovered_marker["publication_id"] != reference["publication_id"]:
                    raise PublicationError(
                        "pending publication output does not match the retry"
                    )
                recovered_reference = _reference(
                    recovered_marker, output, marker_hash(output)
                )
                if recovered_reference != pending:
                    raise PublicationError(
                        "pending publication marker hash or metadata does not match"
                    )
                payload["completed_publications"].append(recovered_reference)
                payload["pending_publication"] = None
                _write_registry(registry_path, payload)
                shutil.rmtree(stage)
                return recovered_paths
            payload["pending_publication"] = None

        payload["pending_publication"] = reference
        _write_registry(registry_path, payload)
        published = publish_staged_report(
            staged_paths,
            output,
            publication_kind=publication_kind,
            benchmark_id=spec.benchmark_id,
            benchmark_version=spec.benchmark_version,
            benchmark_spec_fingerprint=spec.benchmark_spec_fingerprint,
            job_ids=jobs,
            verdict=verdict_value,
            prepared_marker=marker,
        )
        published_marker, _ = validate_completion_directory(
            output, expected_kind=publication_kind
        )
        committed_reference = _reference(
            published_marker, output, marker_hash(output)
        )
        payload["pending_publication"] = committed_reference
        _write_registry(registry_path, payload)
        payload["completed_publications"].append(committed_reference)
        payload["pending_publication"] = None
        _write_registry(registry_path, payload)
        if result_key not in published:
            raise PublicationError("completed result artifact was not published")
        return published
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        lock.unlink(missing_ok=True)
