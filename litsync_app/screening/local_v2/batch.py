from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from .assessor import ASSESSOR_VERSION
from .compiler import COMPILER_VERSION
from .contracts import POLICY_VERSION, SCHEMA_VERSION, ScreeningProtocolV2, StrictModel
from .evidence import EVIDENCE_VALIDATION_VERSION
from .orchestrator import (
    ORCHESTRATOR_VERSION,
    LocalModelPlan,
    StructuredAssessmentEngine,
)
from .runner import (
    RUNNER_VERSION,
    LocalV2Paper,
    LocalV2PaperRunResult,
    run_compiled_local_v2_paper,
)


BATCH_RUNNER_VERSION = "local-v2-batch-v1"
BATCH_CHECKPOINT_VERSION = "local-v2-batch-checkpoint-v1"

CheckpointDisposition = Literal[
    "NOT_REQUESTED",
    "DISABLED",
    "NOT_FOUND",
    "LOADED",
    "IGNORED_INVALID",
    "IGNORED_MISMATCH",
]


class LocalV2BatchPipelineVersions(StrictModel):
    batch_runner_version: Literal[BATCH_RUNNER_VERSION]
    runner_version: Literal[RUNNER_VERSION]
    orchestrator_version: Literal[ORCHESTRATOR_VERSION]
    assessor_version: Literal[ASSESSOR_VERSION]
    evidence_validation_version: Literal[EVIDENCE_VALIDATION_VERSION]
    policy_version: Literal[POLICY_VERSION]
    schema_version: Literal[SCHEMA_VERSION]
    compiler_version: Literal[COMPILER_VERSION]


def current_local_v2_pipeline_versions() -> LocalV2BatchPipelineVersions:
    return LocalV2BatchPipelineVersions(
        batch_runner_version=BATCH_RUNNER_VERSION,
        runner_version=RUNNER_VERSION,
        orchestrator_version=ORCHESTRATOR_VERSION,
        assessor_version=ASSESSOR_VERSION,
        evidence_validation_version=EVIDENCE_VALIDATION_VERSION,
        policy_version=POLICY_VERSION,
        schema_version=SCHEMA_VERSION,
        compiler_version=COMPILER_VERSION,
    )


class LocalV2BatchCheckpointEntry(StrictModel):
    position: int = Field(ge=0)
    paper_fingerprint: str = Field(min_length=64, max_length=64)
    result: LocalV2PaperRunResult

    @model_validator(mode="after")
    def validate_fingerprint(self) -> "LocalV2BatchCheckpointEntry":
        expected = local_v2_paper_fingerprint(self.result.paper)
        if self.paper_fingerprint != expected:
            raise ValueError("checkpoint entry fingerprint must match its paper result")
        return self


class LocalV2BatchCheckpoint(StrictModel):
    checkpoint_version: Literal[BATCH_CHECKPOINT_VERSION]
    batch_runner_version: Literal[BATCH_RUNNER_VERSION]
    batch_id: str = Field(min_length=20, max_length=20)
    protocol_id: str = Field(min_length=1, max_length=80)
    model_plan: LocalModelPlan
    pipeline_versions: LocalV2BatchPipelineVersions
    paper_count: int = Field(ge=1)
    entries: list[LocalV2BatchCheckpointEntry] = Field(default_factory=list)
    complete: bool = False

    @model_validator(mode="after")
    def validate_checkpoint(self) -> "LocalV2BatchCheckpoint":
        positions = [entry.position for entry in self.entries]
        if positions != sorted(positions):
            raise ValueError("checkpoint entries must be ordered by position")
        if len(positions) != len(set(positions)):
            raise ValueError("checkpoint positions must be unique")
        if any(position >= self.paper_count for position in positions):
            raise ValueError("checkpoint position is outside the paper collection")
        if any(entry.result.protocol_id != self.protocol_id for entry in self.entries):
            raise ValueError("checkpoint result protocol_id must match the checkpoint")
        if self.complete != (len(self.entries) == self.paper_count):
            raise ValueError("checkpoint complete flag must match entry coverage")
        return self


class LocalV2BatchMetrics(StrictModel):
    total_papers: int = Field(ge=1)
    keep_count: int = Field(ge=0)
    maybe_count: int = Field(ge=0)
    reject_count: int = Field(ge=0)
    no_screenable_text_count: int = Field(ge=0)
    safe_fallback_count: int = Field(ge=0)
    resumable_result_count: int = Field(ge=0)
    resumed_count: int = Field(ge=0)
    fresh_count: int = Field(ge=0)
    model_call_count: int = Field(ge=0)
    fresh_model_call_count: int = Field(ge=0)
    review_used_count: int = Field(ge=0)
    validator_used_count: int = Field(ge=0)
    model_elapsed_seconds: float = Field(ge=0)
    fresh_model_elapsed_seconds: float = Field(ge=0)
    checkpoint_write_count: int = Field(ge=0)
    route_counts: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_metrics(self) -> "LocalV2BatchMetrics":
        if self.keep_count + self.maybe_count + self.reject_count != self.total_papers:
            raise ValueError("decision counts must cover every paper")
        if self.resumed_count + self.fresh_count != self.total_papers:
            raise ValueError("resumed and fresh counts must cover every paper")
        if self.no_screenable_text_count > self.maybe_count:
            raise ValueError("no-screenable-text papers must be MAYBE")
        if self.safe_fallback_count < self.no_screenable_text_count:
            raise ValueError("missing-text fallbacks must be included in fallback count")
        if self.resumable_result_count > self.total_papers:
            raise ValueError("resumable result count cannot exceed total papers")
        if self.fresh_model_call_count > self.model_call_count:
            raise ValueError("fresh model calls cannot exceed total model calls")
        if self.fresh_model_elapsed_seconds > self.model_elapsed_seconds + 0.0001:
            raise ValueError("fresh model time cannot exceed total recorded model time")
        if any(value < 0 for value in self.route_counts.values()):
            raise ValueError("route counts cannot be negative")
        expected_routes = self.total_papers - self.no_screenable_text_count
        if sum(self.route_counts.values()) != expected_routes:
            raise ValueError("route counts must cover every screenable paper")
        return self


class LocalV2BatchRunResult(StrictModel):
    batch_runner_version: Literal[BATCH_RUNNER_VERSION]
    batch_id: str = Field(min_length=20, max_length=20)
    protocol_id: str = Field(min_length=1, max_length=80)
    model_plan: LocalModelPlan
    pipeline_versions: LocalV2BatchPipelineVersions
    results: list[LocalV2PaperRunResult] = Field(min_length=1)
    metrics: LocalV2BatchMetrics
    checkpoint_path: str | None = Field(default=None, max_length=4000)
    checkpoint_disposition: CheckpointDisposition
    checkpoint_warnings: list[str] = Field(default_factory=list)
    resumed_positions: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_result(self) -> "LocalV2BatchRunResult":
        if len(self.results) != self.metrics.total_papers:
            raise ValueError("batch results must contain every paper")
        if any(result.protocol_id != self.protocol_id for result in self.results):
            raise ValueError("batch result protocol_id must match the batch")
        if self.resumed_positions != sorted(self.resumed_positions):
            raise ValueError("resumed positions must be ordered")
        if len(self.resumed_positions) != len(set(self.resumed_positions)):
            raise ValueError("resumed positions must be unique")
        if any(position >= len(self.results) for position in self.resumed_positions):
            raise ValueError("resumed position is outside the batch")
        if len(self.resumed_positions) != self.metrics.resumed_count:
            raise ValueError("resumed positions must match resumed_count")
        expected_metrics = _build_metrics(
            self.results,
            resumed_positions=set(self.resumed_positions),
            checkpoint_write_count=self.metrics.checkpoint_write_count,
        )
        if self.metrics != expected_metrics:
            raise ValueError("batch metrics must match the contained paper results")
        if self.checkpoint_path is None and self.checkpoint_disposition != "NOT_REQUESTED":
            raise ValueError("checkpoint disposition requires a checkpoint path")
        if self.checkpoint_path is not None and self.checkpoint_disposition == "NOT_REQUESTED":
            raise ValueError("checkpoint path cannot use NOT_REQUESTED disposition")
        return self


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_protocol_identity(protocol: ScreeningProtocolV2) -> None:
    if not protocol.protocol_id:
        raise ValueError("protocol must have a deterministic protocol_id")
    canonical = protocol.model_copy(update={"protocol_id": ""}).with_identity()
    if canonical.protocol_id != protocol.protocol_id:
        raise ValueError("protocol_id does not match the current protocol contents")


def _normalize_papers(
    papers: Iterable[LocalV2Paper | Mapping[str, Any]],
) -> list[LocalV2Paper]:
    normalized = [
        value if isinstance(value, LocalV2Paper) else LocalV2Paper.model_validate(value)
        for value in papers
    ]
    if not normalized:
        raise ValueError("at least one paper is required")
    paper_ids = [paper.paper_id for paper in normalized]
    duplicates = sorted(
        paper_id for paper_id, count in Counter(paper_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError("paper_id values must be unique: " + ", ".join(duplicates))
    return normalized


def local_v2_paper_fingerprint(paper: LocalV2Paper) -> str:
    payload = paper.model_dump(mode="json")
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_local_v2_batch_id(
    protocol: ScreeningProtocolV2,
    papers: Iterable[LocalV2Paper | Mapping[str, Any]],
    *,
    model_plan: LocalModelPlan | None = None,
) -> str:
    """Build an exact identity for protocol, model plan, paper order, and paper text."""

    _validate_protocol_identity(protocol)
    normalized = _normalize_papers(papers)
    plan = model_plan or LocalModelPlan()
    versions = current_local_v2_pipeline_versions()
    payload = {
        "pipeline_versions": versions.model_dump(mode="json"),
        "protocol_id": protocol.protocol_id,
        "model_plan": plan.model_dump(mode="json"),
        "papers": [paper.model_dump(mode="json") for paper in normalized],
    }
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:20]


def is_local_v2_result_resumable(result: LocalV2PaperRunResult) -> bool:
    """Return whether a saved result is stable enough to skip model work on resume.

    Deterministic missing-text results are resumable. Technical, structural, evidence,
    or disagreement fallbacks are deliberately retried on the next run.
    """

    if result.status == "NO_SCREENABLE_TEXT":
        return True
    if result.safe_fallback or result.final_policy.safe_fallback:
        return False
    orchestration = result.orchestration
    if orchestration is None or orchestration.safe_fallback:
        return False
    stages = [orchestration.primary, orchestration.reviewer, orchestration.validator]
    return all(stage.usable for stage in stages if stage is not None)


def _checkpoint_from_results(
    *,
    batch_id: str,
    protocol: ScreeningProtocolV2,
    model_plan: LocalModelPlan,
    papers: list[LocalV2Paper],
    results_by_position: Mapping[int, LocalV2PaperRunResult],
) -> LocalV2BatchCheckpoint:
    entries = [
        LocalV2BatchCheckpointEntry(
            position=position,
            paper_fingerprint=local_v2_paper_fingerprint(papers[position]),
            result=results_by_position[position],
        )
        for position in sorted(results_by_position)
    ]
    return LocalV2BatchCheckpoint(
        checkpoint_version=BATCH_CHECKPOINT_VERSION,
        batch_runner_version=BATCH_RUNNER_VERSION,
        batch_id=batch_id,
        protocol_id=protocol.protocol_id,
        model_plan=model_plan,
        pipeline_versions=current_local_v2_pipeline_versions(),
        paper_count=len(papers),
        entries=entries,
        complete=len(entries) == len(papers),
    )


def _fsync_parent_best_effort(path: Path) -> None:
    try:
        descriptor = os.open(str(path.parent), os.O_RDONLY)
    except (AttributeError, OSError):
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write_checkpoint(path: Path, checkpoint: LocalV2BatchCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    payload = _canonical_json(checkpoint.model_dump(mode="json")) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent_best_effort(path)
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def _checkpoint_matches_inputs(
    checkpoint: LocalV2BatchCheckpoint,
    *,
    batch_id: str,
    protocol: ScreeningProtocolV2,
    model_plan: LocalModelPlan,
    papers: list[LocalV2Paper],
) -> bool:
    if checkpoint.batch_id != batch_id:
        return False
    if checkpoint.protocol_id != protocol.protocol_id:
        return False
    if checkpoint.model_plan != model_plan:
        return False
    if checkpoint.pipeline_versions != current_local_v2_pipeline_versions():
        return False
    if checkpoint.paper_count != len(papers):
        return False
    for entry in checkpoint.entries:
        if entry.paper_fingerprint != local_v2_paper_fingerprint(papers[entry.position]):
            return False
        if entry.result.paper != papers[entry.position]:
            return False
    return True


def _load_resumable_results(
    path: Path,
    *,
    batch_id: str,
    protocol: ScreeningProtocolV2,
    model_plan: LocalModelPlan,
    papers: list[LocalV2Paper],
) -> tuple[
    dict[int, LocalV2PaperRunResult],
    CheckpointDisposition,
    list[str],
]:
    if not path.exists():
        return {}, "NOT_FOUND", []
    try:
        checkpoint = LocalV2BatchCheckpoint.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        return (
            {},
            "IGNORED_INVALID",
            [f"Checkpoint was invalid and was ignored: {str(exc)[:1000]}"],
        )

    if not _checkpoint_matches_inputs(
        checkpoint,
        batch_id=batch_id,
        protocol=protocol,
        model_plan=model_plan,
        papers=papers,
    ):
        return (
            {},
            "IGNORED_MISMATCH",
            [
                "Checkpoint identity did not match this exact protocol, model plan, "
                "and paper collection."
            ],
        )

    results: dict[int, LocalV2PaperRunResult] = {}
    nonresumable = 0
    for entry in checkpoint.entries:
        if is_local_v2_result_resumable(entry.result):
            results[entry.position] = entry.result
        else:
            nonresumable += 1

    warnings: list[str] = []
    if nonresumable:
        warnings.append(
            f"{nonresumable} saved fallback result(s) were deliberately "
            "scheduled for fresh model assessment."
        )
    return results, "LOADED", warnings


def _build_metrics(
    results: list[LocalV2PaperRunResult],
    *,
    resumed_positions: set[int],
    checkpoint_write_count: int,
) -> LocalV2BatchMetrics:
    decision_counts = Counter(result.final_policy.decision for result in results)
    route_counts = Counter(
        result.orchestration.route
        for result in results
        if result.orchestration is not None
    )
    fresh_positions = set(range(len(results))) - resumed_positions
    model_elapsed = round(sum(result.elapsed_seconds for result in results), 4)
    fresh_elapsed = round(
        sum(results[position].elapsed_seconds for position in fresh_positions),
        4,
    )
    return LocalV2BatchMetrics(
        total_papers=len(results),
        keep_count=decision_counts["KEEP"],
        maybe_count=decision_counts["MAYBE"],
        reject_count=decision_counts["REJECT"],
        no_screenable_text_count=sum(
            result.status == "NO_SCREENABLE_TEXT" for result in results
        ),
        safe_fallback_count=sum(result.safe_fallback for result in results),
        resumable_result_count=sum(
            is_local_v2_result_resumable(result) for result in results
        ),
        resumed_count=len(resumed_positions),
        fresh_count=len(results) - len(resumed_positions),
        model_call_count=sum(result.model_call_count for result in results),
        fresh_model_call_count=sum(
            results[position].model_call_count for position in fresh_positions
        ),
        review_used_count=sum(
            bool(result.orchestration and result.orchestration.review_used)
            for result in results
        ),
        validator_used_count=sum(
            bool(result.orchestration and result.orchestration.validator_used)
            for result in results
        ),
        model_elapsed_seconds=model_elapsed,
        fresh_model_elapsed_seconds=fresh_elapsed,
        checkpoint_write_count=checkpoint_write_count,
        route_counts=dict(sorted(route_counts.items())),
    )


def run_compiled_local_v2_batch(
    engine: StructuredAssessmentEngine,
    protocol: ScreeningProtocolV2,
    *,
    papers: Iterable[LocalV2Paper | Mapping[str, Any]],
    model_plan: LocalModelPlan | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    resume: bool = True,
    on_result: Callable[[int, LocalV2PaperRunResult, bool], None] | None = None,
) -> LocalV2BatchRunResult:
    """Screen an ordered paper collection sequentially with atomic resume checkpoints.

    The protocol and model plan are frozen into an exact batch identity. Model work is
    intentionally sequential so a local GPU is never loaded by competing papers. A
    checkpoint is replaced atomically after every fresh result. Stable semantic results
    and deterministic missing-text outcomes may resume; technical/safety fallbacks are
    always reassessed.
    """

    _validate_protocol_identity(protocol)
    normalized_papers = _normalize_papers(papers)
    plan = model_plan or LocalModelPlan()
    batch_id = build_local_v2_batch_id(
        protocol,
        normalized_papers,
        model_plan=plan,
    )

    resolved_checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpoint_warnings: list[str] = []
    checkpoint_write_count = 0

    if resolved_checkpoint is None:
        results_by_position: dict[int, LocalV2PaperRunResult] = {}
        disposition: CheckpointDisposition = "NOT_REQUESTED"
    elif not resume:
        results_by_position = {}
        disposition = "DISABLED"
    else:
        results_by_position, disposition, checkpoint_warnings = _load_resumable_results(
            resolved_checkpoint,
            batch_id=batch_id,
            protocol=protocol,
            model_plan=plan,
            papers=normalized_papers,
        )

    resumed_positions = set(results_by_position)
    if on_result is not None:
        for position in sorted(resumed_positions):
            on_result(position, results_by_position[position], True)

    if resolved_checkpoint is not None and disposition != "LOADED":
        empty_checkpoint = _checkpoint_from_results(
            batch_id=batch_id,
            protocol=protocol,
            model_plan=plan,
            papers=normalized_papers,
            results_by_position={},
        )
        _atomic_write_checkpoint(resolved_checkpoint, empty_checkpoint)
        checkpoint_write_count += 1

    for position, paper in enumerate(normalized_papers):
        if position in resumed_positions:
            continue
        result = run_compiled_local_v2_paper(
            engine,
            protocol,
            paper=paper,
            model_plan=plan,
        )
        results_by_position[position] = result

        if resolved_checkpoint is not None:
            checkpoint = _checkpoint_from_results(
                batch_id=batch_id,
                protocol=protocol,
                model_plan=plan,
                papers=normalized_papers,
                results_by_position=results_by_position,
            )
            _atomic_write_checkpoint(resolved_checkpoint, checkpoint)
            checkpoint_write_count += 1

        if on_result is not None:
            on_result(position, result, False)

    ordered_results = [results_by_position[position] for position in range(len(normalized_papers))]
    metrics = _build_metrics(
        ordered_results,
        resumed_positions=resumed_positions,
        checkpoint_write_count=checkpoint_write_count,
    )
    return LocalV2BatchRunResult(
        batch_runner_version=BATCH_RUNNER_VERSION,
        batch_id=batch_id,
        protocol_id=protocol.protocol_id,
        model_plan=plan,
        pipeline_versions=current_local_v2_pipeline_versions(),
        results=ordered_results,
        metrics=metrics,
        checkpoint_path=str(resolved_checkpoint) if resolved_checkpoint is not None else None,
        checkpoint_disposition=disposition,
        checkpoint_warnings=checkpoint_warnings,
        resumed_positions=sorted(resumed_positions),
    )
