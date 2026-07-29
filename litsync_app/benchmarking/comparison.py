from __future__ import annotations

from typing import Any

from .contracts import BenchmarkComparison, BenchmarkResult
from .errors import ComparisonError


def _is_false_reject(row: dict[str, Any]) -> bool:
    return row["gold_label"] == "KEEP" and row["decision"] == "REJECT"


def _is_false_keep(row: dict[str, Any]) -> bool:
    return row["gold_label"] == "REJECT" and row["decision"] == "KEEP"


def _transition_claim(
    source_id: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    change: str,
) -> dict[str, Any]:
    fresh = (
        baseline["reuse_status"] == "FRESH"
        and candidate["reuse_status"] == "FRESH"
    )
    return {
        "source_row_id": source_id,
        "gold_label": baseline["gold_label"],
        "baseline_decision": baseline["decision"],
        "candidate_decision": candidate["decision"],
        "baseline_reuse_status": baseline["reuse_status"],
        "candidate_reuse_status": candidate["reuse_status"],
        "change": change,
        "claim_status": (
            "CLAIMED_FRESH_CHANGE"
            if fresh
            else "OBSERVED_TRANSITION_WITH_REUSE"
        ),
    }


def _compare_pair(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
) -> dict[str, Any]:
    baseline_rows = {
        str(row["source_row_id"]): row for row in baseline.row_outcomes
    }
    candidate_rows = {
        str(row["source_row_id"]): row for row in candidate.row_outcomes
    }
    common = [
        source_id
        for source_id in baseline.resolved_gold_source_row_ids
        if source_id in candidate_rows
    ]
    movement = {
        old: {new: 0 for new in ("KEEP", "MAYBE", "REJECT")}
        for old in ("KEEP", "MAYBE", "REJECT")
    }
    transitions: list[dict[str, Any]] = []
    categories = {
        "new_false_rejects": [],
        "corrected_false_rejects": [],
        "new_false_keeps": [],
        "corrected_false_keeps": [],
    }
    for source_id in common:
        old = baseline_rows[source_id]
        new = candidate_rows[source_id]
        movement[old["decision"]][new["decision"]] += 1
        if old["decision"] != new["decision"]:
            transitions.append(_transition_claim(source_id, old, new, "decision_transition"))
        if not _is_false_reject(old) and _is_false_reject(new):
            categories["new_false_rejects"].append(
                _transition_claim(source_id, old, new, "new_false_reject")
            )
        if _is_false_reject(old) and not _is_false_reject(new):
            categories["corrected_false_rejects"].append(
                _transition_claim(source_id, old, new, "corrected_false_reject")
            )
        if not _is_false_keep(old) and _is_false_keep(new):
            categories["new_false_keeps"].append(
                _transition_claim(source_id, old, new, "new_false_keep")
            )
        if _is_false_keep(old) and not _is_false_keep(new):
            categories["corrected_false_keeps"].append(
                _transition_claim(source_id, old, new, "corrected_false_keep")
            )
    deltas: dict[str, float | int | None] = {}
    for name in sorted(set(baseline.metrics) & set(candidate.metrics)):
        old_value = baseline.metrics[name].value
        new_value = candidate.metrics[name].value
        deltas[name] = (
            round(float(new_value) - float(old_value), 6)
            if old_value is not None and new_value is not None
            else None
        )
    return {
        "baseline_job_id": baseline.job_id,
        "candidate_job_id": candidate.job_id,
        "metric_deltas": deltas,
        "movement_matrix": movement,
        "transitions": transitions,
        **categories,
    }


def compare_results(results: list[BenchmarkResult]) -> BenchmarkComparison:
    if len(results) < 2:
        raise ComparisonError("comparison requires at least two completed results")
    baseline = results[0]
    reasons: list[str] = []
    for result in results[1:]:
        if (
            result.benchmark_id != baseline.benchmark_id
            or result.benchmark_version != baseline.benchmark_version
            or result.benchmark_spec_fingerprint
            != baseline.benchmark_spec_fingerprint
        ):
            reasons.append("runs use different benchmark identities")
        if result.gold_file_fingerprint != baseline.gold_file_fingerprint:
            reasons.append("runs use different frozen gold")
        if (
            result.resolved_gold_source_row_ids
            != baseline.resolved_gold_source_row_ids
        ):
            reasons.append("runs use different resolved-gold populations")
        if (
            result.provenance.source_dataset_fingerprint
            != baseline.provenance.source_dataset_fingerprint
        ):
            reasons.append("runs use different source datasets")
        if (
            result.provenance.run_selected_source_row_ids
            != baseline.provenance.run_selected_source_row_ids
        ):
            reasons.append("runs use different screened populations")
        if (
            result.provenance.protocol_id != baseline.provenance.protocol_id
            or result.provenance.protocol_cache_version
            != baseline.provenance.protocol_cache_version
        ):
            reasons.append("runs use different protocol identities")
        if not result.provenance.valid or not baseline.provenance.valid:
            reasons.append("invalid provenance prevents improvement claims")
    pairwise = (
        [_compare_pair(baseline, candidate) for candidate in results[1:]]
        if not reasons
        else []
    )
    final = pairwise[-1] if pairwise else {
        "metric_deltas": {},
        "movement_matrix": {},
        "transitions": [],
        "new_false_rejects": [],
        "corrected_false_rejects": [],
        "new_false_keeps": [],
        "corrected_false_keeps": [],
    }
    return BenchmarkComparison(
        benchmark_id=baseline.benchmark_id,
        benchmark_version=baseline.benchmark_version,
        benchmark_spec_fingerprint=baseline.benchmark_spec_fingerprint,
        job_ids=[result.job_id for result in results],
        valid=not reasons,
        reasons=sorted(set(reasons)),
        metric_deltas=final["metric_deltas"],
        movement_matrix=final["movement_matrix"],
        transitions=final["transitions"],
        newly_introduced_false_rejects=final["new_false_rejects"],
        corrected_false_rejects=final["corrected_false_rejects"],
        newly_introduced_false_keeps=final["new_false_keeps"],
        corrected_false_keeps=final["corrected_false_keeps"],
        pairwise_comparisons=pairwise,
    )
