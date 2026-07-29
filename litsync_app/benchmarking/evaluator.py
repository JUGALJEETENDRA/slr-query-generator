from __future__ import annotations

import json
import math
from typing import Any

from .contracts import (
    BenchmarkResult,
    ConfidenceInterval,
    MetricValue,
)
from .loader import LoadedGold, LoadedRun, LoadedSpec
from .release_gate import evaluate_gate
from .provenance import finite_number


METRIC_DEFINITION_VERSIONS = {
    "resolved_sample_size": "resolved-sample-v1",
    "gold_keep_count": "gold-count-v1",
    "gold_reject_count": "gold-count-v1",
    "model_keep_count": "model-count-v1",
    "model_maybe_count": "model-count-v1",
    "model_reject_count": "model-count-v1",
    "keep_or_maybe_recall": "keep-or-maybe-recall-v1",
    "false_reject_rate": "false-reject-rate-v1",
    "definitive_keep_precision": "definitive-keep-precision-v1",
    "definitive_reject_precision": "definitive-reject-precision-v1",
    "specificity": "specificity-v1",
    "manual_review_burden_full_run": "manual-review-full-run-v1",
    "manual_review_burden_resolved_gold": "manual-review-resolved-gold-v1",
    "exact_evidence_rate": "exact-evidence-v1",
    "structurally_validated_rate": "structural-validation-v1",
    "invalid_definitive_decisions": "invalid-definitive-v1",
    "runtime_seconds": "summary-observation-v1",
    "papers_per_minute": "summary-observation-v1",
    "retry_count": "summary-observation-v1",
    "primary_batches": "summary-observation-v1",
    "verification_batches": "summary-observation-v1",
    "structured_terminal_fallbacks": "summary-observation-v1",
    "technical_fallbacks": "summary-observation-v1",
    "verification_requests": "summary-observation-v1",
    "verification_agreements": "summary-observation-v1",
    "verification_disagreements": "summary-observation-v1",
    "semantic_validation_failures": "summary-observation-v1",
    "replay_attempts": "summary-observation-v1",
    "replay_recoveries": "summary-observation-v1",
    "resumed_rows": "run-provenance-v1",
    "assessment_cache_hits": "run-provenance-v1",
    "fresh_primary_rows": "run-provenance-v1",
    "directly_handled_without_primary_rows": "run-provenance-v1",
}


def validate_metric_definitions(definitions: dict[str, str]) -> None:
    unknown = sorted(set(definitions) - set(METRIC_DEFINITION_VERSIONS))
    missing = sorted(set(METRIC_DEFINITION_VERSIONS) - set(definitions))
    mismatched = sorted(
        name
        for name, version in definitions.items()
        if METRIC_DEFINITION_VERSIONS.get(name) != version
    )
    if unknown or missing or mismatched:
        raise ValueError(
            "unsupported metric definitions; "
            f"unknown={unknown}, missing={missing}, mismatched={mismatched}"
        )


def _wilson(successes: int, total: int) -> ConfidenceInterval | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    estimate = successes / total
    denominator = 1 + z * z / total
    center = (estimate + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            estimate * (1 - estimate) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return ConfidenceInterval(
        lower=round(max(0.0, center - margin), 6),
        upper=round(min(1.0, center + margin), 6),
    )


def _rate(
    name: str,
    numerator: int,
    denominator: int,
    population: str,
    source: str = "derived",
    field: str = "",
) -> MetricValue:
    return MetricValue(
        numerator=numerator,
        denominator=denominator,
        value=round(numerator / denominator, 6) if denominator else None,
        confidence_interval=_wilson(numerator, denominator),
        population_scope=population,
        metric_definition_version=METRIC_DEFINITION_VERSIONS[name],
        artifact_source=source,
        artifact_field=field or name,
    )


def _count(
    name: str,
    value: int | float,
    population: str,
    source: str,
    field: str,
) -> MetricValue:
    return MetricValue(
        numerator=value,
        denominator=1,
        value=value,
        confidence_interval=None,
        population_scope=population,
        metric_definition_version=METRIC_DEFINITION_VERSIONS[name],
        artifact_source=source,
        artifact_field=field,
    )


def _row_reuse(run: LoadedRun, source_id: str) -> str:
    if source_id in run.provenance.resumed_source_row_ids:
        return "RESUMED"
    if source_id in run.provenance.cache_hit_source_row_ids:
        return "CACHE_HIT"
    if source_id in run.provenance.fresh_primary_source_row_ids:
        return "FRESH"
    if source_id in run.provenance.directly_handled_without_primary_source_row_ids:
        return "DIRECT"
    return "UNKNOWN"


def _evidence_metrics(run: LoadedRun) -> tuple[int, int, int, int]:
    exact = total = invalid_payloads = validated = 0
    for row in run.rows.values():
        validated += int(row.get("Validation_Status", "").lower() == "validated")
        try:
            evidence = json.loads(row.get("Evidence_JSON") or "[]")
            if not isinstance(evidence, list):
                raise ValueError
        except (json.JSONDecodeError, TypeError, ValueError):
            invalid_payloads += 1
            evidence = []
        for span in evidence:
            if not isinstance(span, dict):
                invalid_payloads += 1
                continue
            total += 1
            source = (
                row.get("Title", "")
                if span.get("source") == "title"
                else row.get("Abstract", "")
            )
            quote = str(span.get("quote") or "")
            exact += int(bool(quote) and quote in source)
    return exact, total, invalid_payloads, validated


def evaluate_run(
    loaded_spec: LoadedSpec,
    gold: LoadedGold,
    run: LoadedRun,
) -> BenchmarkResult:
    spec = loaded_spec.spec
    validate_metric_definitions(spec.metric_definitions)
    confusion = {
        gold_label: {decision: 0 for decision in ("KEEP", "MAYBE", "REJECT")}
        for gold_label in ("KEEP", "REJECT")
    }
    row_outcomes: list[dict[str, Any]] = []
    false_keeps: list[str] = []
    false_rejects: list[str] = []
    for source_id in spec.gold_selected_source_row_ids:
        gold_label = gold.rows[source_id]["Gold_Decision"]
        decision = run.rows[source_id]["Decision"]
        reuse = _row_reuse(run, source_id)
        row_outcomes.append({
            "source_row_id": source_id,
            "gold_label": gold_label,
            "decision": decision,
            "reuse_status": reuse,
            "title": run.rows[source_id].get("Title", ""),
        })
        if gold_label == "UNSURE":
            continue
        confusion[gold_label][decision] += 1
        if gold_label == "REJECT" and decision == "KEEP":
            false_keeps.append(source_id)
        if gold_label == "KEEP" and decision == "REJECT":
            false_rejects.append(source_id)

    resolved = len(gold.resolved_ids)
    gold_keep = sum(confusion["KEEP"].values())
    gold_reject = sum(confusion["REJECT"].values())
    resolved_model = {
        decision: sum(confusion[label][decision] for label in confusion)
        for decision in ("KEEP", "MAYBE", "REJECT")
    }
    full_model = {
        decision: sum(row["Decision"] == decision for row in run.rows.values())
        for decision in ("KEEP", "MAYBE", "REJECT")
    }
    exact, evidence_total, invalid_payloads, validated = _evidence_metrics(run)
    invalid_definitive = sum(
        row["Decision"] in {"KEEP", "REJECT"}
        and row.get("Validation_Status", "").lower() != "validated"
        for row in run.rows.values()
    )
    metrics: dict[str, MetricValue] = {
        "resolved_sample_size": _count(
            "resolved_sample_size", resolved, "resolved_gold", "gold_csv", "Gold_Decision"
        ),
        "gold_keep_count": _count(
            "gold_keep_count", gold_keep, "resolved_gold", "gold_csv", "Gold_Decision"
        ),
        "gold_reject_count": _count(
            "gold_reject_count", gold_reject, "resolved_gold", "gold_csv", "Gold_Decision"
        ),
        "model_keep_count": _count(
            "model_keep_count", full_model["KEEP"], "full_run", "screening_csv", "Decision"
        ),
        "model_maybe_count": _count(
            "model_maybe_count", full_model["MAYBE"], "full_run", "screening_csv", "Decision"
        ),
        "model_reject_count": _count(
            "model_reject_count", full_model["REJECT"], "full_run", "screening_csv", "Decision"
        ),
        "keep_or_maybe_recall": _rate(
            "keep_or_maybe_recall",
            confusion["KEEP"]["KEEP"] + confusion["KEEP"]["MAYBE"],
            gold_keep,
            "resolved_gold_keep",
        ),
        "false_reject_rate": _rate(
            "false_reject_rate", confusion["KEEP"]["REJECT"], gold_keep,
            "resolved_gold_keep",
        ),
        "definitive_keep_precision": _rate(
            "definitive_keep_precision", confusion["KEEP"]["KEEP"],
            resolved_model["KEEP"], "resolved_gold_model_keep",
        ),
        "definitive_reject_precision": _rate(
            "definitive_reject_precision", confusion["REJECT"]["REJECT"],
            resolved_model["REJECT"], "resolved_gold_model_reject",
        ),
        "specificity": _rate(
            "specificity", confusion["REJECT"]["REJECT"], gold_reject,
            "resolved_gold_reject",
        ),
        "manual_review_burden_full_run": _rate(
            "manual_review_burden_full_run", full_model["MAYBE"], len(run.rows),
            "full_run",
        ),
        "manual_review_burden_resolved_gold": _rate(
            "manual_review_burden_resolved_gold", resolved_model["MAYBE"], resolved,
            "resolved_gold",
        ),
        "exact_evidence_rate": _rate(
            "exact_evidence_rate", exact, evidence_total, "full_run_evidence_spans",
            "screening_csv", "Evidence_JSON",
        ),
        "structurally_validated_rate": _rate(
            "structurally_validated_rate", validated, len(run.rows), "full_run",
            "screening_csv", "Validation_Status",
        ),
        "invalid_definitive_decisions": _count(
            "invalid_definitive_decisions", invalid_definitive, "full_run",
            "screening_csv", "Decision+Validation_Status",
        ),
    }
    summary_fields = {
        "runtime_seconds": "runtime_seconds",
        "papers_per_minute": "papers_per_minute",
        "retry_count": "retry_count",
        "primary_batches": "primary_batches_submitted",
        "verification_batches": "verification_batches_submitted",
        "technical_fallbacks": "technical_fallback_count",
        "verification_requests": "verification_papers_requested",
        "verification_agreements": "verification_validated_agreements",
        "verification_disagreements": "verification_validated_disagreements",
        "semantic_validation_failures": "verification_semantic_validation_failures",
        "replay_attempts": "degraded_subgroup_replay_count",
        "replay_recoveries": "degraded_subgroup_replay_success_count",
    }
    for metric_name, field_name in summary_fields.items():
        metrics[metric_name] = _count(
            metric_name,
            finite_number(run.summary.get(field_name, 0)),
            "full_run",
            "diagnostic_summary",
            field_name,
        )
    detector_outcomes = run.summary.get("detector_outcomes")
    if not isinstance(detector_outcomes, dict):
        detector_outcomes = {}
    metrics["structured_terminal_fallbacks"] = _count(
        "structured_terminal_fallbacks",
        finite_number(detector_outcomes.get("structured_output_terminal", 0)),
        "full_run",
        "diagnostic_summary",
        "detector_outcomes.structured_output_terminal",
    )
    for name, values, field in (
        ("resumed_rows", run.provenance.resumed_source_row_ids, "resumed_source_row_ids"),
        ("assessment_cache_hits", run.provenance.cache_hit_source_row_ids, "assessment_cache_hit_source_row_ids"),
        ("fresh_primary_rows", run.provenance.fresh_primary_source_row_ids, "fresh_primary_source_row_ids"),
        (
            "directly_handled_without_primary_rows",
            run.provenance.directly_handled_without_primary_source_row_ids,
            "directly_handled_without_primary_source_row_ids",
        ),
    ):
        metrics[name] = _count(
            name, len(values), "full_run", "diagnostic_summary", field
        )
    result = BenchmarkResult(
        benchmark_id=spec.benchmark_id,
        benchmark_version=spec.benchmark_version,
        job_id=run.job_id,
        benchmark_spec_fingerprint=spec.benchmark_spec_fingerprint,
        gold_file_fingerprint=gold.file_fingerprint,
        resolved_gold_source_row_ids=gold.resolved_ids,
        unsure_gold_source_row_ids=gold.unsure_ids,
        provenance=run.provenance,
        metrics=metrics,
        confusion_matrix=confusion,
        false_keep_source_row_ids=false_keeps,
        false_reject_source_row_ids=false_rejects,
        row_outcomes=row_outcomes,
        gate={"verdict": "INVALID", "failures": []},
    )
    result.gate = evaluate_gate(spec, result)
    return result
