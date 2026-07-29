from __future__ import annotations

import json
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from litsync_app.benchmarking.cli import main as benchmark_main
from litsync_app.benchmarking.comparison import compare_results
from litsync_app.benchmarking.contracts import BenchmarkVerdict, ProvenanceClass
from litsync_app.benchmarking.errors import (
    GoldValidationError,
    PublicationError,
    RunArtifactError,
    SpecImmutabilityError,
    SpecValidationError,
)
from litsync_app.benchmarking.evaluator import (
    METRIC_DEFINITION_VERSIONS,
    evaluate_run,
)
from litsync_app.benchmarking.loader import load_gold, load_run, load_spec
from litsync_app.benchmarking.provenance import (
    canonical_fingerprint,
    file_fingerprint,
    screening_output_fingerprint,
    source_dataset_fingerprint,
)
from litsync_app.benchmarking.registry import (
    REGISTRY_SCHEMA_VERSION,
    check_registry,
    publish_completed_evaluation,
)
from litsync_app.benchmarking.report import (
    COMPLETE_MARKER_NAME,
    marker_hash,
    stage_result_report,
    validate_completion_directory,
    write_comparison_report,
    write_result_report,
)
import litsync_app.benchmarking.registry as registry_module


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _spec_fingerprint(payload: dict) -> str:
    value = dict(payload)
    value.pop("benchmark_spec_fingerprint", None)
    return canonical_fingerprint(value)


def _fixture(
    tmp_path: Path,
    *,
    job_id="cold-job",
    run_count: int = 5,
    gold_count: int = 4,
) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifacts = tmp_path / "outputs"
    dataset = tmp_path / "source.csv"
    source = pd.DataFrame({
        "Title": [f"Study {index}" for index in range(run_count)],
        "Abstract": [f"Measured analysis {index}" for index in range(run_count)],
        "Stable_Code": [f"{index:03d}" for index in range(run_count)],
    })
    source.to_csv(dataset, index=False, encoding="utf-8-sig")
    run_ids = [str(index) for index in range(run_count)]
    gold_ids = run_ids[:gold_count]
    gold_labels = (
        ["KEEP", "KEEP", "REJECT", "UNSURE"]
        if gold_count == 4
        else [
            "UNSURE" if index == gold_count - 1 else (
                "KEEP" if index % 2 == 0 else "REJECT"
            )
            for index in range(gold_count)
        ]
    )
    gold_path = tmp_path / "gold.csv"
    pd.DataFrame([
        {
            "Source_Row_Index": source_id,
            "Title": source.loc[int(source_id), "Title"],
            "Abstract": source.loc[int(source_id), "Abstract"],
            "Gold_Decision": label,
        }
        for source_id, label in zip(gold_ids, gold_labels)
    ]).to_csv(gold_path, index=False, encoding="utf-8-sig")
    protocol_id = "protocol-phase3a"
    rows = []
    decisions = ["KEEP", "MAYBE", "REJECT", "KEEP", "REJECT"]
    for index, source_id in enumerate(run_ids):
        decision = decisions[index % len(decisions)]
        title = source.loc[int(source_id), "Title"]
        rows.append({
            "Title": title,
            "Abstract": source.loc[int(source_id), "Abstract"],
            "Decision": decision,
            "Validation_Status": "validated",
            "Evidence_JSON": json.dumps([{
                "criterion_id": "criterion",
                "source": "title",
                "evidence_id": "title_001",
                "quote": title,
            }]),
            "Criteria_JSON": "[]",
            "Protocol_ID": protocol_id,
            "Prompt_Version": "gemini-web-v2.4-assessment-prompt-v5",
            "Route_Used": "primary",
            "Verification_Status": "not_required",
            "Failure_Class": "",
            "Cache_Hit": "False",
            "Source_Row_Index": source_id,
            "Execution_Origin": "fresh_primary",
            "Direct_Handling_Reason": "",
        })
    screening_input = file_fingerprint(dataset)
    dataset_fingerprint = source_dataset_fingerprint(dataset)
    csv_path = artifacts / "runs" / f"screened-{job_id}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary = {
        "job_id": job_id,
        "run_status": "complete",
        "run_selected_count": len(run_ids),
        "run_selected_source_row_ids": run_ids,
        "resumed_count": 0,
        "resumed_source_row_ids": [],
        "assessment_cache_hits_loaded": 0,
        "assessment_cache_hit_source_row_ids": [],
        "fresh_primary_papers": len(run_ids),
        "fresh_primary_source_row_ids": run_ids,
        "directly_handled_without_primary_count": 0,
        "directly_handled_without_primary_source_row_ids": [],
        "direct_handling_reasons": {},
        "missing_abstract_count": 0,
        "missing_abstract_source_row_ids": [],
        "runtime_seconds": 50,
        "papers_per_minute": 6,
        "retry_count": 1,
        "primary_batches_submitted": 1,
        "primary_papers_requested": len(run_ids),
        "verification_batches_submitted": 0,
        "verification_papers_requested": 0,
        "technical_fallback_count": 0,
        "verification_validated_agreements": 0,
        "verification_validated_disagreements": 0,
        "verification_semantic_validation_failures": 0,
        "degraded_subgroup_replay_count": 0,
        "degraded_subgroup_replay_success_count": 0,
        "detector_outcomes": {"structured_output_terminal": 0},
        "route_counts": {"primary": len(run_ids)},
        "verification_outcomes": {"not_required": len(run_ids)},
        "diagnostics_path": str(
            artifacts / "cache" / "gemini_web_v24" / "diagnostics" / f"{job_id}.jsonl"
        ),
        "source_dataset_fingerprint": dataset_fingerprint,
        "screening_input_fingerprint": screening_input,
        "screening_output_fingerprint": screening_output_fingerprint(rows),
        "architecture_version": "gemini-web-batched-v2.4",
        "protocol_id": protocol_id,
        "protocol_cache_version": "gemini-web-v2.4-protocol-v3",
        "assessment_prompt_version": "gemini-web-v2.4-assessment-prompt-v5",
        "assessment_cache_version": "gemini-web-v2.4-assessment-v5",
    }
    diagnostic_root = artifacts / "cache" / "gemini_web_v24" / "diagnostics"
    _write_json(diagnostic_root / f"{job_id}.summary.json", summary)
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    (diagnostic_root / f"{job_id}.jsonl").write_text(
        json.dumps({"event": "gemini_web_attempt", "outcome": "completed"}) + "\n",
        encoding="utf-8",
    )
    counts = {
        name.lower(): sum(row["Decision"] == name for row in rows)
        for name in ("KEEP", "MAYBE", "REJECT")
    }
    protocol_inputs = {
        "research_question": "How is the requested method evaluated?",
        "research_context": "Frozen context",
        "inclusion_criteria": "Include evaluated primary work.",
        "exclusion_criteria": "Exclude secondary discussion.",
    }
    prisma = {
        "workflow_id": job_id,
        "job_id": job_id,
        "kind": "screening",
        "status": "screening",
        "input_fingerprint": screening_input,
        "protocol_inputs": protocol_inputs,
        "screening_plan": {"records_selected": len(run_ids)},
        "screening_state": {"counts": counts},
    }
    _write_json(artifacts / "prisma" / f"{job_id}.json", prisma)
    spec = {
        "spec_schema_version": "litsync-screening-benchmark-spec-v1",
        "benchmark_id": "domain-neutral-fixture",
        "benchmark_version": "1.0.0",
        "name": "Domain-neutral offline fixture",
        **protocol_inputs,
        "gold_label_file": "gold.csv",
        "source_dataset_fingerprint": dataset_fingerprint,
        "screening_input_fingerprint": screening_input,
        "gold_file_fingerprint": file_fingerprint(gold_path),
        "benchmark_spec_fingerprint": "0" * 64,
        "run_selected_source_row_ids": run_ids,
        "gold_selected_source_row_ids": gold_ids,
        "unresolved_label_policy": "exclude_from_resolved_metrics",
        "metric_definitions": METRIC_DEFINITION_VERSIONS,
        "release_thresholds": {
            "quality": {
                "keep_or_maybe_recall": {"comparator": "ge", "value": 0.95},
                "false_reject_rate": {"comparator": "le", "value": 0.05},
                "definitive_keep_precision": {"comparator": "gt", "value": 0.85},
            },
            "reliability": {
                "structured_terminal_fallbacks": {"comparator": "le", "value": 2},
                "technical_fallbacks": {"comparator": "le", "value": 2},
                "invalid_definitive_decisions": {"comparator": "eq", "value": 0},
                "exact_evidence_rate": {"comparator": "ge", "value": 1.0},
                "structurally_validated_rate": {"comparator": "ge", "value": 1.0},
            },
            "require_cold": True,
        },
        "expected_protocol_identity": {
            "protocol_id": protocol_id,
            "protocol_cache_version": "gemini-web-v2.4-protocol-v3",
        },
        "expected_assessment_identity": {
            "architecture_version": "gemini-web-batched-v2.4",
            "assessment_prompt_version": "gemini-web-v2.4-assessment-prompt-v5",
            "assessment_cache_version": "gemini-web-v2.4-assessment-v5",
        },
        "notes": "Synthetic fixture",
    }
    spec["benchmark_spec_fingerprint"] = _spec_fingerprint(spec)
    spec_path = tmp_path / "benchmark.json"
    _write_json(spec_path, spec)
    return {
        "artifacts": artifacts,
        "dataset": dataset,
        "gold": gold_path,
        "spec": spec_path,
        "job_id": job_id,
        "csv": csv_path,
        "summary": diagnostic_root / f"{job_id}.summary.json",
        "prisma": artifacts / "prisma" / f"{job_id}.json",
    }


def _load_all(fixture):
    spec = load_spec(fixture["spec"])
    gold = load_gold(spec)
    run = load_run(spec, gold, fixture["job_id"], fixture["artifacts"])
    return spec, gold, run


def _rewrite_run(fixture, *, decisions=None, summary_changes=None):
    frame = pd.read_csv(fixture["csv"], dtype=str, keep_default_na=False)
    if decisions:
        for source_id, decision in decisions.items():
            frame.loc[
                frame["Source_Row_Index"].astype(str) == source_id,
                "Decision",
            ] = decision
    if summary_changes and any(
        field in summary_changes
        for field in (
            "resumed_source_row_ids",
            "assessment_cache_hit_source_row_ids",
            "fresh_primary_source_row_ids",
            "directly_handled_without_primary_source_row_ids",
        )
    ):
        origin_fields = {
            "resumed": summary_changes.get("resumed_source_row_ids", []),
            "assessment_cache_hit": summary_changes.get(
                "assessment_cache_hit_source_row_ids", []
            ),
            "fresh_primary": summary_changes.get("fresh_primary_source_row_ids", []),
            "directly_handled_without_primary": summary_changes.get(
                "directly_handled_without_primary_source_row_ids", []
            ),
        }
        reasons = summary_changes.get("direct_handling_reasons", {})
        for origin, source_ids in origin_fields.items():
            mask = frame["Source_Row_Index"].astype(str).isin(source_ids)
            frame.loc[mask, "Execution_Origin"] = origin
            frame.loc[mask, "Direct_Handling_Reason"] = [
                reasons.get(str(source_id), "")
                for source_id in frame.loc[mask, "Source_Row_Index"]
            ]
    frame.to_csv(fixture["csv"], index=False, encoding="utf-8-sig")
    rows = frame.to_dict(orient="records")
    summary = json.loads(fixture["summary"].read_text(encoding="utf-8"))
    summary["screening_output_fingerprint"] = screening_output_fingerprint(rows)
    if summary_changes:
        summary.update(summary_changes)
        if "fresh_primary_source_row_ids" in summary_changes:
            summary["primary_papers_requested"] = len(
                summary_changes["fresh_primary_source_row_ids"]
            )
    _write_json(fixture["summary"], summary)
    prisma = json.loads(fixture["prisma"].read_text(encoding="utf-8"))
    prisma["screening_state"]["counts"] = {
        name.lower(): sum(row["Decision"] == name for row in rows)
        for name in ("KEEP", "MAYBE", "REJECT")
    }
    _write_json(fixture["prisma"], prisma)


def test_spec_and_gold_are_strict_and_validate_spec_is_read_only(tmp_path):
    fixture = _fixture(tmp_path)
    registry = tmp_path / "registry"
    code = benchmark_main([
        "validate-spec", "--spec", str(fixture["spec"]),
        "--registry-dir", str(registry), "--json",
    ])
    assert code == 0
    assert not registry.exists()
    payload = json.loads(fixture["spec"].read_text(encoding="utf-8"))
    payload["notes"] = "mutated"
    _write_json(fixture["spec"], payload)
    with pytest.raises(SpecValidationError, match="fingerprint"):
        load_spec(fixture["spec"])


def test_run_population_and_gold_population_are_separate(tmp_path):
    fixture = _fixture(tmp_path)
    spec, gold, run = _load_all(fixture)
    result = evaluate_run(spec, gold, run)
    assert len(run.rows) == 5
    assert len(gold.rows) == 4
    assert result.metrics["resolved_sample_size"].value == 3
    assert result.metrics["model_reject_count"].value == 2
    assert result.gate.verdict == BenchmarkVerdict.PASS


def test_100_run_rows_and_60_frozen_gold_rows_remain_separate(tmp_path):
    fixture = _fixture(tmp_path, run_count=100, gold_count=60)
    loaded_spec, gold, run = _load_all(fixture)
    result = evaluate_run(loaded_spec, gold, run)
    assert len(run.rows) == 100
    assert len(gold.rows) == 60
    assert set(gold.rows).issubset(run.rows)
    assert len(result.row_outcomes) == 60
    assert result.metrics["manual_review_burden_full_run"].denominator == 100
    assert result.metrics["manual_review_burden_resolved_gold"].denominator == 59


def test_auditable_metrics_include_both_manual_review_populations(tmp_path):
    result = evaluate_run(*_load_all(_fixture(tmp_path)))
    full = result.metrics["manual_review_burden_full_run"]
    resolved = result.metrics["manual_review_burden_resolved_gold"]
    assert (full.numerator, full.denominator, full.value) == (1, 5, 0.2)
    assert (resolved.numerator, resolved.denominator) == (1, 3)
    assert resolved.confidence_interval is not None
    assert all(
        metric.metric_definition_version
        and metric.population_scope
        and metric.numerator is not None
        and metric.denominator is not None
        for metric in result.metrics.values()
    )


@pytest.mark.parametrize(
    ("resumed", "cached", "fresh", "classification", "verdict"),
    [
        ([], ["0"], ["1", "2", "3", "4"], ProvenanceClass.WARM_CACHE, BenchmarkVerdict.PROVISIONAL),
        (["0"], [], ["1", "2", "3", "4"], ProvenanceClass.PARTIALLY_RESUMED, BenchmarkVerdict.FAIL),
        (["0", "1", "2", "3", "4"], [], [], ProvenanceClass.FULLY_RESUMED, BenchmarkVerdict.FAIL),
    ],
)
def test_reuse_provenance_classes_and_gates(
    tmp_path, resumed, cached, fresh, classification, verdict,
):
    fixture = _fixture(tmp_path)
    _rewrite_run(fixture, summary_changes={
        "resumed_count": len(resumed),
        "resumed_source_row_ids": resumed,
        "assessment_cache_hits_loaded": len(cached),
        "assessment_cache_hit_source_row_ids": cached,
        "fresh_primary_papers": len(fresh),
        "fresh_primary_source_row_ids": fresh,
    })
    result = evaluate_run(*_load_all(fixture))
    assert result.provenance.classification == classification
    assert result.gate.verdict == verdict


def test_mixed_or_missing_provenance_is_invalid(tmp_path):
    fixture = _fixture(tmp_path)
    _rewrite_run(fixture, summary_changes={
        "assessment_prompt_version": "old-prompt",
        "source_dataset_fingerprint": "",
    })
    result = evaluate_run(*_load_all(fixture))
    assert result.gate.verdict == BenchmarkVerdict.INVALID
    fixture["prisma"].unlink()
    spec = load_spec(fixture["spec"])
    gold = load_gold(spec)
    with pytest.raises(RunArtifactError, match="missing run artifacts"):
        load_run(spec, gold, fixture["job_id"], fixture["artifacts"])


def test_unsure_is_reported_but_excluded_from_quality_denominators(tmp_path):
    result = evaluate_run(*_load_all(_fixture(tmp_path)))
    assert result.unsure_gold_source_row_ids == ["3"]
    assert result.metrics["resolved_sample_size"].value == 3
    assert sum(sum(row.values()) for row in result.confusion_matrix.values()) == 3


def test_false_reject_precision_and_reliability_regressions_fail(tmp_path):
    false_reject = _fixture(tmp_path / "false-reject")
    _rewrite_run(false_reject, decisions={"0": "REJECT"})
    result = evaluate_run(*_load_all(false_reject))
    assert result.gate.verdict == BenchmarkVerdict.FAIL
    assert "0" in result.false_reject_source_row_ids

    precision = _fixture(tmp_path / "precision")
    _rewrite_run(precision, decisions={"2": "KEEP"})
    precision_result = evaluate_run(*_load_all(precision))
    assert precision_result.gate.verdict == BenchmarkVerdict.FAIL
    assert "2" in precision_result.false_keep_source_row_ids

    reliability = _fixture(tmp_path / "reliability")
    _rewrite_run(reliability, summary_changes={"technical_fallback_count": 3})
    reliability_result = evaluate_run(*_load_all(reliability))
    assert reliability_result.gate.verdict == BenchmarkVerdict.FAIL


def test_valid_fail_locks_registry_but_invalid_does_not(tmp_path):
    fixture = _fixture(tmp_path / "valid")
    _rewrite_run(fixture, decisions={"0": "REJECT"})
    registry = tmp_path / "registry"
    output = tmp_path / "result"
    code = benchmark_main([
        "evaluate", "--spec", str(fixture["spec"]), "--job-id", fixture["job_id"],
        "--artifacts-root", str(fixture["artifacts"]), "--output-dir", str(output),
        "--registry-dir", str(registry), "--json",
    ])
    assert code == 0
    loaded = load_spec(fixture["spec"])
    check_registry(loaded.spec, registry)
    changed = json.loads(fixture["spec"].read_text(encoding="utf-8"))
    changed["notes"] = "new meaning"
    changed["benchmark_spec_fingerprint"] = _spec_fingerprint(changed)
    _write_json(fixture["spec"], changed)
    with pytest.raises(SpecImmutabilityError):
        check_registry(load_spec(fixture["spec"]).spec, registry)

    invalid = _fixture(tmp_path / "invalid")
    _rewrite_run(invalid, summary_changes={"source_dataset_fingerprint": ""})
    invalid_registry = tmp_path / "invalid-registry"
    code = benchmark_main([
        "evaluate", "--spec", str(invalid["spec"]), "--job-id", invalid["job_id"],
        "--artifacts-root", str(invalid["artifacts"]),
        "--output-dir", str(tmp_path / "invalid-result"),
        "--registry-dir", str(invalid_registry), "--json",
    ])
    assert code == 2
    assert not invalid_registry.exists()


def test_comparison_discloses_reuse_and_limits_change_claims(tmp_path):
    baseline_fixture = _fixture(tmp_path / "baseline", job_id="baseline")
    candidate_fixture = _fixture(tmp_path / "candidate", job_id="candidate")
    _rewrite_run(candidate_fixture, decisions={"0": "REJECT"}, summary_changes={
        "resumed_count": 1,
        "resumed_source_row_ids": ["0"],
        "fresh_primary_papers": 4,
        "fresh_primary_source_row_ids": ["1", "2", "3", "4"],
    })
    baseline = evaluate_run(*_load_all(baseline_fixture))
    loaded_spec, loaded_gold, _ = _load_all(baseline_fixture)
    candidate_run = load_run(
        loaded_spec,
        loaded_gold,
        candidate_fixture["job_id"],
        candidate_fixture["artifacts"],
    )
    candidate = evaluate_run(loaded_spec, loaded_gold, candidate_run)
    comparison = compare_results([baseline, candidate])
    assert comparison.newly_introduced_false_rejects[0]["claim_status"] == (
        "OBSERVED_TRANSITION_WITH_REUSE"
    )
    assert comparison.transitions[0]["candidate_reuse_status"] == "RESUMED"


def test_reports_are_self_contained_escaped_and_deterministic_json(tmp_path):
    fixture = _fixture(tmp_path / "fixture")
    frame = pd.read_csv(fixture["csv"], dtype=str, keep_default_na=False)
    frame.loc[0, "Title"] = "<script>alert(1)</script>"
    frame.loc[0, "Evidence_JSON"] = json.dumps([{
        "source": "title", "quote": "<script>alert(1)</script>"
    }])
    frame.loc[0, "Decision"] = "REJECT"
    frame.to_csv(fixture["csv"], index=False, encoding="utf-8-sig")
    _rewrite_run(fixture)
    result = evaluate_run(*_load_all(fixture))
    paths = write_result_report(result, tmp_path / "report")
    report = paths["html"].read_text(encoding="utf-8")
    assert "&lt;script&gt;" in report
    assert "<script>alert(1)</script>" not in report
    parsed = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert parsed["job_id"] == fixture["job_id"]


def test_cli_enforcement_and_comparison_outputs(tmp_path):
    first = _fixture(tmp_path / "first", job_id="first")
    second = _fixture(tmp_path / "second", job_id="second")
    # Both fixture trees use the same spec identity/content except absolute gold path.
    # Use one artifact root containing both jobs and one shared frozen spec.
    shared = tmp_path / "shared"
    shared_artifacts = shared / "outputs"
    shared_spec_fixture = _fixture(shared, job_id="first")
    other = _fixture(tmp_path / "other", job_id="second")
    for name in ("csv", "summary", "prisma"):
        source = other[name]
        if name == "csv":
            target = shared_artifacts / "runs" / "screened-second.csv"
        elif name == "summary":
            target = shared_artifacts / "cache" / "gemini_web_v24" / "diagnostics" / "second.summary.json"
        else:
            target = shared_artifacts / "prisma" / "second.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    source_diag = other["artifacts"] / "cache" / "gemini_web_v24" / "diagnostics" / "second.jsonl"
    target_diag = shared_artifacts / "cache" / "gemini_web_v24" / "diagnostics" / "second.jsonl"
    target_diag.write_bytes(source_diag.read_bytes())
    # Rebind copied job artifacts to the shared spec fingerprints and paths.
    summary = json.loads((shared_artifacts / "cache" / "gemini_web_v24" / "diagnostics" / "second.summary.json").read_text())
    shared_spec = json.loads(shared_spec_fixture["spec"].read_text())
    summary.update({
        "source_dataset_fingerprint": shared_spec["source_dataset_fingerprint"],
        "screening_input_fingerprint": shared_spec["screening_input_fingerprint"],
        "diagnostics_path": str(target_diag),
    })
    _write_json(shared_artifacts / "cache" / "gemini_web_v24" / "diagnostics" / "second.summary.json", summary)
    prisma = json.loads((shared_artifacts / "prisma" / "second.json").read_text())
    prisma["input_fingerprint"] = shared_spec["screening_input_fingerprint"]
    _write_json(shared_artifacts / "prisma" / "second.json", prisma)
    registry = tmp_path / "registry"
    code = benchmark_main([
        "compare", "--spec", str(shared_spec_fixture["spec"]),
        "--job-id", "first", "--job-id", "second",
        "--artifacts-root", str(shared_artifacts),
        "--output-dir", str(tmp_path / "comparison"),
        "--registry-dir", str(registry), "--json",
    ])
    assert code == 0
    assert (tmp_path / "comparison" / "benchmark-comparison.json").is_file()


@pytest.mark.parametrize(
    "unsafe",
    [
        "../escape", r"C:escape", "CON", "CON.txt", "name.", " name",
        "name ", "a/b", r"a\b", "a\u2215b", ".", "..",
    ],
)
@pytest.mark.parametrize("field", ["benchmark_id", "benchmark_version"])
def test_benchmark_identifiers_reject_unsafe_ascii_and_windows_names(
    tmp_path, unsafe, field,
):
    fixture = _fixture(tmp_path)
    payload = json.loads(fixture["spec"].read_text(encoding="utf-8"))
    payload[field] = unsafe
    payload["benchmark_spec_fingerprint"] = _spec_fingerprint(payload)
    _write_json(fixture["spec"], payload)
    with pytest.raises(SpecValidationError):
        load_spec(fixture["spec"])


@pytest.mark.parametrize(
    "unsafe",
    ["../job", r"C:job", "CON", "CON.txt", "job.", " job", "job ", "a/b", r"a\b"],
)
def test_job_identifiers_are_rejected_without_normalization(tmp_path, unsafe):
    fixture = _fixture(tmp_path)
    spec = load_spec(fixture["spec"])
    gold = load_gold(spec)
    with pytest.raises(RunArtifactError, match="invalid screening job ID"):
        load_run(spec, gold, unsafe, fixture["artifacts"])


@pytest.mark.parametrize("gold_path", ["../gold.csv", r"C:\gold.csv"])
def test_gold_path_must_be_relative_and_contained(tmp_path, gold_path):
    fixture = _fixture(tmp_path)
    payload = json.loads(fixture["spec"].read_text(encoding="utf-8"))
    payload["gold_label_file"] = gold_path
    payload["benchmark_spec_fingerprint"] = _spec_fingerprint(payload)
    _write_json(fixture["spec"], payload)
    with pytest.raises(GoldValidationError):
        load_gold(load_spec(fixture["spec"]))


def test_persisted_row_origins_are_authoritative_and_missing_can_overlap(tmp_path):
    fixture = _fixture(tmp_path)
    frame = pd.read_csv(fixture["csv"], dtype=str, keep_default_na=False)
    frame.loc[frame["Source_Row_Index"] == "4", "Abstract"] = ""
    frame.to_csv(fixture["csv"], index=False, encoding="utf-8-sig")
    _rewrite_run(fixture, summary_changes={
        "fresh_primary_papers": 4,
        "fresh_primary_source_row_ids": ["0", "1", "2", "3"],
        "directly_handled_without_primary_count": 1,
        "directly_handled_without_primary_source_row_ids": ["4"],
        "direct_handling_reasons": {"4": "missing_abstract"},
        "missing_abstract_count": 1,
        "missing_abstract_source_row_ids": ["4"],
    })
    _, _, run = _load_all(fixture)
    assert run.provenance.classification == ProvenanceClass.COLD
    assert run.provenance.directly_handled_without_primary_source_row_ids == ["4"]
    assert run.provenance.direct_handling_reasons == {"4": "missing_abstract"}
    assert run.provenance.missing_abstract_source_row_ids == ["4"]

    _rewrite_run(fixture, summary_changes={"missing_abstract_count": 9})
    invalid = evaluate_run(*_load_all(fixture))
    assert invalid.gate.verdict == BenchmarkVerdict.INVALID
    assert any(
        "missing_abstract_count" in reason
        for reason in invalid.provenance.reasons
    )


def test_row_origin_summary_disagreement_and_direct_reason_mismatch_fail_closed(tmp_path):
    fixture = _fixture(tmp_path)
    _rewrite_run(fixture, summary_changes={
        "resumed_count": 1,
        "resumed_source_row_ids": ["0"],
        "fresh_primary_papers": 4,
        "fresh_primary_source_row_ids": ["1", "2", "3", "4"],
    })
    frame = pd.read_csv(fixture["csv"], dtype=str, keep_default_na=False)
    frame.loc[frame["Source_Row_Index"] == "0", "Execution_Origin"] = "fresh_primary"
    frame.to_csv(fixture["csv"], index=False, encoding="utf-8-sig")
    _rewrite_run(fixture)
    result = evaluate_run(*_load_all(fixture))
    assert result.gate.verdict == BenchmarkVerdict.INVALID
    assert any("summary IDs differ" in reason for reason in result.provenance.reasons)

    direct = _fixture(tmp_path / "direct")
    frame = pd.read_csv(direct["csv"], dtype=str, keep_default_na=False)
    frame.loc[0, "Execution_Origin"] = "directly_handled_without_primary"
    frame.loc[0, "Direct_Handling_Reason"] = ""
    frame.to_csv(direct["csv"], index=False, encoding="utf-8-sig")
    _rewrite_run(direct, summary_changes={
        "fresh_primary_papers": 4,
        "fresh_primary_source_row_ids": ["1", "2", "3", "4"],
        "directly_handled_without_primary_count": 1,
        "directly_handled_without_primary_source_row_ids": ["0"],
        "direct_handling_reasons": {},
    })
    direct_result = evaluate_run(*_load_all(direct))
    assert direct_result.gate.verdict == BenchmarkVerdict.INVALID
    assert any("lacks Direct_Handling_Reason" in reason for reason in direct_result.provenance.reasons)


def test_source_row_ids_preserve_leading_zero_strings():
    from litsync_app.benchmarking.provenance import source_row_id

    assert source_row_id("001") == "001"
    assert source_row_id("01.0") == "01.0"


def test_malformed_evidence_member_is_invalid_without_cli_traceback(tmp_path, capsys):
    fixture = _fixture(tmp_path)
    frame = pd.read_csv(fixture["csv"], dtype=str, keep_default_na=False)
    frame.loc[0, "Evidence_JSON"] = '["not-an-object"]'
    frame.to_csv(fixture["csv"], index=False, encoding="utf-8-sig")
    _rewrite_run(fixture)
    code = benchmark_main([
        "evaluate", "--spec", str(fixture["spec"]), "--job-id", fixture["job_id"],
        "--artifacts-root", str(fixture["artifacts"]),
        "--output-dir", str(tmp_path / "invalid-output"),
        "--registry-dir", str(tmp_path / "registry"), "--json",
    ])
    captured = capsys.readouterr()
    assert code == 2
    assert json.loads(captured.out)["verdict"] == "INVALID"
    assert "Traceback" not in captured.err
    assert (tmp_path / "invalid-output" / COMPLETE_MARKER_NAME).is_file()
    assert not (tmp_path / "registry").exists()


def test_malformed_nested_summary_and_prisma_values_fail_closed(tmp_path):
    fixture = _fixture(tmp_path)
    _rewrite_run(fixture, summary_changes={
        "detector_outcomes": ["not", "an", "object"],
        "retry_count": {"not": "numeric"},
    })
    prisma = json.loads(fixture["prisma"].read_text(encoding="utf-8"))
    prisma["screening_state"]["counts"] = ["not", "an", "object"]
    _write_json(fixture["prisma"], prisma)
    result = evaluate_run(*_load_all(fixture))
    assert result.gate.verdict == BenchmarkVerdict.INVALID
    assert any("detector_outcomes" in reason for reason in result.provenance.reasons)
    assert any("retry_count" in reason for reason in result.provenance.reasons)
    assert any("screening counts" in reason for reason in result.provenance.reasons)


def _publish_fixture_result(fixture, output: Path, registry: Path):
    loaded_spec, gold, run = _load_all(fixture)
    result = evaluate_run(loaded_spec, gold, run)
    staged = stage_result_report(result, output)
    paths = publish_completed_evaluation(
        loaded_spec.spec,
        result.gate.verdict,
        staged,
        output,
        registry,
        result_key="result",
        publication_kind="evaluation",
        job_ids=[result.job_id],
    )
    return loaded_spec, result, paths


def test_completion_marker_integrity_idempotence_and_mixed_destination_rejection(tmp_path):
    fixture = _fixture(tmp_path / "fixture")
    registry = tmp_path / "registry"
    output = tmp_path / "result"
    loaded_spec, result, paths = _publish_fixture_result(fixture, output, registry)
    marker, validated = validate_completion_directory(output, expected_kind="evaluation")
    original_created_at = marker["created_at"]
    assert set(marker["artifacts"]) == {
        "benchmark-result.json", "benchmark-report.html", "benchmark-errors.csv",
    }
    assert marker["publication_id"]
    assert validated["result"] == paths["result"]

    staged = stage_result_report(result, output)
    publish_completed_evaluation(
        loaded_spec.spec,
        result.gate.verdict,
        staged,
        output,
        registry,
        result_key="result",
        publication_kind="evaluation",
        job_ids=[result.job_id],
    )
    repeated, _ = validate_completion_directory(output)
    assert repeated["created_at"] == original_created_at
    registry_payload = json.loads(
        (registry / loaded_spec.spec.benchmark_id / "1.0.0.json").read_text()
    )
    assert len(registry_payload["completed_publications"]) == 1

    mixed = tmp_path / "mixed"
    mixed.mkdir()
    (mixed / "older.txt").write_text("old", encoding="utf-8")
    with pytest.raises(PublicationError, match="completion marker"):
        write_result_report(result, mixed)
    assert sorted(path.name for path in mixed.iterdir()) == ["older.txt"]


def test_registry_preserves_history_and_detects_completed_corruption(tmp_path):
    fixture = _fixture(tmp_path / "fixture")
    registry = tmp_path / "registry"
    loaded_spec, result, _ = _publish_fixture_result(
        fixture, tmp_path / "first", registry
    )
    second = result.model_copy(update={"job_id": "second-job"})
    staged = stage_result_report(second, tmp_path / "second")
    publish_completed_evaluation(
        loaded_spec.spec,
        second.gate.verdict,
        staged,
        tmp_path / "second",
        registry,
        result_key="result",
        publication_kind="evaluation",
        job_ids=["second-job"],
    )
    registry_path = registry / loaded_spec.spec.benchmark_id / "1.0.0.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert payload["registry_schema_version"] == REGISTRY_SCHEMA_VERSION
    assert len(payload["completed_publications"]) == 2

    (tmp_path / "first" / "benchmark-result.json").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(SpecImmutabilityError, match="corrupt"):
        check_registry(loaded_spec.spec, registry)


def test_pending_publication_recovers_only_the_same_valid_publication(tmp_path):
    fixture = _fixture(tmp_path / "fixture")
    loaded_spec, gold, run = _load_all(fixture)
    result = evaluate_run(loaded_spec, gold, run)
    output = tmp_path / "orphan"
    write_result_report(result, output)
    marker, _ = validate_completion_directory(output)
    reference = {
        "publication_id": marker["publication_id"],
        "publication_kind": marker["publication_kind"],
        "job_ids": marker["job_ids"],
        "verdict": marker["verdict"],
        "completion_marker_path": str(output / COMPLETE_MARKER_NAME),
        "completion_marker_sha256": marker_hash(output),
    }
    registry = tmp_path / "registry"
    registry_path = registry / loaded_spec.spec.benchmark_id / "1.0.0.json"
    _write_json(registry_path, {
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "benchmark_id": loaded_spec.spec.benchmark_id,
        "benchmark_version": loaded_spec.spec.benchmark_version,
        "benchmark_spec_fingerprint": loaded_spec.spec.benchmark_spec_fingerprint,
        "completed_publications": [],
        "pending_publication": reference,
    })
    with pytest.raises(SpecImmutabilityError, match="incomplete pending"):
        check_registry(loaded_spec.spec, registry)
    staged = stage_result_report(result, output)
    publish_completed_evaluation(
        loaded_spec.spec,
        result.gate.verdict,
        staged,
        output,
        registry,
        result_key="result",
        publication_kind="evaluation",
        job_ids=[result.job_id],
    )
    recovered = json.loads(registry_path.read_text(encoding="utf-8"))
    assert recovered["pending_publication"] is None
    assert recovered["completed_publications"] == [reference]

    foreign_output = tmp_path / "foreign"
    foreign_registry = tmp_path / "foreign-registry"
    foreign_path = foreign_registry / loaded_spec.spec.benchmark_id / "1.0.0.json"
    foreign = dict(reference)
    foreign["publication_id"] = "f" * 64
    _write_json(foreign_path, {
        **{key: recovered[key] for key in (
            "registry_schema_version", "benchmark_id", "benchmark_version",
            "benchmark_spec_fingerprint",
        )},
        "completed_publications": [],
        "pending_publication": foreign,
    })
    staged = stage_result_report(result, foreign_output)
    with pytest.raises(PublicationError, match="different incomplete"):
        publish_completed_evaluation(
            loaded_spec.spec,
            result.gate.verdict,
            staged,
            foreign_output,
            foreign_registry,
            result_key="result",
            publication_kind="evaluation",
            job_ids=[result.job_id],
        )
    assert json.loads(foreign_path.read_text())["pending_publication"] == foreign


@pytest.mark.parametrize(
    ("failure_call", "raise_before"),
    [(1, True), (1, False), (2, False), (3, False)],
)
def test_publication_failure_boundaries_recover_idempotently(
    tmp_path, monkeypatch, failure_call, raise_before,
):
    fixture = _fixture(tmp_path / "fixture")
    loaded_spec, gold, run = _load_all(fixture)
    result = evaluate_run(loaded_spec, gold, run)
    output = tmp_path / f"output-{failure_call}-{raise_before}"
    registry = tmp_path / f"registry-{failure_call}-{raise_before}"
    original_write = registry_module._write_registry
    calls = {"count": 0}

    def injected(path, payload):
        calls["count"] += 1
        if calls["count"] == failure_call and raise_before:
            raise OSError("injected registry failure")
        original_write(path, payload)
        if calls["count"] == failure_call and not raise_before:
            raise OSError("injected registry failure")

    monkeypatch.setattr(registry_module, "_write_registry", injected)
    staged = stage_result_report(result, output)
    with pytest.raises(OSError, match="injected registry failure"):
        publish_completed_evaluation(
            loaded_spec.spec,
            result.gate.verdict,
            staged,
            output,
            registry,
            result_key="result",
            publication_kind="evaluation",
            job_ids=[result.job_id],
        )
    monkeypatch.setattr(registry_module, "_write_registry", original_write)

    staged = stage_result_report(result, output)
    publish_completed_evaluation(
        loaded_spec.spec,
        result.gate.verdict,
        staged,
        output,
        registry,
        result_key="result",
        publication_kind="evaluation",
        job_ids=[result.job_id],
    )
    check_registry(loaded_spec.spec, registry)
    payload = json.loads(
        (
            registry / loaded_spec.spec.benchmark_id
            / f"{loaded_spec.spec.benchmark_version}.json"
        ).read_text()
    )
    assert payload["pending_publication"] is None
    assert len(payload["completed_publications"]) == 1


def test_completion_marker_rejects_traversal_and_legacy_registry_fails_closed(tmp_path):
    fixture = _fixture(tmp_path / "fixture")
    loaded_spec, result, _ = _publish_fixture_result(
        fixture, tmp_path / "result", tmp_path / "registry"
    )
    marker_path = tmp_path / "result" / COMPLETE_MARKER_NAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["artifact_relative_paths"][0] = "../benchmark-result.json"
    _write_json(marker_path, marker)
    with pytest.raises(PublicationError, match="unsafe artifact path"):
        validate_completion_directory(tmp_path / "result")

    legacy = tmp_path / "legacy" / loaded_spec.spec.benchmark_id / "1.0.0.json"
    _write_json(legacy, {
        "registry_schema_version": "litsync-benchmark-registry-v1",
        "benchmark_id": loaded_spec.spec.benchmark_id,
        "benchmark_version": "1.0.0",
        "benchmark_spec_fingerprint": loaded_spec.spec.benchmark_spec_fingerprint,
        "first_recorded_result": "old.json",
    })
    with pytest.raises(SpecImmutabilityError, match="unsupported"):
        check_registry(loaded_spec.spec, tmp_path / "legacy")


def test_csv_injection_protection_covers_every_string_shape(tmp_path):
    result = evaluate_run(*_load_all(_fixture(tmp_path / "fixture")))
    dangerous = [
        "=formula",
        "  +formula",
        "\t-formula",
        "\r\n@formula",
        '  =2+2,"quoted"\nnext',
    ]
    rows = []
    for index, title in enumerate(dangerous):
        rows.append({
            "source_row_id": f"x{index}",
            "gold_label": "KEEP",
            "decision": "REJECT",
            "reuse_status": "FRESH",
            "title": title,
        })
    payload = result.model_dump(mode="python")
    payload["row_outcomes"] = rows
    payload["false_reject_source_row_ids"] = [row["source_row_id"] for row in rows]
    protected_result = type(result).model_validate(payload)
    paths = write_result_report(protected_result, tmp_path / "report")
    with paths["errors"].open(encoding="utf-8-sig", newline="") as handle:
        serialized = list(csv.DictReader(handle))
    assert [row["title"] for row in serialized] == [
        "'" + value for value in dangerous
    ]
    json_payload = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert [row["title"] for row in json_payload["row_outcomes"]] == [
        value.replace("\r\n", "\n").replace("\r", "\n")
        for value in dangerous
    ]
    assert all(
        not row["title"].startswith("'")
        for row in json_payload["row_outcomes"]
    )


def test_invalid_comparison_emits_no_deltas_transitions_or_claims(tmp_path):
    baseline = evaluate_run(*_load_all(_fixture(tmp_path)))
    changed_provenance = baseline.provenance.model_copy(update={
        "source_dataset_fingerprint": "f" * 64,
    })
    candidate = baseline.model_copy(update={
        "job_id": "candidate",
        "provenance": changed_provenance,
    })
    comparison = compare_results([baseline, candidate])
    assert not comparison.valid
    assert comparison.metric_deltas == {}
    assert comparison.movement_matrix == {}
    assert comparison.transitions == []
    assert comparison.pairwise_comparisons == []
    assert comparison.newly_introduced_false_rejects == []
    assert comparison.corrected_false_rejects == []


def _cli_subprocess(*arguments: str):
    return subprocess.run(
        [sys.executable, "-m", "litsync_app.benchmarking.cli", *arguments],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_cli_subprocess_phase2a_exit_codes_and_json_contract(tmp_path):
    root = tmp_path / "space dir"
    passed = _fixture(root / "pass")
    validate = _cli_subprocess(
        "validate-spec", "--spec", str(passed["spec"]),
        "--registry-dir", str(root / "validate-registry"), "--json",
    )
    assert validate.returncode == 0
    assert json.loads(validate.stdout)["status"] == "valid"
    assert validate.stderr == ""

    pass_run = _cli_subprocess(
        "evaluate", "--spec", str(passed["spec"]), "--job-id", passed["job_id"],
        "--artifacts-root", str(passed["artifacts"]),
        "--output-dir", str(root / "po"),
        "--registry-dir", str(root / "pr"),
        "--enforce-gate", "--json",
    )
    assert pass_run.returncode == 0
    assert json.loads(pass_run.stdout)["verdict"] == "PASS"

    second = _fixture(root / "second-source", job_id="second-job")
    shared_artifacts = passed["artifacts"]
    targets = {
        "csv": shared_artifacts / "runs" / "screened-second-job.csv",
        "summary": (
            shared_artifacts / "cache" / "gemini_web_v24" / "diagnostics"
            / "second-job.summary.json"
        ),
        "prisma": shared_artifacts / "prisma" / "second-job.json",
    }
    for name, target in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(second[name].read_bytes())
    source_diagnostics = (
        second["artifacts"] / "cache" / "gemini_web_v24" / "diagnostics"
        / "second-job.jsonl"
    )
    target_diagnostics = (
        shared_artifacts / "cache" / "gemini_web_v24" / "diagnostics"
        / "second-job.jsonl"
    )
    target_diagnostics.write_bytes(source_diagnostics.read_bytes())
    shared_spec = json.loads(passed["spec"].read_text(encoding="utf-8"))
    second_summary = json.loads(targets["summary"].read_text(encoding="utf-8"))
    second_summary.update({
        "source_dataset_fingerprint": shared_spec["source_dataset_fingerprint"],
        "screening_input_fingerprint": shared_spec["screening_input_fingerprint"],
        "diagnostics_path": str(target_diagnostics),
    })
    _write_json(targets["summary"], second_summary)
    second_prisma = json.loads(targets["prisma"].read_text(encoding="utf-8"))
    second_prisma["input_fingerprint"] = shared_spec["screening_input_fingerprint"]
    _write_json(targets["prisma"], second_prisma)
    comparison = _cli_subprocess(
        "compare", "--spec", str(passed["spec"]),
        "--job-id", passed["job_id"], "--job-id", "second-job",
        "--artifacts-root", str(shared_artifacts),
        "--output-dir", str(root / "co"),
        "--registry-dir", str(root / "pr"), "--json",
    )
    assert comparison.returncode == 0
    assert json.loads(comparison.stdout)["valid"] is True

    second_summary["source_dataset_fingerprint"] = "f" * 64
    _write_json(targets["summary"], second_summary)
    invalid_comparison = _cli_subprocess(
        "compare", "--spec", str(passed["spec"]),
        "--job-id", passed["job_id"], "--job-id", "second-job",
        "--artifacts-root", str(shared_artifacts),
        "--output-dir", str(root / "ic"),
        "--registry-dir", str(root / "pr"), "--json",
    )
    assert invalid_comparison.returncode == 2, invalid_comparison.stderr
    assert json.loads(invalid_comparison.stdout)["valid"] is False
    invalid_comparison_payload = json.loads(
        (
            root / "ic" / "benchmark-comparison.json"
        ).read_text(encoding="utf-8")
    )
    assert invalid_comparison_payload["metric_deltas"] == {}
    assert invalid_comparison_payload["transitions"] == []

    failed = _fixture(root / "fail")
    _rewrite_run(failed, decisions={"0": "REJECT"})
    fail_run = _cli_subprocess(
        "evaluate", "--spec", str(failed["spec"]), "--job-id", failed["job_id"],
        "--artifacts-root", str(failed["artifacts"]),
        "--output-dir", str(root / "fail-output"),
        "--registry-dir", str(root / "fail-registry"),
        "--enforce-gate", "--json",
    )
    assert fail_run.returncode == 3
    assert json.loads(fail_run.stdout)["verdict"] == "FAIL"

    provisional = _fixture(root / "provisional")
    _rewrite_run(provisional, summary_changes={
        "assessment_cache_hits_loaded": 1,
        "assessment_cache_hit_source_row_ids": ["0"],
        "fresh_primary_papers": 4,
        "fresh_primary_source_row_ids": ["1", "2", "3", "4"],
    })
    provisional_run = _cli_subprocess(
        "evaluate", "--spec", str(provisional["spec"]),
        "--job-id", provisional["job_id"],
        "--artifacts-root", str(provisional["artifacts"]),
        "--output-dir", str(root / "provisional-output"),
        "--registry-dir", str(root / "provisional-registry"),
        "--enforce-gate", "--json",
    )
    assert provisional_run.returncode == 3
    assert json.loads(provisional_run.stdout)["verdict"] == "PROVISIONAL"

    invalid = _fixture(root / "invalid")
    frame = pd.read_csv(invalid["csv"], dtype=str, keep_default_na=False)
    frame.loc[0, "Evidence_JSON"] = '["malformed"]'
    frame.to_csv(invalid["csv"], index=False, encoding="utf-8-sig")
    _rewrite_run(invalid)
    invalid_run = _cli_subprocess(
        "evaluate", "--spec", str(invalid["spec"]), "--job-id", invalid["job_id"],
        "--artifacts-root", str(invalid["artifacts"]),
        "--output-dir", str(root / "invalid-output"),
        "--registry-dir", str(root / "invalid-registry"),
        "--json",
    )
    assert invalid_run.returncode == 2
    assert json.loads(invalid_run.stdout)["verdict"] == "INVALID"
    assert "Traceback" not in invalid_run.stderr
