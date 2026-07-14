from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bulk_screen import screen_csv
from evaluation.local_ai_benchmark import evaluate_rows


def summarize_screening_rows(rows, focused_dataset=True, expected_ranges=None):
    decisions = [str(row.get("Decision", "")).upper() for row in rows]
    validation = [str(row.get("Validation_Status", "")) for row in rows]
    counts = {name: decisions.count(name.upper()) for name in ("keep", "maybe", "reject")}
    escalated = sum(str(row.get("Escalated", "")).lower() in {"1", "true"} for row in rows)
    uncached_times = [
        float(row.get("Processing_Seconds") or 0.0)
        for row in rows
        if str(row.get("Cache_Hit", "")).lower() not in {"1", "true"}
    ]
    models = [str(row.get("Model", "")) for row in rows]
    batch_metrics = []
    evidence_total = 0
    evidence_exact = 0
    for row in rows:
        try:
            batch_metrics.extend(json.loads(str(row.get("Layer_Metrics_JSON") or "[]")))
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            spans = json.loads(str(row.get("Evidence_JSON") or "[]"))
        except (json.JSONDecodeError, TypeError):
            spans = []
        for span in spans:
            evidence_total += 1
            source_text = str(row.get("Title" if span.get("source") == "title" else "Abstract", ""))
            evidence_exact += bool(span.get("quote")) and str(span["quote"]) in source_text
    warnings = []
    if decisions and len(set(decisions)) == 1:
        warnings.append(f"suspicious_all_{decisions[0].lower()}")
    if any(status not in {"validated", "unresolved"} for status in validation):
        warnings.append("unknown_validation_status")
    range_checks = {}
    for name, count in counts.items():
        bounds = (expected_ranges or {}).get(name)
        if bounds:
            range_checks[name] = {"actual": count, "expected": list(bounds), "passed": bounds[0] <= count <= bounds[1]}
    return {
        "total": len(rows), **counts,
        "parse_error": 0,
        "warnings": warnings,
        "validated": validation.count("validated"),
        "unresolved": validation.count("unresolved"),
        "structural_validity_rate": round(validation.count("validated") / len(rows), 4) if rows else 0.0,
        "exact_evidence_rate": round(evidence_exact / evidence_total, 4) if evidence_total else 1.0,
        "malformed_definitive_count": sum(
            decision in {"KEEP", "REJECT"} and status != "validated"
            for decision, status in zip(decisions, validation)
        ),
        "escalated": escalated,
        "repair_call_rate": round(escalated / len(rows), 4) if rows else 0.0,
        "median_uncached_processing_seconds": round(float(pd.Series(uncached_times).median()), 4) if uncached_times else 0.0,
        "automatic_14b_call_count": sum("14b" in model.lower() for model in models),
        "automatic_8b_call_count": sum("8b" in model.lower() for model in models),
        "models_used": sorted(set(filter(None, models))),
        "batch_calls_recorded": len({
            (item.get("layer"), item.get("batch_id"))
            for item in batch_metrics
        }),
        "batch_retry_calls": len({
            (item.get("layer"), item.get("batch_id"))
            for item in batch_metrics if int(item.get("retry") or 0) > 0
        }),
        "expected_range_checks": range_checks,
        "expected_ranges_passed": all(item["passed"] for item in range_checks.values()) if range_checks else None,
    }


def compare_screening_runs(candidate_rows, baseline_rows):
    """Decision/structure comparison only; neither run is treated as human truth."""
    def identity(row):
        source = str(row.get("Source_Row_Index", "")).strip()
        return ("source", source) if source else ("title", str(row.get("Title", "")).strip().casefold())

    baseline = {identity(row): row for row in baseline_rows}
    matched = [(row, baseline[identity(row)]) for row in candidate_rows if identity(row) in baseline]
    disagreements = [
        {"Source_Row_Index": row.get("Source_Row_Index"), "Title": row.get("Title"),
         "candidate": row.get("Decision"), "baseline": old.get("Decision")}
        for row, old in matched if str(row.get("Decision")) != str(old.get("Decision"))
    ]
    critic_reversals = []
    evidence_total = 0
    evidence_exact = 0
    for row, _ in matched:
        try:
            trace = json.loads(str(row.get("Layer_Trace_JSON") or "[]"))
        except (json.JSONDecodeError, TypeError):
            trace = []
        if len(trace) >= 2 and trace[-1].get("name") == "edge_critic":
            before, after = trace[-2].get("decision"), trace[-1].get("decision")
            if before != after:
                critic_reversals.append({
                    "Source_Row_Index": row.get("Source_Row_Index"), "Title": row.get("Title"),
                    "before": before, "after": after,
                })
        try:
            evidence = json.loads(str(row.get("Evidence_JSON") or "[]"))
        except (json.JSONDecodeError, TypeError):
            evidence = []
        for span in evidence:
            evidence_total += 1
            source_text = str(row.get("Title" if span.get("source") == "title" else "Abstract", ""))
            evidence_exact += bool(span.get("quote")) and str(span["quote"]) in source_text
    return {
        "matched": len(matched),
        "decision_disagreements": len(disagreements),
        "decision_disagreement_rate": round(len(disagreements) / len(matched), 4) if matched else None,
        "disagreement_records": disagreements,
        "critic_reversals": critic_reversals,
        "structurally_valid": sum(str(row.get("Validation_Status")) == "validated" for row, _ in matched),
        "exact_evidence_rate": round(evidence_exact / evidence_total, 4) if evidence_total else 1.0,
        "note": "Comparison baseline is not gold truth.",
    }


def run_benchmark(config_path, model_mode=None, limit_override=None, full=False, save_rows=None, baseline=None):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = save_rows or config.get("output_path", "outputs/benchmark_local_ai.csv")
    limit = limit_override if limit_override is not None else config.get("first_n") or config.get("row_limit")
    summary = screen_csv(
        csv_path=config["dataset_path"],
        research_question=config["research_question"],
        output_path=output,
        max_rows=None if full else limit,
        model_tier=model_mode if model_mode in {"compact", "balanced", "performance"} else None,
        inclusion_criteria=config.get("inclusion_criteria", ""),
        exclusion_criteria=config.get("exclusion_criteria", ""),
        resume=False,
    )
    rows = pd.read_csv(output).to_dict(orient="records")
    smoke = summarize_screening_rows(rows, expected_ranges=config.get("expected_ranges"))
    runtime = float(summary.get("runtime_seconds") or 0.0)
    smoke["runtime_seconds"] = runtime
    smoke["papers_per_minute"] = round(len(rows) / runtime * 60, 3) if runtime else None
    smoke["thousand_paper_30_minute_target_met"] = (
        runtime <= 1800 and len(rows) >= 1000
    ) if len(rows) >= 1000 else None
    if config.get("gold_path"):
        smoke["gold_metrics"] = evaluate_rows(rows, pd.read_csv(config["gold_path"]).to_dict(orient="records"))
    if baseline:
        smoke["baseline_comparison"] = compare_screening_runs(
            rows, pd.read_csv(baseline).to_dict(orient="records")
        )
    return {**summary, "benchmark": smoke}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--tier", choices=["compact", "balanced", "performance"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--save-rows")
    parser.add_argument("--baseline")
    args = parser.parse_args()
    print(json.dumps(run_benchmark(
        args.config, args.tier, args.limit, args.full, args.save_rows, args.baseline
    ), indent=2))
