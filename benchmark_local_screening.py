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
    model_roles = {
        "triage_or_final": sorted(set(filter(None, models))),
        "protocol": sorted(set(str(row.get("Protocol_Model") or "") for row in rows) - {""}),
        "deep": sorted(set(str(row.get("Deep_Model") or "") for row in rows) - {""}),
        "edge": sorted(set(str(row.get("Edge_Model") or "") for row in rows) - {""}),
    }
    batch_metrics = []
    validation_failures = []
    evidence_total = 0
    evidence_exact = 0
    critic_reversals = []
    critic_failures = 0
    for row in rows:
        try:
            validation_failures.extend(json.loads(str(row.get("Validation_Errors") or "[]")))
        except (json.JSONDecodeError, TypeError):
            validation_failures.append("unparseable Validation_Errors")
        try:
            batch_metrics.extend(json.loads(str(row.get("Layer_Metrics_JSON") or "[]")))
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            trace = json.loads(str(row.get("Layer_Trace_JSON") or "[]"))
        except (json.JSONDecodeError, TypeError):
            trace = []
        if len(trace) >= 2 and trace[-1].get("name") == "edge_critic":
            before, after = trace[-2].get("decision"), trace[-1].get("decision")
            if before != after:
                critic_reversals.append({
                    "Source_Row_Index": row.get("Source_Row_Index"),
                    "Title": row.get("Title"), "before": before, "after": after,
                })
        critic_failures += sum(
            item.get("name") == "edge_critic"
            and item.get("validation_status") not in {"validated", ""}
            for item in trace
        )
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
    unique_metrics = {}
    for index, item in enumerate(batch_metrics):
        batch_id = item.get("batch_id")
        key = (
            str(item.get("layer") or "unknown"), str(batch_id), int(item.get("retry") or 0)
        ) if batch_id else ("legacy", index)
        unique_metrics.setdefault(key, item)
    layer_runtime = {}
    for item in unique_metrics.values():
        layer = str(item.get("layer") or "unknown")
        layer_runtime[layer] = round(
            layer_runtime.get(layer, 0.0) + float(item.get("wall_seconds") or 0.0), 4
        )
    batch_errors = [
        {"layer": item.get("layer"), "batch_id": item.get("batch_id"),
         "retry": item.get("retry"), "error": str(item.get("error") or "")}
        for item in unique_metrics.values() if str(item.get("error") or "").strip()
    ]
    failure_categories = {
        "evidence": 0, "criterion_contract": 0, "schema_or_parse": 0, "other": 0,
    }
    for failure in validation_failures:
        lowered = failure.casefold()
        if any(token in lowered for token in ("json", "schema", "validation error", "batch ids")):
            failure_categories["schema_or_parse"] += 1
        elif "evidence" in lowered:
            failure_categories["evidence"] += 1
        elif "criterion" in lowered:
            failure_categories["criterion_contract"] += 1
        else:
            failure_categories["other"] += 1
    return {
        "total": len(rows), **counts,
        "parse_error": failure_categories["schema_or_parse"],
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
        "models_by_role": model_roles,
        "local_profiles": sorted(set(str(row.get("Local_Profile") or "") for row in rows) - {""}),
        "rq_frame_ids": sorted(set(str(row.get("RQ_Frame_ID") or "") for row in rows) - {""}),
        "rq_frame_versions": sorted(set(str(row.get("RQ_Frame_Version") or "") for row in rows) - {""}),
        "rq_frame_statuses": sorted(set(str(row.get("RQ_Frame_Status") or "") for row in rows) - {""}),
        "validation_failure_count": len(validation_failures),
        "validation_failures": validation_failures,
        "raw_model_contract_violation_count": len(validation_failures),
        "validation_failure_categories": failure_categories,
        "final_unresolved_count": validation.count("unresolved"),
        "critic_failure_count": critic_failures,
        "per_layer_runtime_seconds": layer_runtime,
        "per_layer_runtime_note": "Deduplicated by layer, batch_id, and retry before summing wall time.",
        "critic_reversal_count": len(critic_reversals),
        "critic_reversals": critic_reversals,
        "batch_calls_recorded": len(unique_metrics),
        "batch_retry_calls": sum(int(item.get("retry") or 0) > 0 for item in unique_metrics.values()),
        "batch_failure_count": len(batch_errors),
        "batch_failures": batch_errors,
        "decision_ratio_note": "Decision counts are descriptive only and are never release gates.",
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
    suspicious_false_keeps = [item for item in disagreements if item["candidate"] == "KEEP" and item["baseline"] != "KEEP"]
    suspicious_false_rejects = [item for item in disagreements if item["candidate"] == "REJECT" and item["baseline"] != "REJECT"]
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
        "suspicious_false_keep_candidates": suspicious_false_keeps,
        "suspicious_false_reject_candidates": suspicious_false_rejects,
        "suspicious_queue_note": "Unconfirmed audit candidates only; baseline output is not human gold.",
        "critic_reversals": critic_reversals,
        "structurally_valid": sum(str(row.get("Validation_Status")) == "validated" for row, _ in matched),
        "exact_evidence_rate": round(evidence_exact / evidence_total, 4) if evidence_total else 1.0,
        "note": "Comparison baseline is not gold truth.",
    }


def save_blinded_disagreements(
    candidate_rows, baseline_rows, research_question: str, output_path: str,
    manifest_root: str | Path | None = None,
) -> dict[str, object]:
    def identity(row):
        source = str(row.get("Source_Row_Index", "")).strip()
        return ("source", source) if source else ("title", str(row.get("Title", "")).strip().casefold())

    baseline = {identity(row): row for row in baseline_rows}
    disagreements = [
        (row, baseline[identity(row)]) for row in candidate_rows
        if identity(row) in baseline
        and str(row.get("Decision")) != str(baseline[identity(row)].get("Decision"))
    ]
    def excel_safe(value):
        text = str(value or "")
        return "'" + text if text.startswith(("=", "+", "-", "@")) else text

    blinded = [{
        "Source_Row_Index": row.get("Source_Row_Index", ""),
        "Research_Question": excel_safe(research_question),
        "Title": excel_safe(row.get("Title", "")),
        "Abstract": excel_safe(row.get("Abstract", "")),
        "Gold_Decision": "", "Reviewer_Notes": "",
    } for row, _ in disagreements]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(blinded, columns=[
        "Source_Row_Index", "Research_Question", "Title", "Abstract",
        "Gold_Decision", "Reviewer_Notes",
    ]).to_csv(destination, index=False, encoding="utf-8-sig")
    manifest_dir = Path(manifest_root or "benchmark/private/screening_audits")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{destination.stem}.manifest.json"
    manifest_path.write_text(json.dumps({
        "note": "Private mapping; the labeling CSV is prediction-blind.",
        "research_question": research_question,
        "rows": [{
            "Source_Row_Index": row.get("Source_Row_Index", ""),
            "candidate_decision": row.get("Decision", ""),
            "baseline_decision": old.get("Decision", ""),
        } for row, old in disagreements],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "row_count": len(disagreements), "label_path": str(destination),
        "private_manifest_path": str(manifest_path),
    }


def run_benchmark(
    config_path, model_mode=None, limit_override=None, full=False, save_rows=None,
    baseline=None, local_profile="baseline-v3.12", rq_structure=None,
    save_blinded_disagreements_path=None,
):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = save_rows or config.get("output_path", "outputs/benchmark_local_ai.csv")
    limit = limit_override if limit_override is not None else config.get("first_n") or config.get("row_limit")
    rq_structure_json = None
    if rq_structure:
        rq_payload = json.loads(Path(rq_structure).read_text(encoding="utf-8-sig"))
        rq_structure_json = rq_payload.get("concepts", rq_payload) if isinstance(rq_payload, dict) else rq_payload
        if not isinstance(rq_structure_json, dict):
            raise ValueError("--rq-structure must contain a concepts object or a full /generate response")
    summary = screen_csv(
        csv_path=config["dataset_path"],
        research_question=config["research_question"],
        output_path=output,
        max_rows=None if full else limit,
        model_tier=model_mode if model_mode in {"compact", "balanced", "performance"} else None,
        inclusion_criteria=config.get("inclusion_criteria", ""),
        exclusion_criteria=config.get("exclusion_criteria", ""),
        research_context=config.get("research_context", ""),
        resume=False,
        local_profile=local_profile,
        rq_structure_json=rq_structure_json,
    )
    rows = pd.read_csv(output).to_dict(orient="records")
    smoke = summarize_screening_rows(rows, expected_ranges=config.get("expected_ranges"))
    runtime = float(summary.get("runtime_seconds") or 0.0)
    smoke["runtime_seconds"] = runtime
    smoke["papers_per_minute"] = round(len(rows) / runtime * 60, 3) if runtime else None
    smoke["thousand_paper_30_minute_target_met"] = (
        runtime <= 1800 and len(rows) >= 1000
    ) if len(rows) >= 1000 else None
    smoke["prisma_counts_match_outputs"] = bool(summary.get("prisma_counts_match_outputs")) and (
        len(rows) == smoke["keep"] + smoke["maybe"] + smoke["reject"]
        and all(int(summary.get(key) or 0) == smoke[key] for key in ("keep", "maybe", "reject"))
    )
    if config.get("gold_path"):
        smoke["gold_metrics"] = evaluate_rows(rows, pd.read_csv(config["gold_path"]).to_dict(orient="records"))
    if baseline:
        baseline_rows = pd.read_csv(baseline).to_dict(orient="records")
        smoke["baseline_comparison"] = compare_screening_runs(
            rows, baseline_rows
        )
        if save_blinded_disagreements_path:
            smoke["blinded_disagreement_export"] = save_blinded_disagreements(
                rows, baseline_rows, config["research_question"], save_blinded_disagreements_path
            )
    elif save_blinded_disagreements_path:
        raise ValueError("--save-blinded-disagreements requires --baseline")
    report = {**summary, "benchmark": smoke}
    Path(output).with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--tier", choices=["compact", "balanced", "performance"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--save-rows")
    parser.add_argument("--baseline")
    parser.add_argument("--rq-structure")
    parser.add_argument("--save-blinded-disagreements")
    parser.add_argument("--profile", default="baseline-v3.12", choices=[
        "baseline-v3.12", "structured-current", "structured-qwen35-4b",
        "structured-qwen3-8b", "structured-gpt-oss-protocol",
        "structured-grounded-v4.1", "structured-grounded-qwen3-8b-v4.1",
        "structured-grounded-qwen35-4b-v4.1",
    ])
    args = parser.parse_args()
    print(json.dumps(run_benchmark(
        args.config, args.tier, args.limit, args.full, args.save_rows, args.baseline,
        args.profile, args.rq_structure, args.save_blinded_disagreements,
    ), indent=2))
