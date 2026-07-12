import argparse
import json
import os
from pathlib import Path

import pandas as pd

from bulk_screen import screen_csv


def _value(row, name, default=""):
    value = row.get(name)
    if value in (None, ""):
        value = row.get("stage1_" + name, default)
    return value


def _as_bool(value):
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def summarize_screening_rows(rows, focused_dataset=True, expected_ranges=None):
    decisions = [str(row.get("Decision", "")).upper() for row in rows]
    keep = decisions.count("KEEP")
    maybe = decisions.count("MAYBE")
    reject = decisions.count("REJECT")
    parse_error = decisions.count("PARSE_ERROR")
    total = len(decisions)

    warnings = []
    if total and reject == total:
        warnings.append("suspicious all-reject warning")
    if total and keep == total:
        warnings.append("suspicious all-keep warning")
    if total and keep == 0 and maybe == 0:
        warnings.append("all-zero collapse warning")
    if total and focused_dataset and reject / total > 0.20:
        warnings.append("suspicious_high_reject_rate_for_focused_dataset")

    suspicious_false_rejects = []
    suspicious_maybes = []
    keeps_missing_dimensions = []
    suspicious_keeps = []
    relation_conflict_keeps = []
    missing_relation_keeps = []
    false_keep_risk_rows = []
    stage2_demotions = []
    strong_intent_rows = []
    medium_intent_rows = []
    subject_direction_rows = []
    workflow_direction_rows = []
    rejected_with_intent = []
    kept_subject_without_workflow = []
    coverage_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for row in rows:
        method = _as_float(row.get("method_alignment_score") or row.get("stage1_method_alignment_score"))
        task = _as_float(row.get("task_alignment_score") or row.get("stage1_task_alignment_score"))
        context = _as_float(row.get("context_alignment_score") or row.get("stage1_context_alignment_score"))
        combined = method + task + context
        if row.get("Decision") == "REJECT" and combined >= 2.0:
            suspicious_false_rejects.append(row.get("Title", ""))
        if row.get("Decision") == "MAYBE" and combined >= 2.2:
            suspicious_maybes.append(row.get("Title", ""))
        coverage = int(_as_float(_value(row, "evidence_coverage_count")))
        coverage_counts[min(3, max(0, coverage))] += 1
        if row.get("Decision") == "REJECT" and (
            coverage >= 3 or _as_bool(_value(row, "suspicious_reject"))
        ):
            suspicious_false_rejects.append(row.get("Title", ""))
        if row.get("Decision") == "MAYBE" and coverage >= 3:
            suspicious_maybes.append(row.get("Title", ""))
        if row.get("Decision") == "KEEP" and _value(row, "required_dimensions_missing"):
            keeps_missing_dimensions.append(row.get("Title", ""))
        if row.get("Decision") == "KEEP" and _as_bool(_value(row, "suspicious_keep")):
            suspicious_keeps.append(row.get("Title", ""))
        if row.get("Decision") == "KEEP" and _as_bool(_value(row, "relation_conflict")):
            relation_conflict_keeps.append(row.get("Title", ""))
        if row.get("Decision") == "KEEP" and not _as_bool(_value(row, "relation_match")):
            missing_relation_keeps.append(row.get("Title", ""))
        if _as_float(_value(row, "false_keep_risk_score")) >= 0.60:
            false_keep_risk_rows.append(row.get("Title", ""))
        if _as_bool(row.get("stage2_demoted_to_reject")):
            stage2_demotions.append(row.get("Title", ""))
        strong_intent = _as_bool(_value(row, "strong_implied_workflow_intent"))
        medium_intent = _as_bool(_value(row, "medium_implied_workflow_intent"))
        subject_direction = _as_bool(_value(row, "subject_review_direction_detected"))
        workflow_direction = _as_bool(_value(row, "workflow_direction_detected"))
        if strong_intent:
            strong_intent_rows.append(row.get("Title", ""))
        if medium_intent:
            medium_intent_rows.append(row.get("Title", ""))
        if subject_direction:
            subject_direction_rows.append(row.get("Title", ""))
        if workflow_direction:
            workflow_direction_rows.append(row.get("Title", ""))
        if row.get("Decision") == "REJECT" and (strong_intent or medium_intent):
            rejected_with_intent.append(row.get("Title", ""))
        if row.get("Decision") == "KEEP" and subject_direction and not workflow_direction:
            kept_subject_without_workflow.append(row.get("Title", ""))

    first = rows[0] if rows else {}
    rq_suspect = _as_bool(_value(first, "rq_extraction_suspect"))
    corpus_empty = not any(
        _value(first, field)
        for field in ("corpus_method_terms", "corpus_task_terms", "corpus_context_terms")
    )
    if suspicious_false_rejects:
        warnings.append("rejects_with_all_required_dimensions_met")
    if suspicious_maybes:
        warnings.append("maybes_with_all_required_dimensions_met")
    if keeps_missing_dimensions:
        warnings.append("keeps_with_missing_required_dimensions")
    if suspicious_keeps:
        warnings.append("suspicious_keeps")
    if relation_conflict_keeps:
        warnings.append("relation_conflicts_among_keeps")
    if missing_relation_keeps:
        warnings.append("missing_relation_but_keep")
    if rq_suspect:
        warnings.append("rq_extraction_suspect")
    if corpus_empty:
        warnings.append("corpus_profile_empty")

    range_checks = {}
    for name, count in (("keep", keep), ("maybe", maybe), ("reject", reject)):
        bounds = (expected_ranges or {}).get(name)
        if bounds:
            range_checks[name] = {
                "actual": count,
                "expected": list(bounds),
                "passed": bounds[0] <= count <= bounds[1],
            }

    stage1_seconds = sum(_as_float(row.get("stage1_processing_seconds")) for row in rows)
    stage2_rows = [row for row in rows if _as_bool(row.get("stage2_rescreened"))]
    stage2_selected = [
        row for row in rows if _as_bool(row.get("stage2_escalation_selected"))
    ]
    stage2_skipped = [
        row for row in rows
        if str(row.get("stage1_decision", "")).upper() == "MAYBE"
        and not _as_bool(row.get("stage2_escalation_selected"))
    ]
    stage2_seconds = sum(_as_float(row.get("stage2_processing_seconds")) for row in stage2_rows)
    rq_contract_created_count = int(_as_float(first.get("rq_contract_created_count")))
    corpus_profile_created_count = int(_as_float(first.get("corpus_profile_created_count")))
    rq_contract_reuse_rate = _as_float(first.get("rq_contract_reuse_rate"))
    if not rq_contract_reuse_rate and stage2_rows:
        rq_contract_reuse_rate = round(
            sum(_as_bool(row.get("stage2_rq_contract_reused")) for row in stage2_rows)
            / len(stage2_rows),
            4,
        )
    model_promotions = [
        row for row in rows
        if _as_bool(_value(row, "model_promoted_from_reject"))
        or _as_bool(_value(row, "model_promoted_from_maybe"))
    ]
    model_demotions = [
        row for row in rows if _as_bool(_value(row, "model_demoted_from_keep"))
    ]
    model_disagreements = [
        row for row in rows
        if str(_value(row, "model_fusion_action")) not in {"", "disabled", "preserve"}
    ]
    adjudicator_rescues = [
        row for row in rows
        if str(row.get("final_adjudication_action") or "").startswith("recall_guard_")
        or str(row.get("final_adjudication_action") or "") in {
            "rescue_reject_to_maybe",
            "promote_maybe_to_keep",
        }
    ]
    model_positive_scores = [_as_float(_value(row, "model_positive_score")) for row in rows]
    model_negative_scores = [_as_float(_value(row, "model_negative_score")) for row in rows]
    model_scores_present = any(score > 0 for score in model_positive_scores + model_negative_scores)
    if rows and _as_bool(_value(rows[0], "model_judges_enabled")) and not model_scores_present:
        warnings.append("model_score_all_zero_collapse")

    return {
        "total": total,
        "keep": keep,
        "maybe": maybe,
        "reject": reject,
        "parse_error": parse_error,
        "warnings": warnings,
        "top_suspicious_false_rejects": suspicious_false_rejects[:10],
        "top_suspicious_maybes": suspicious_maybes[:10],
        "top_keeps_with_missing_required_dimensions": keeps_missing_dimensions[:10],
        "rq_extraction_suspect": rq_suspect,
        "corpus_profile_empty": corpus_empty,
        "method_context_task_coverage_summary": coverage_counts,
        "expected_range_checks": range_checks,
        "expected_ranges_passed": all(
            check["passed"] for check in range_checks.values()
        ) if range_checks else None,
        "top_suspicious_keeps": suspicious_keeps[:10],
        "top_relation_conflict_keeps": relation_conflict_keeps[:10],
        "top_missing_relation_keeps": missing_relation_keeps[:10],
        "top_false_keep_risk_rows": false_keep_risk_rows[:10],
        "stage2_demotion_summary": {
            "count": len(stage2_demotions),
            "top_rows": stage2_demotions[:10],
        },
        "performance_summary": {
            "stage1_seconds": round(stage1_seconds, 3),
            "stage2_seconds": round(stage2_seconds, 3),
            "total_screening_seconds": round(stage1_seconds + stage2_seconds, 3),
            "stage1_average_seconds": round(stage1_seconds / total, 3) if total else 0.0,
            "stage2_average_seconds": (
                round(stage2_seconds / len(stage2_rows), 3) if stage2_rows else 0.0
            ),
            "stage2_rescreen_rate": (
                round(len(stage2_rows) / total, 4) if total else 0.0
            ),
            "stage2_selected_count": len(stage2_selected),
            "stage2_stable_maybe_skipped_count": len(stage2_skipped),
            "rq_contract_reuse_rate": (
                rq_contract_reuse_rate
            ),
            "rq_contract_created_count": rq_contract_created_count,
            "corpus_profile_created_count": corpus_profile_created_count,
        },
        "workflow_intent_summary": {
            "strong": len(strong_intent_rows),
            "medium": len(medium_intent_rows),
            "subject_direction": len(subject_direction_rows),
            "workflow_direction": len(workflow_direction_rows),
        },
        "top_rejects_with_workflow_intent": rejected_with_intent[:10],
        "top_keeps_with_subject_direction_only": kept_subject_without_workflow[:10],
        "model_assisted_summary": {
            "enabled": _as_bool(_value(first, "model_judges_enabled")),
            "mode": _value(first, "model_judge_mode"),
            "models_used": _value(first, "model_judge_models_used"),
            "promotion_count": len(model_promotions),
            "demotion_count": len(model_demotions),
            "disagreement_count": len(model_disagreements),
            "top_model_rescued_papers": [row.get("Title", "") for row in model_promotions[:10]],
            "top_model_demoted_papers": [row.get("Title", "") for row in model_demotions[:10]],
            "final_adjudicator_rescue_count": len(adjudicator_rescues),
            "top_final_adjudicator_rescues": [row.get("Title", "") for row in adjudicator_rescues[:10]],
            "positive_score_min_max": [
                round(min(model_positive_scores), 4) if model_positive_scores else 0.0,
                round(max(model_positive_scores), 4) if model_positive_scores else 0.0,
            ],
            "negative_score_min_max": [
                round(min(model_negative_scores), 4) if model_negative_scores else 0.0,
                round(max(model_negative_scores), 4) if model_negative_scores else 0.0,
            ],
        },
    }


def _as_float(value):
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _title_column(df):
    for candidate in ("Title", "title", "Document Title", "document_title"):
        if candidate in df.columns:
            return candidate
    return df.columns[0]


def _row_limit(config):
    return config.get("first_n") or config.get("row_limit")


def print_benchmark_startup(config_path, config):
    dataset_path = Path(config["dataset_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Benchmark dataset does not exist: {dataset_path}")

    preview = pd.read_csv(dataset_path, nrows=3)
    title_col = _title_column(preview)
    titles = preview[title_col].fillna("").astype(str).tolist()
    print("========== BENCHMARK STARTUP ==========")
    print(f"config_file: {Path(config_path).name}")
    print(f"dataset_path: {dataset_path}")
    print(f"research_question: {config['research_question']}")
    limit = _row_limit(config)
    print(f"first_n: {limit if limit is not None else 'FULL'}")
    print("first_3_titles:")
    for i, title in enumerate(titles, start=1):
        print(f"{i}. {title}")
    print("=======================================")


def run_benchmark(config_path, model_mode=None, limit_override=None, full=False, save_rows=None):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if full:
        config = dict(config)
        config.pop("first_n", None)
        config.pop("row_limit", None)
    elif limit_override is not None:
        config = dict(config)
        config["first_n"] = limit_override
        config["row_limit"] = limit_override
    output_path = config.get("output_path", "outputs/benchmark_screened.csv")
    print_benchmark_startup(config_path, config)
    previous_mode = os.environ.get("MODEL_JUDGE_MODE")
    if model_mode:
        os.environ["MODEL_JUDGE_MODE"] = model_mode
    try:
        screen_csv(
            csv_path=config["dataset_path"],
            research_question=config["research_question"],
            max_rows=_row_limit(config),
            output_path=output_path,
        )
    finally:
        if model_mode:
            if previous_mode is None:
                os.environ.pop("MODEL_JUDGE_MODE", None)
            else:
                os.environ["MODEL_JUDGE_MODE"] = previous_mode
    rows = pd.read_csv(output_path).to_dict("records")
    if save_rows:
        _save_benchmark_rows(output_path, save_rows)
    expected = {
        name: config.get(f"expected_{name}_range")
        for name in ("keep", "maybe", "reject")
        if config.get(f"expected_{name}_range")
    }
    return summarize_screening_rows(rows, expected_ranges=expected)


def _save_benchmark_rows(source_csv, save_path):
    df = pd.read_csv(source_csv)
    output = pd.DataFrame()
    output["row_index"] = range(1, len(df) + 1)
    mappings = {
        "title": "Title",
        "final_decision": "Decision",
        "stage1_decision": "stage1_decision",
        "stage2_decision": "stage2_decision_raw",
        "final_adjudication_action": "final_adjudication_action",
        "directional_relation": "stage1_directional_relation",
        "workflow_use": "stage1_directional_uses_ai_for_review_workflow",
        "external_domain": "stage1_directional_is_review_about_ai_external_domain",
        "model_fusion_action": "stage1_model_fusion_action",
        "fast_mode_current_equivalence_blocked_reason": "stage1_fast_mode_current_equivalence_blocked_reason",
        "cache_missing_adjudication_fields": "stage1_cache_missing_adjudication_fields",
        "cache_schema_complete": "stage1_cache_schema_complete",
        "fast_preserved_current_uncertainty": "stage1_fast_preserved_current_uncertainty",
        "fast_recomputed_due_to_incomplete_cache": "stage1_fast_recomputed_due_to_incomplete_cache",
        "llm_route": "stage1_llm_route",
        "cache_source": "stage1_llm_directional_cache_source",
        "final_reason": "Reason",
        "stage1_processing_seconds": "stage1_processing_seconds",
        "perf_route": "perf_route",
    }
    for out_col, src_col in mappings.items():
        output[out_col] = df[src_col] if src_col in df.columns else ""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(save_path, index=False)


def dry_run_summary(csv_path):
    rows = pd.read_csv(csv_path).to_dict("records")
    return summarize_screening_rows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run a lightweight local screening benchmark.")
    parser.add_argument("config", nargs="?", help="Path to benchmark JSON config.")
    parser.add_argument(
        "--dry-run-summary",
        metavar="CSV",
        help="Summarize an existing screening CSV without running Ollama.",
    )
    parser.add_argument(
        "--model-mode",
        choices=["off", "fast", "balanced", "full"],
        default=None,
        help="Override MODEL_JUDGE_MODE for benchmark runs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override benchmark first_n/row_limit for a controlled debug run.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Explicitly ignore benchmark first_n/row_limit and run the full dataset.",
    )
    parser.add_argument("--save-rows", default="", help="Write compact row-level benchmark diagnostics CSV.")
    args = parser.parse_args()
    if not args.dry_run_summary and not args.config:
        parser.error("config is required unless --dry-run-summary is used")
    if args.model_mode and args.dry_run_summary:
        os.environ["MODEL_JUDGE_MODE"] = args.model_mode
    summary = (
        dry_run_summary(args.dry_run_summary)
        if args.dry_run_summary
        else run_benchmark(args.config, args.model_mode, args.limit, args.full, args.save_rows or None)
    )
    print(json.dumps(summary, indent=2))
    if not args.dry_run_summary and summary.get("expected_ranges_passed") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
