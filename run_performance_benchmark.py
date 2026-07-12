from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from benchmark_local_screening import run_benchmark
from semantic_frame import configure_semantic_frame_cache, get_semantic_frame_cache_stats


BENCHMARKS = [
    ("slr", "SLR", "benchmark_slr_automation_first100.json"),
    ("blockchain", "Blockchain", "benchmark_blockchain_first100.json"),
    ("medical", "Medical", "benchmark_heart_disease_first100.json"),
]
STABLE_SLR_BASELINE = Path("outputs/benchmark_rows/slr_current_after_stability_fix.csv")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["current", "two_pass_fast"], default="two_pass_fast")
    parser.add_argument("--semantic-cache", action="store_true")
    parser.add_argument("--no-batch-llm", action="store_true")
    parser.add_argument("--cache-state", choices=["cold", "warm", "both"], default="both")
    parser.add_argument("--cache-directory", default=None, help="Reuse an existing isolated benchmark cache directory.")
    parser.add_argument("--model-mode", choices=["off", "fast", "balanced", "full"], default="balanced")
    args = parser.parse_args()

    os.environ["SCREENING_PIPELINE_MODE"] = args.pipeline
    os.environ["ENABLE_SEMANTIC_FRAME_CACHE"] = "true" if args.semantic_cache else "false"
    os.environ["ENABLE_AGGRESSIVE_LLM_GATING"] = "true" if args.pipeline == "two_pass_fast" else "false"
    os.environ["ENABLE_BATCH_LLM_JUDGE"] = "false" if args.no_batch_llm else os.getenv("ENABLE_BATCH_LLM_JUDGE", "false")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_dir = Path(args.cache_directory) if args.cache_directory else Path("outputs/benchmark_cache") / timestamp
    states = ["cold", "warm"] if args.cache_state == "both" else [args.cache_state]
    rows = []

    for cache_state in states:
        for slug, label, config_path in BENCHMARKS:
            cache_path = cache_dir / f"{slug}_semantic_frames.jsonl"
            configure_semantic_frame_cache(str(cache_path), reset_memory=True)
            output_path = Path("outputs/benchmark_rows") / f"{slug}_fast_{cache_state}_{timestamp}.csv"
            started = time.perf_counter()
            summary = run_benchmark(config_path, model_mode=args.model_mode, save_rows=str(output_path))
            runtime = round(time.perf_counter() - started, 3)
            frame_stats = get_semantic_frame_cache_stats()
            df = pd.read_csv(output_path).fillna("")
            llm_calls = _truthy_count(df, "stage1_llm_directional_judge_used")
            workflow_reject = _safety_count(df, "stage1_directional_uses_ai_for_review_workflow", "REJECT")
            external_keep = _safety_count(df, "stage1_directional_is_review_about_ai_external_domain", "KEEP")
            lookups = int(frame_stats.get("semantic_frame_cache_hits", 0)) + int(frame_stats.get("semantic_frame_cache_misses", 0))
            rows.append({
                "dataset": label, "mode": args.pipeline, "cache_state": cache_state,
                "keep": summary["keep"], "maybe": summary["maybe"], "reject": summary["reject"],
                "parse_error": summary["parse_error"], "expected_passed": summary.get("expected_ranges_passed"),
                "runtime_seconds": runtime,
                "cache_hit_rate": round(int(frame_stats.get("semantic_frame_cache_hits", 0)) / lookups, 4) if lookups else 0.0,
                "llm_calls": llm_calls, "safety_passed": not workflow_reject and not external_keep,
                "workflow_reject_violations": workflow_reject, "external_keep_violations": external_keep,
                "row_csv": str(output_path), "semantic_frame_cache_stats": dict(frame_stats),
            })

    comparison_path = ""
    slr_fast = next((row for row in rows if row["dataset"] == "SLR" and row["cache_state"] == "cold"), None)
    if slr_fast and STABLE_SLR_BASELINE.exists():
        comparison = Path("outputs/benchmark_compare") / f"{timestamp}_current_vs_fast.txt"
        completed = subprocess.run(
            [sys.executable, "compare_benchmark_runs.py", str(STABLE_SLR_BASELINE), slr_fast["row_csv"], "--output", str(comparison)],
            check=False,
        )
        comparison_path = str(comparison)
        slr_fast["comparison_exit_code"] = completed.returncode

    report = {
        "timestamp": timestamp, "pipeline": args.pipeline,
        "stable_slr_baseline": str(STABLE_SLR_BASELINE), "cache_directory": str(cache_dir),
        "comparison_report": comparison_path, "runs": rows,
    }
    report_path = Path("outputs/benchmark_reports") / f"performance_{timestamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Dataset | Mode | Cache State | KEEP | MAYBE | REJECT | PARSE | Expected Passed | Runtime | Cache Hit Rate | LLM Calls | Safety Passed")
    for row in rows:
        print(f"{row['dataset']} | {row['mode']} | {row['cache_state']} | {row['keep']} | {row['maybe']} | {row['reject']} | {row['parse_error']} | {row['expected_passed']} | {row['runtime_seconds']} | {row['cache_hit_rate']} | {row['llm_calls']} | {row['safety_passed']}")
    print(f"performance_report={report_path}")
    return 0 if all(row["safety_passed"] and row["parse_error"] == 0 for row in rows) else 1


def _truthy_count(df: pd.DataFrame, column: str) -> int:
    if column not in df:
        return 0
    return int(df[column].astype(str).str.lower().isin({"true", "1", "yes"}).sum())


def _safety_count(df: pd.DataFrame, flag: str, decision: str) -> int:
    if flag not in df or "Decision" not in df:
        return 0
    return int((df[flag].astype(str).str.lower().isin({"true", "1", "yes"}) & df["Decision"].astype(str).eq(decision)).sum())


if __name__ == "__main__":
    raise SystemExit(main())
