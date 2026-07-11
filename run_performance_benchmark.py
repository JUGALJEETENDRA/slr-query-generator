from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from benchmark_local_screening import run_benchmark


BENCHMARKS = [
    ("SLR", "benchmark_slr_automation_first100.json"),
    ("Blockchain", "benchmark_blockchain_first100.json"),
    ("Medical", "benchmark_heart_disease_first100.json"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["current", "two_pass_fast"], default=None)
    parser.add_argument("--batch-llm", action="store_true")
    parser.add_argument("--model-mode", choices=["off", "fast", "balanced", "full"], default="balanced")
    args = parser.parse_args()

    if args.pipeline:
        os.environ["SCREENING_PIPELINE_MODE"] = args.pipeline
    if args.batch_llm:
        os.environ["ENABLE_BATCH_LLM_JUDGE"] = "true"

    rows = []
    for name, config_path in BENCHMARKS:
        started = time.perf_counter()
        summary = run_benchmark(config_path, model_mode=args.model_mode)
        runtime = round(time.perf_counter() - started, 3)
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        output_path = config["output_path"]
        llm_calls = cache_hits = avg_seconds = 0
        if Path(output_path).exists():
            df = pd.read_csv(output_path)
            llm_calls = int(df.get("stage1_llm_directional_judge_used", pd.Series([], dtype=bool)).astype(str).str.lower().eq("true").sum())
            cache_hits = int(df.get("stage1_llm_directional_cache_hit", pd.Series([], dtype=bool)).astype(str).str.lower().eq("true").sum())
            avg_seconds = round(float(df.get("stage1_processing_seconds", pd.Series([0.0])).astype(float).mean()), 3)
        rows.append({
            "Dataset": name,
            "KEEP": summary["keep"],
            "MAYBE": summary["maybe"],
            "REJECT": summary["reject"],
            "PARSE_ERROR": summary["parse_error"],
            "expected_passed": summary.get("expected_ranges_passed"),
            "runtime": runtime,
            "llm_calls": llm_calls,
            "cache_hits": cache_hits,
            "avg_seconds": avg_seconds,
        })

    print("Dataset | KEEP | MAYBE | REJECT | PARSE_ERROR | expected_passed | runtime | llm_calls | cache_hits | avg_seconds")
    for row in rows:
        print(
            f"{row['Dataset']} | {row['KEEP']} | {row['MAYBE']} | {row['REJECT']} | "
            f"{row['PARSE_ERROR']} | {row['expected_passed']} | {row['runtime']} | "
            f"{row['llm_calls']} | {row['cache_hits']} | {row['avg_seconds']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
