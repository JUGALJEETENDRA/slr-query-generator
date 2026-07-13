import argparse
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

from benchmark_local_screening import run_benchmark


"""
Digital twin holdout/generalization runner.

This dataset is a final holdout test path, separate from the internal SLR,
blockchain, and medical benchmark gates. Do not use this runner to tune rules
unless a general bug is found. First-100 ranges are loose sanity bounds; the
full run is for distribution and runtime sanity, not strict pass/fail.
"""

FIRST100_CONFIG = Path("benchmark_digital_twin_holdout_first100.json")
FULL_CONFIG = Path("benchmark_digital_twin_holdout_full.json")
FIRST100_ROWS = Path("outputs/benchmark_rows/digital_twin_holdout_first100_stage0.csv")
FULL_ROWS = Path("outputs/benchmark_rows/digital_twin_holdout_full_stage0.csv")
REPORT_DIR = Path("outputs/benchmark_reports")

HOLDOUT_ENV = {
    "SCREENING_PIPELINE_MODE": "two_pass_fast",
    "ENABLE_STAGE0_FAST_TRIAGE": "true",
    "ENABLE_HEURISTIC_FAST_FRAMES": "true",
    "ENABLE_BATCH_SEMANTIC_FRAME_EXTRACTION": "false",
    "ENABLE_PARALLEL_SCREENING": "false",
    "ENABLE_CURRENT_MODE_CACHE": "false",
    "ENABLE_AGGRESSIVE_LLM_GATING": "true",
    "ENABLE_BATCH_LLM_JUDGE": "false",
    "MODEL_JUDGE_MODE": "balanced",
    "MODEL_JUDGE_PROFILE": "light",
    "ENABLE_MODEL_JUDGES": "true",
    "ENABLE_HF_MODEL_LOADING": "false",
    "ENABLE_HF_MODEL_DOWNLOAD": "false",
    "ENABLE_LLM_JUDGE": "true",
    "ENABLE_DOMAIN_LLM_JUDGE": "true",
    "MODEL_JUDGE_TIMEOUT_SECONDS": "60",
}


def main():
    parser = argparse.ArgumentParser(description="Run the digital twin holdout benchmark.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--first100", action="store_true", help="Run the first-100 holdout sanity test.")
    mode.add_argument("--full", action="store_true", help="Run the full holdout dataset.")
    parser.add_argument(
        "--cache",
        choices=["on", "off"],
        default="off",
        help="Semantic-frame cache mode. Default is off for a cold holdout test.",
    )
    parser.add_argument(
        "--save-rows",
        default="",
        help="Optional compact row diagnostics CSV path. Defaults to the holdout path for the selected mode.",
    )
    args = parser.parse_args()

    run_full = bool(args.full)
    config_path = FULL_CONFIG if run_full else FIRST100_CONFIG
    save_rows = Path(args.save_rows) if args.save_rows else (FULL_ROWS if run_full else FIRST100_ROWS)

    env = dict(HOLDOUT_ENV)
    env["ENABLE_SEMANTIC_FRAME_CACHE"] = "true" if args.cache == "on" else "false"

    with _temporary_env(env):
        summary = run_benchmark(
            str(config_path),
            model_mode="balanced",
            full=run_full,
            save_rows=str(save_rows),
        )

    full_output = Path(_load_config(config_path).get("output_path", "outputs/benchmark_screened.csv"))
    summary = _augment_summary(summary, full_output, save_rows)
    report_path = _write_report(summary, mode="full" if run_full else "first100")
    _print_summary(summary, save_rows, report_path)


def _load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def _temporary_env(values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _augment_summary(summary, full_output, save_rows):
    rows = pd.read_csv(full_output) if full_output.exists() else pd.DataFrame()
    total = int(summary.get("total") or len(rows))
    perf = summary.get("performance_summary", {})
    runtime = float(perf.get("total_screening_seconds") or 0.0)
    if not runtime and "stage1_processing_seconds" in rows:
        runtime = float(pd.to_numeric(rows["stage1_processing_seconds"], errors="coerce").fillna(0).sum())

    stage0 = {
        "heuristic_frame_count": _count_true(rows, "stage1_heuristic_frame_used"),
        "full_extraction_count": _count_true(rows, "stage1_stage0_requires_full_extraction"),
        "ollama_calls_avoided": _sum_numeric(rows, "stage1_stage0_ollama_calls_avoided"),
    }
    summary["holdout_summary"] = {
        "total_rows_screened": total,
        "runtime_seconds": round(runtime, 3),
        "average_seconds_per_paper": round(runtime / total, 3) if total else 0.0,
        "stage0": stage0,
        "output_csv_path": str(full_output),
        "row_csv_path": str(save_rows),
    }
    return summary


def _count_true(df, column):
    if column not in df:
        return 0
    return int(df[column].astype(str).str.lower().isin({"true", "1", "yes"}).sum())


def _sum_numeric(df, column):
    if column not in df:
        return 0
    return int(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())


def _write_report(summary, mode):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"digital_twin_holdout_{mode}_{timestamp}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def _print_summary(summary, save_rows, report_path):
    holdout = summary.get("holdout_summary", {})
    print("========== DIGITAL TWIN HOLDOUT SUMMARY ==========")
    print(f"total rows screened: {summary.get('total', 0)}")
    print(f"KEEP: {summary.get('keep', 0)}")
    print(f"MAYBE: {summary.get('maybe', 0)}")
    print(f"REJECT: {summary.get('reject', 0)}")
    print(f"PARSE_ERROR: {summary.get('parse_error', 0)}")
    print(f"expected_ranges_passed: {summary.get('expected_ranges_passed')}")
    print(f"runtime_seconds: {holdout.get('runtime_seconds', 0.0)}")
    print(f"average_seconds_per_paper: {holdout.get('average_seconds_per_paper', 0.0)}")
    stage0 = holdout.get("stage0", {})
    print(f"stage0 heuristic frame count: {stage0.get('heuristic_frame_count', 0)}")
    print(f"stage0 full extraction count: {stage0.get('full_extraction_count', 0)}")
    print(f"stage0 ollama calls avoided: {stage0.get('ollama_calls_avoided', 0)}")
    print(f"top suspicious false rejects: {summary.get('top_suspicious_false_rejects', [])}")
    print(f"top suspicious keeps: {summary.get('top_suspicious_keeps', [])}")
    print(f"output CSV path: {holdout.get('output_csv_path', '')}")
    print(f"row CSV path: {save_rows}")
    print(f"report path: {report_path}")
    print("==================================================")


if __name__ == "__main__":
    main()
