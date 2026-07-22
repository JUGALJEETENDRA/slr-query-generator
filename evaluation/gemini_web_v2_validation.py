from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from evaluation.gemini_web_mixed_control import compare_screening_runs, score_mixed_control
from gemini_web_screening import GEMINI_WEB_VERSION


BASELINE_QUESTION = (
    "How can blockchain technology improve transparency, traceability, trust, and security "
    "in supply chain management?"
)
DIGITAL_TWIN_QUESTION = (
    "How are digital twins used in smart manufacturing and Industry 4.0/5.0 applications?"
)
BLOCKCHAIN_SOURCE = Path(
    "uploads/6a19d17c-2649-4f01-a886-4de75edc7748-LitSync_Clean_Dataset_2026-06-13 (1) (2).csv"
)
DIGITAL_TWIN_SOURCE = Path("data/holdout/LitSync_Clean_Dataset_2026-07-09.csv")
BASELINE_PAPERS = Path(".benchmarks/gemini_web_mixed_control_40.csv")
BASELINE_GOLD = Path(".benchmarks/gemini_web_mixed_control_40.gold.csv")

HARD_RELEVANT_SOURCE_ROWS = tuple(range(20, 35)) + tuple(range(36, 41))
HARD_NEAR_MISS_SOURCE_ROWS = (
    268, 331, 368, 781, 410, 416, 424, 478, 479, 512,
    525, 539, 542, 553, 561, 945, 848, 950, 958, 971,
)
HARD_AMBIGUOUS_SOURCE_ROWS = {331, 368}

DIGITAL_TWIN_REJECT_POSITIONS = {
    11, 15, 17, 21, 28, 33, 35, 40, 60, 65, 79, 82, 83,
}
DIGITAL_TWIN_MAYBE_POSITIONS = {1, 4, 13, 18, 22, 23, 24, 31, 34, 42, 47, 78, 95}

APPROVED_DIAGNOSTIC_FIELDS = {
    "event", "submission_number", "stage", "retry_number", "outcome",
    "recovery_action", "attempt_duration_ms", "response_selector",
    "response_container_count", "response_state", "generation_detected",
    "timeout_stage", "fallback_reason",
}
FORBIDDEN_DIAGNOSTIC_FIELDS = {
    "prompt", "question", "research_question", "title", "abstract",
    "response", "raw_response", "response_text", "content", "content_hash",
}


def _complete_rows(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for source_index, row in frame.iterrows():
        title = "" if pd.isna(row.get("Title")) else str(row.get("Title")).strip()
        abstract = "" if pd.isna(row.get("Abstract")) else str(row.get("Abstract")).strip()
        if title and abstract:
            rows.append({
                "source_dataset_row": int(source_index),
                "Title": title,
                "Abstract": abstract,
            })
    return rows


def _write_fixture(
    papers_path: Path,
    gold_path: Path,
    rows: list[dict[str, Any]],
    labels: list[dict[str, str]],
) -> None:
    if len(rows) != len(labels):
        raise ValueError("fixture rows and labels must have identical lengths")
    papers = [{"Title": row["Title"], "Abstract": row["Abstract"]} for row in rows]
    gold = []
    for source_row_index, (row, label) in enumerate(zip(rows, labels)):
        gold.append({
            "Source_Row_Index": source_row_index,
            "Gold_Decision": label["decision"],
            "Control_Group": label["group"],
            "Adjudication_Rationale": label["rationale"],
            "Source_Dataset_Row": row["source_dataset_row"],
            "Title": row["Title"],
        })
    papers_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(papers, columns=["Title", "Abstract"]).to_csv(papers_path, index=False)
    pd.DataFrame(gold).to_csv(gold_path, index=False)
    _assert_blinded_fixture(papers_path, gold_path)


def _assert_blinded_fixture(papers_path: Path, gold_path: Path) -> None:
    papers = pd.read_csv(papers_path)
    gold = pd.read_csv(gold_path)
    if list(papers.columns) != ["Title", "Abstract"]:
        raise ValueError(f"screening fixture is not blinded: {list(papers.columns)}")
    forbidden = {"Gold_Decision", "Control_Group", "Adjudication_Rationale"}
    if forbidden & set(papers.columns):
        raise ValueError("gold metadata leaked into the screening fixture")
    if len(papers) != len(gold):
        raise ValueError("screening fixture and gold sidecar have different row counts")


def _hard_fixture(root: Path) -> tuple[Path, Path]:
    source = pd.read_csv(BLOCKCHAIN_SOURCE)
    selected_rows: list[dict[str, Any]] = []
    labels: list[dict[str, str]] = []
    for relevant_index, near_miss_index in zip(
        HARD_RELEVANT_SOURCE_ROWS, HARD_NEAR_MISS_SOURCE_ROWS
    ):
        relevant = source.iloc[relevant_index]
        near_miss = source.iloc[near_miss_index]
        selected_rows.extend([
            {
                "source_dataset_row": relevant_index,
                "Title": str(relevant["Title"]).strip(),
                "Abstract": str(relevant["Abstract"]).strip(),
            },
            {
                "source_dataset_row": near_miss_index,
                "Title": str(near_miss["Title"]).strip(),
                "Abstract": str(near_miss["Abstract"]).strip(),
            },
        ])
        labels.append({
            "decision": "KEEP",
            "group": "hard_relevant",
            "rationale": "The abstract directly evaluates blockchain in a supply-chain application.",
        })
        if near_miss_index in HARD_AMBIGUOUS_SOURCE_ROWS:
            labels.append({
                "decision": "MAYBE",
                "group": "hard_ambiguous",
                "rationale": (
                    "The paper's primary application is outside supply-chain management, but the "
                    "abstract explicitly claims a secondary supply-chain use, so title/abstract "
                    "screening cannot resolve eligibility confidently."
                ),
            })
        else:
            labels.append({
                "decision": "REJECT",
                "group": "hard_near_miss",
                "rationale": (
                    "The abstract discusses blockchain and adjacent outcomes but not a supply-chain "
                    "management application answering the research question."
                ),
            })
    papers = root / "fixtures" / "blockchain_hard_40.csv"
    gold = root / "gold" / "blockchain_hard_40.gold.csv"
    _write_fixture(papers, gold, selected_rows, labels)
    return papers, gold


def _digital_twin_fixture(root: Path) -> tuple[Path, Path]:
    rows = _complete_rows(DIGITAL_TWIN_SOURCE)[:100]
    if len(rows) != 100:
        raise ValueError("digital-twin source does not contain 100 complete papers")
    labels: list[dict[str, str]] = []
    for position in range(100):
        if position in DIGITAL_TWIN_REJECT_POSITIONS:
            labels.append({
                "decision": "REJECT",
                "group": "cross_domain_negative",
                "rationale": (
                    "The abstract is outside smart manufacturing/Industry 4.0/5.0 or does not "
                    "substantively study digital twins."
                ),
            })
        elif position in DIGITAL_TWIN_MAYBE_POSITIONS:
            labels.append({
                "decision": "MAYBE",
                "group": "cross_domain_ambiguous",
                "rationale": (
                    "The abstract discusses digital twins or Industry 4.0, but title/abstract evidence "
                    "is insufficient to establish the required intersection confidently."
                ),
            })
        else:
            labels.append({
                "decision": "KEEP",
                "group": "cross_domain_relevant",
                "rationale": (
                    "The abstract directly discusses a digital-twin use, method, or enabling role in "
                    "smart manufacturing or an Industry 4.0/5.0 application."
                ),
            })
    papers = root / "fixtures" / "digital_twin_natural_100.csv"
    gold = root / "gold" / "digital_twin_natural_100.gold.csv"
    _write_fixture(papers, gold, rows, labels)
    return papers, gold


def prepare_validation_suite(root: str | Path) -> dict[str, Any]:
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    baseline_papers = target / "fixtures" / "blockchain_baseline_40.csv"
    baseline_gold = target / "gold" / "blockchain_baseline_40.gold.csv"
    baseline_papers.parent.mkdir(parents=True, exist_ok=True)
    baseline_gold.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BASELINE_PAPERS, baseline_papers)
    shutil.copyfile(BASELINE_GOLD, baseline_gold)
    _assert_blinded_fixture(baseline_papers, baseline_gold)
    hard_papers, hard_gold = _hard_fixture(target)
    digital_papers, digital_gold = _digital_twin_fixture(target)
    cases = [
        {
            "name": "fresh_baseline_40", "papers": str(baseline_papers),
            "gold": str(baseline_gold), "question": BASELINE_QUESTION,
        },
        {
            "name": "hard_control_a_40", "papers": str(hard_papers),
            "gold": str(hard_gold), "question": BASELINE_QUESTION,
        },
        {
            "name": "hard_control_b_40", "papers": str(hard_papers),
            "gold": str(hard_gold), "question": BASELINE_QUESTION,
        },
        {
            "name": "digital_twin_natural_100", "papers": str(digital_papers),
            "gold": str(digital_gold), "question": DIGITAL_TWIN_QUESTION,
        },
    ]
    manifest = {
        "suite": "gemini_web_v2_3_validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": "e5626e68419871ec74c64f787a7a530bf473c80f",
        "candidate_version": GEMINI_WEB_VERSION,
        "production_change_scope": [
            "acronym_grounding", "bounded_browser_lifecycle", "verified_decision_routing",
        ],
        "raw_debug_capture": False,
        "total_papers": 220,
        "cases": cases,
    }
    manifest_path = target / "suite-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path)}


def validate_diagnostics(path: str | Path) -> dict[str, Any]:
    events = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        unexpected = set(event) - APPROVED_DIAGNOSTIC_FIELDS
        forbidden = set(event) & FORBIDDEN_DIAGNOSTIC_FIELDS
        if unexpected or forbidden:
            raise ValueError(
                f"unsafe Gemini Web diagnostic fields: {sorted(unexpected | forbidden)}"
            )
        events.append(event)
    return {"event_count": len(events), "approved_fields_only": True}


def _raw_debug_enabled() -> bool:
    return os.getenv("GEMINI_WEB_CAPTURE_RAW_DEBUG", "").strip().casefold() in {
        "1", "true", "yes",
    }


def run_validation_suite(
    manifest_path: str | Path,
    *,
    screen: Callable[..., dict[str, Any]] | None = None,
    case_names: set[str] | None = None,
) -> dict[str, Any]:
    if _raw_debug_enabled():
        raise RuntimeError("validation requires GEMINI_WEB_CAPTURE_RAW_DEBUG to remain disabled")
    if screen is None:
        from bulk_screen import screen_csv
        screen_function = screen_csv
    else:
        screen_function = screen
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    artifact_root = manifest_file.parent
    run_records = []
    selected_cases = [
        case for case in manifest["cases"]
        if not case_names or case["name"] in case_names
    ]
    if case_names and len(selected_cases) != len(case_names):
        known = {case["name"] for case in manifest["cases"]}
        raise ValueError(f"unknown validation cases: {sorted(case_names - known)}")
    for case in selected_cases:
        job_id = f"gweb-v2-validation-{case['name']}-{uuid.uuid4()}"
        output_path = Path("outputs") / "runs" / f"screened-{job_id}.csv"
        summary = screen_function(
            csv_path=case["papers"],
            research_question=case["question"],
            output_path=str(output_path),
            progress_job_id=job_id,
            screening_engine="gemini_web",
            resume=False,
        )
        if int(summary.get("resumed_count", -1)) != 0:
            raise RuntimeError(f"{case['name']} was not a fresh screening run")
        diagnostics_path = Path(str(summary["diagnostics_path"]))
        diagnostics_check = validate_diagnostics(diagnostics_path)
        diagnostics_summary = diagnostics_path.with_suffix(".summary.json")
        report = score_mixed_control(
            output_path, case["gold"], diagnostics_path=diagnostics_summary,
        )
        case_record = {
            "name": case["name"],
            "job_id": job_id,
            "papers": case["papers"],
            "gold": case["gold"],
            "screened": str(output_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_summary": str(diagnostics_summary),
            "diagnostics_check": diagnostics_check,
            "run_summary": summary,
            "score": report,
        }
        run_records.append(case_record)
        (artifact_root / f"{case['name']}.report.json").write_text(
            json.dumps(case_record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    by_name = {record["name"]: record for record in run_records}
    repeatability = _repeatability(by_name)
    suite_report = _suite_report(run_records, repeatability)
    report_path = artifact_root / "suite-report.json"
    report_path.write_text(
        json.dumps(suite_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {**suite_report, "report_path": str(report_path)}


def rescore_validation_suite(manifest_path: str | Path) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    records = []
    for case in manifest["cases"]:
        report_path = manifest_file.parent / f"{case['name']}.report.json"
        record = json.loads(report_path.read_text(encoding="utf-8"))
        record["gold"] = case["gold"]
        record["score"] = score_mixed_control(
            record["screened"], case["gold"],
            diagnostics_path=record["diagnostics_summary"],
        )
        report_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        records.append(record)
    by_name = {record["name"]: record for record in records}
    suite_report = _suite_report(records, _repeatability(by_name))
    report_path = manifest_file.parent / "suite-report.json"
    report_path.write_text(
        json.dumps(suite_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {**suite_report, "report_path": str(report_path)}


def _repeatability(by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
    required = {"hard_control_a_40", "hard_control_b_40"}
    if not required <= set(by_name):
        return {"available": False, "passed": False}
    return compare_screening_runs(
        by_name["hard_control_a_40"]["screened"],
        by_name["hard_control_b_40"]["screened"],
        gold_path=by_name["hard_control_a_40"]["gold"],
    )


def _suite_report(
    records: list[dict[str, Any]], repeatability: dict[str, Any]
) -> dict[str, Any]:
    hard_records = [record for record in records if record["name"].startswith("hard_control_")]
    repeated_false_keeps = _repeated_failure_keys(hard_records, "false_keeps")
    repeated_positive_rejects = _repeated_failure_keys(hard_records, "positive_rejects")
    transport_failure_cases = [
        record["name"] for record in records
        if int(record["score"].get("timeout_fallback_count") or 0) > 0
    ]
    repeated_transport_failure = len(transport_failure_cases) >= 2
    reproducible_failures = bool(
        repeated_false_keeps or repeated_positive_rejects or repeated_transport_failure
    )
    case_summaries = []
    for record in records:
        score = record["score"]
        runtime = score.get("runtime_seconds")
        rows = score.get("screened_rows") or 0
        case_summaries.append({
            "name": record["name"],
            "screened_rows": rows,
            "runtime_seconds": runtime,
            "seconds_per_paper": round(float(runtime) / rows, 4) if runtime and rows else None,
            "retry_count": score.get("retry_count"),
            "timeout_fallback_count": score.get("timeout_fallback_count"),
            "timeout_fallback_row_count": score.get("timeout_fallback_row_count"),
            "invalid_structured_count": score.get("invalid_structured_count"),
            "unverified_definitive_count": score.get("unverified_definitive_count"),
            "scope_support_counts": score.get("scope_support_counts"),
            "non_substantive_maybe_count": score.get("non_substantive_maybe_count"),
            "false_keep_count": score.get("false_keep_count"),
            "positive_reject_count": score.get("positive_reject_count"),
            "unresolved_near_miss_count": score.get("unresolved_near_miss_count"),
            "detector_outcomes": score.get("detector_outcomes"),
            "recovery_actions": score.get("recovery_actions"),
            "critic_route_counts": score.get("critic_route_counts"),
            "verification_outcomes": score.get("verification_outcomes"),
            "verified_reject_count": score.get("verified_reject_count"),
            "verification_fallback_count": score.get("verification_fallback_count"),
            "protocol_cache_version": score.get("protocol_cache_version"),
            "clean_chat_rotations": score.get("clean_chat_rotations"),
            "passed": score.get("passed"),
        })
    return {
        "suite": "gemini_web_v2_3_validation",
        "case_summaries": case_summaries,
        "repeatability": repeatability,
        "repeated_false_keep_rows": repeated_false_keeps,
        "repeated_positive_reject_rows": repeated_positive_rejects,
        "repeated_transport_failure": repeated_transport_failure,
        "transport_failure_cases": transport_failure_cases,
        "production_change_supported": reproducible_failures,
        "recommendation": _recommendation(
            repeated_false_keeps, repeated_positive_rejects, repeated_transport_failure
        ),
        "cases": records,
    }


def _recommendation(
    repeated_false_keeps: list[str],
    repeated_positive_rejects: list[str],
    repeated_transport_failure: bool,
) -> str:
    if repeated_false_keeps or repeated_positive_rejects:
        return (
            "Reproduce the repeated semantic failures in a focused control, then propose a narrowly "
            "targeted protocol change without weakening evidence validation."
        )
    if repeated_transport_failure:
        return (
            "Keep the semantic and evidence policy unchanged. Prototype a narrowly targeted transport "
            "recovery change for repeated late-session no-container timeouts: recycle the browser "
            "context and apply bounded backoff before the existing single retry, while preserving safe "
            "MAYBE fallback behavior."
        )
    return "Retain the current Gemini Web candidate; no repeatable production weakness was observed."


def _repeated_failure_keys(
    records: list[dict[str, Any]], field: str
) -> list[str]:
    key_sets = [
        {str(item["source_row_index"]) for item in record["score"].get(field, [])}
        for record in records
    ]
    if not key_sets:
        return []
    return sorted(set.intersection(*key_sets), key=lambda value: int(value))


def _default_artifact_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(".codex_test_outputs") / "gemini-web-v2.3-validation" / stamp


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and run the Gemini Web v2.x validation suite.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--artifact-root")
    run = commands.add_parser("run")
    run.add_argument("--manifest")
    run.add_argument("--artifact-root")
    run.add_argument("--case", action="append", dest="cases")
    rescore = commands.add_parser("score-suite")
    rescore.add_argument("--manifest", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare_validation_suite(args.artifact_root or _default_artifact_root())
    elif args.command == "run":
        if args.manifest:
            manifest = args.manifest
        else:
            prepared = prepare_validation_suite(args.artifact_root or _default_artifact_root())
            manifest = prepared["manifest_path"]
        result = run_validation_suite(manifest, case_names=set(args.cases or []))
    else:
        result = rescore_validation_suite(args.manifest)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
