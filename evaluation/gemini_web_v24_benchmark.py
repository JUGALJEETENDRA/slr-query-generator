from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.gemini_web_mixed_control import (
    compare_screening_runs,
    score_mixed_control,
)


VERSIONS = {
    "v2.3": {
        "engine": "gemini_web",
        "architecture": "gemini-web-batched-v2.3",
        "cache_directory": "gemini_web",
    },
    "v2.4": {
        "engine": "gemini_web_v24",
        "architecture": "gemini-web-batched-v2.4",
        "cache_directory": "gemini_web_v24",
    },
}


def _quality_metrics(score: dict[str, Any]) -> dict[str, Any]:
    confusion = score["confusion"]
    relevant = confusion["KEEP"]
    relevant_total = sum(relevant.values())
    predicted_keep = sum(confusion[label]["KEEP"] for label in confusion)
    total = score["matched_rows"]
    maybe_total = sum(confusion[label]["MAYBE"] for label in confusion)
    return {
        "relevant_recall_keep_or_maybe": (
            round((relevant["KEEP"] + relevant["MAYBE"]) / relevant_total, 4)
            if relevant_total else None
        ),
        "false_reject_rate": (
            round(relevant["REJECT"] / relevant_total, 4) if relevant_total else None
        ),
        "definitive_keep_precision": (
            round(relevant["KEEP"] / predicted_keep, 4) if predicted_keep else None
        ),
        "manual_review_rate": round(maybe_total / total, 4) if total else None,
    }


def _run_once(
    *,
    version: str,
    papers: Path,
    gold: Path,
    question: str,
    context: str,
    inclusion: str,
    exclusion: str,
    output_root: Path,
    repetition: int,
) -> dict[str, Any]:
    from bulk_screen import screen_csv

    definition = VERSIONS[version]
    job_id = f"gweb-benchmark-{version.replace('.', '-')}-{repetition}-{uuid.uuid4()}"
    output_path = output_root / f"fresh-{repetition}" / version / "screened.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = screen_csv(
        csv_path=str(papers),
        research_question=question,
        research_context=context,
        inclusion_criteria=inclusion,
        exclusion_criteria=exclusion,
        output_path=str(output_path),
        progress_job_id=job_id,
        screening_engine=definition["engine"],
        resume=False,
    )
    if summary.get("architecture_version") != definition["architecture"]:
        raise RuntimeError(
            f"{version} used unexpected architecture {summary.get('architecture_version')!r}"
        )
    diagnostics_summary = Path(str(summary["diagnostics_path"])).with_suffix(".summary.json")
    score = score_mixed_control(
        output_path,
        gold,
        diagnostics_path=diagnostics_summary,
    )
    return {
        "version": version,
        "repetition": repetition,
        "job_id": job_id,
        "screened": str(output_path),
        "diagnostics_summary": str(diagnostics_summary),
        "summary": summary,
        "score": score,
        "quality": _quality_metrics(score),
    }


def _seed_immutable_protocol(
    output_root: Path, version: str, *, source_repetition: int, target_repetition: int,
) -> None:
    cache_directory = VERSIONS[version]["cache_directory"]
    source = (
        output_root / f"fresh-{source_repetition}" / "cache"
        / cache_directory / "protocols"
    )
    target = (
        output_root / f"fresh-{target_repetition}" / "cache"
        / cache_directory / "protocols"
    )
    if not source.exists():
        raise RuntimeError(f"{version} did not produce an immutable protocol cache")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)


def run_benchmark(
    *,
    papers: str | Path,
    gold: str | Path,
    question: str,
    context: str = "",
    inclusion: str = "",
    exclusion: str = "",
    output_root: str | Path,
    repetitions: int = 2,
    versions: tuple[str, ...] = ("v2.3", "v2.4"),
    limit: int = 0,
) -> dict[str, Any]:
    if os.getenv("GEMINI_WEB_CAPTURE_RAW_DEBUG", "").strip().casefold() in {
        "1", "true", "yes",
    }:
        raise RuntimeError("benchmark requires raw Gemini response capture to be disabled")
    if repetitions not in {1, 2}:
        raise ValueError("repetitions must be one or two")
    unknown_versions = set(versions) - set(VERSIONS)
    if unknown_versions or not versions:
        raise ValueError(f"unknown or empty version selection: {sorted(unknown_versions)}")
    papers_path = Path(papers)
    gold_path = Path(gold)
    root = Path(output_root)
    if limit:
        if limit < 1:
            raise ValueError("limit must be positive")
        subset_root = root / "blinded-subset"
        subset_root.mkdir(parents=True, exist_ok=True)
        papers_frame = pd.read_csv(papers_path).head(limit)
        source_ids = {str(index) for index in papers_frame.index}
        gold_frame = pd.read_csv(gold_path)
        gold_frame = gold_frame[
            gold_frame["Source_Row_Index"].astype(str).isin(source_ids)
        ]
        papers_path = subset_root / "papers.csv"
        gold_path = subset_root / "gold.csv"
        papers_frame.to_csv(papers_path, index=False)
        gold_frame.to_csv(gold_path, index=False)
    runs: dict[str, list[dict[str, Any]]] = {version: [] for version in versions}
    for version in versions:
        for repetition in range(1, repetitions + 1):
            if repetition > 1:
                _seed_immutable_protocol(
                    root, version, source_repetition=1, target_repetition=repetition,
                )
            runs[version].append(_run_once(
                version=version,
                papers=papers_path,
                gold=gold_path,
                question=question,
                context=context,
                inclusion=inclusion,
                exclusion=exclusion,
                output_root=root,
                repetition=repetition,
            ))

    repeatability = {}
    if repetitions == 2:
        for version, version_runs in runs.items():
            repeatability[version] = compare_screening_runs(
                version_runs[0]["screened"],
                version_runs[1]["screened"],
                gold_path=gold_path,
            )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "papers": str(papers_path),
        "gold": str(gold_path),
        "repetitions": repetitions,
        "versions": list(versions),
        "limit": limit,
        "raw_debug_capture": False,
        "runs": runs,
        "repeatability": repeatability,
        "comparison": {
            version: {
                "runtime_seconds": [run["summary"]["runtime_seconds"] for run in version_runs],
                "attempt_count": [run["summary"].get("attempt_count") for run in version_runs],
                "verification_count": [
                    run["summary"].get("verification_count", run["summary"].get("escalated_count"))
                    for run in version_runs
                ],
                "retry_count": [run["summary"].get("retry_count") for run in version_runs],
                "timeout_fallback_count": [
                    run["summary"].get("timeout_fallback_count") for run in version_runs
                ],
                "quality": [run["quality"] for run in version_runs],
            }
            for version, version_runs in runs.items()
        },
    }
    report_path = root / "benchmark-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {**report, "report_path": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare frozen Gemini Web v2.3 with the opt-in v2.4 candidate."
    )
    parser.add_argument("--papers", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--inclusion", default="")
    parser.add_argument("--exclusion", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repetitions", type=int, choices=(1, 2), default=2)
    parser.add_argument("--versions", nargs="+", choices=tuple(VERSIONS), default=list(VERSIONS))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    report = run_benchmark(
        papers=args.papers,
        gold=args.gold,
        question=args.question,
        context=args.context,
        inclusion=args.inclusion,
        exclusion=args.exclusion,
        output_root=args.output_root,
        repetitions=args.repetitions,
        versions=tuple(args.versions),
        limit=args.limit,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
