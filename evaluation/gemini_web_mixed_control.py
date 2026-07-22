from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


VALID_DECISIONS = {"KEEP", "MAYBE", "REJECT"}
STRUCTURED_COLUMNS = (
    "Criteria_JSON", "Evidence_JSON", "Layer_Trace_JSON",
    "Uncertainty_JSON", "Contradictions_JSON", "Validation_Errors",
)


def _column(frame: pd.DataFrame, name: str) -> str:
    matches = [column for column in frame.columns if str(column).strip().casefold() == name.casefold()]
    if not matches:
        raise ValueError(f"CSV is missing a {name!r} column")
    return str(matches[0])


def _eligible_rows(path: str | Path, count: int) -> list[dict[str, str]]:
    frame = pd.read_csv(path)
    title_column = _column(frame, "Title")
    abstract_column = _column(frame, "Abstract")
    selected: list[dict[str, str]] = []
    for _, row in frame.iterrows():
        title = "" if pd.isna(row[title_column]) else str(row[title_column]).strip()
        abstract = "" if pd.isna(row[abstract_column]) else str(row[abstract_column]).strip()
        if not title or not abstract:
            continue
        selected.append({"Title": title, "Abstract": abstract})
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"{path} contains only {len(selected)} complete papers; {count} are required")
    return selected


def build_mixed_control(
    positive_path: str | Path,
    negative_path: str | Path,
    papers_path: str | Path,
    gold_path: str | Path,
    *,
    per_group: int = 20,
) -> dict[str, Any]:
    """Build an alternating, label-blinded screening CSV plus a gold sidecar."""
    if per_group < 1:
        raise ValueError("per_group must be at least 1")
    positives = _eligible_rows(positive_path, per_group)
    negatives = _eligible_rows(negative_path, per_group)
    papers: list[dict[str, str]] = []
    gold: list[dict[str, Any]] = []
    for offset in range(per_group):
        for source, decision, group in (
            (positives[offset], "KEEP", "clearly_relevant"),
            (negatives[offset], "REJECT", "clear_negative"),
        ):
            source_index = len(papers)
            papers.append(dict(source))
            gold.append({
                "Source_Row_Index": source_index,
                "Gold_Decision": decision,
                "Control_Group": group,
                "Title": source["Title"],
            })
    papers_output = Path(papers_path)
    gold_output = Path(gold_path)
    papers_output.parent.mkdir(parents=True, exist_ok=True)
    gold_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(papers).to_csv(papers_output, index=False)
    pd.DataFrame(gold).to_csv(gold_output, index=False)
    return {
        "papers_path": str(papers_output),
        "gold_path": str(gold_output),
        "total_papers": len(papers),
        "per_group": per_group,
        "screening_columns": list(pd.DataFrame(papers).columns),
        "labels_blinded": True,
    }


def _source_key(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text


def _json_list(value: Any) -> list | None:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value or "[]"))
        return parsed if isinstance(parsed, list) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _truthy(value: Any) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes"}


def load_run_diagnostics(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.suffix.casefold() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    outcomes: dict[str, int] = {}
    timeout_fallback_count = 0
    attempt_count = 0
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        outcome = str(event.get("outcome") or "unknown")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        attempt_count += int(event.get("event") == "gemini_web_attempt")
        timeout_fallback_count += int(event.get("event") == "gemini_web_fallback")
    return {
        "attempt_count": attempt_count,
        "timeout_fallback_count": timeout_fallback_count,
        "detector_outcomes": outcomes,
    }


def score_mixed_control(
    screened_path: str | Path,
    gold_path: str | Path,
    *,
    runtime_seconds: float | None = None,
    retry_count: int | None = None,
    diagnostics_path: str | Path | None = None,
) -> dict[str, Any]:
    screened_rows = pd.read_csv(screened_path).to_dict(orient="records")
    gold_rows = pd.read_csv(gold_path).to_dict(orient="records")
    screened = {_source_key(row.get("Source_Row_Index")): row for row in screened_rows}
    gold = {_source_key(row.get("Source_Row_Index")): row for row in gold_rows}
    shared = [key for key in gold if key in screened]
    confusion = {
        "KEEP": {decision: 0 for decision in sorted(VALID_DECISIONS)},
        "REJECT": {decision: 0 for decision in sorted(VALID_DECISIONS)},
    }
    false_keeps: list[dict[str, Any]] = []
    positive_rejects: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    critic_count = 0
    processing_seconds = 0.0
    for key in shared:
        result = screened[key]
        label = str(gold[key].get("Gold_Decision") or "").strip().upper()
        prediction = str(result.get("Decision") or "").strip().upper()
        if label in confusion and prediction in VALID_DECISIONS:
            confusion[label][prediction] += 1
        structured = (
            prediction in VALID_DECISIONS
            and str(result.get("Validation_Status")) == "validated"
            and all(_json_list(result.get(column)) is not None for column in STRUCTURED_COLUMNS)
        )
        if not structured:
            invalid_rows.append({
                "source_row_index": key, "title": result.get("Title", ""),
                "decision": prediction, "validation_status": result.get("Validation_Status", ""),
            })
        if label == "REJECT" and prediction == "KEEP":
            false_keeps.append({
                "source_row_index": key,
                "title": result.get("Title", gold[key].get("Title", "")),
                "reason": result.get("Reason", ""),
                "confidence": result.get("Confidence"),
                "evidence": _json_list(result.get("Evidence_JSON")) or [],
            })
        if label == "KEEP" and prediction == "REJECT":
            positive_rejects.append({
                "source_row_index": key,
                "title": result.get("Title", gold[key].get("Title", "")),
                "reason": result.get("Reason", ""),
            })
        critic_count += int(_truthy(result.get("Escalated")))
        try:
            processing_seconds += float(result.get("Processing_Seconds") or 0)
        except (TypeError, ValueError):
            pass
    complete = len(shared) == len(gold) == len(screened_rows)
    diagnostics = load_run_diagnostics(diagnostics_path) if diagnostics_path else {}
    report = {
        "matched_rows": len(shared),
        "expected_rows": len(gold),
        "screened_rows": len(screened_rows),
        "confusion": confusion,
        "false_keep_count": len(false_keeps),
        "positive_reject_count": len(positive_rejects),
        "invalid_structured_count": len(invalid_rows),
        "critic_count": critic_count,
        "retry_count": retry_count if retry_count is not None else diagnostics.get("retry_count"),
        "runtime_seconds": runtime_seconds if runtime_seconds is not None else diagnostics.get("runtime_seconds"),
        "timeout_fallback_count": diagnostics.get("timeout_fallback_count"),
        "attempt_count": diagnostics.get("attempt_count"),
        "detector_outcomes": diagnostics.get("detector_outcomes", {}),
        "summed_processing_seconds": round(processing_seconds, 4),
        "false_keeps": false_keeps,
        "positive_rejects": positive_rejects,
        "invalid_rows": invalid_rows,
        "gates": {
            "complete": complete,
            "all_structurally_valid": not invalid_rows and complete,
            "no_clear_negative_kept": not false_keeps and complete,
            "no_clear_positive_rejected": not positive_rejects and complete,
        },
    }
    report["passed"] = all(report["gates"].values())
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and score a blinded Gemini Web mixed control.")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Create screening and gold-sidecar CSV files.")
    build.add_argument("--positive", required=True)
    build.add_argument("--negative", required=True)
    build.add_argument("--papers-out", required=True)
    build.add_argument("--gold-out", required=True)
    build.add_argument("--per-group", type=int, default=20)
    score = commands.add_parser("score", help="Score an exported Gemini Web screening CSV.")
    score.add_argument("--screened", required=True)
    score.add_argument("--gold", required=True)
    score.add_argument("--runtime-seconds", type=float)
    score.add_argument("--retry-count", type=int)
    score.add_argument("--diagnostics")
    score.add_argument("--report-out")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build":
        report = build_mixed_control(
            args.positive, args.negative, args.papers_out, args.gold_out,
            per_group=args.per_group,
        )
    else:
        report = score_mixed_control(
            args.screened, args.gold,
            runtime_seconds=args.runtime_seconds, retry_count=args.retry_count,
            diagnostics_path=args.diagnostics,
        )
        if args.report_out:
            target = Path(args.report_out)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if args.command == "build" or report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
