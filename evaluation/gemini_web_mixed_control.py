from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


VALID_DECISIONS = {"KEEP", "MAYBE", "REJECT"}
ORDERED_DECISIONS = ("KEEP", "MAYBE", "REJECT")
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
        label: {decision: 0 for decision in ORDERED_DECISIONS}
        for label in ORDERED_DECISIONS
    }
    control_groups: dict[str, dict[str, Any]] = {}
    false_keeps: list[dict[str, Any]] = []
    positive_rejects: list[dict[str, Any]] = []
    unresolved_near_misses: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    unverified_definitive_rows: list[dict[str, Any]] = []
    non_substantive_maybe_rows: list[dict[str, Any]] = []
    scope_support_counts = {
        "SUBSTANTIVE": 0, "INCIDENTAL": 0, "INSUFFICIENT": 0, "MISSING": 0,
    }
    critic_count = 0
    timeout_fallback_row_count = 0
    processing_seconds = 0.0
    for key in shared:
        result = screened[key]
        label = str(gold[key].get("Gold_Decision") or "").strip().upper()
        prediction = str(result.get("Decision") or "").strip().upper()
        group = str(gold[key].get("Control_Group") or "unclassified").strip() or "unclassified"
        group_report = control_groups.setdefault(group, {
            "expected_rows": 0,
            "predictions": {decision: 0 for decision in ORDERED_DECISIONS},
        })
        group_report["expected_rows"] += 1
        if prediction in VALID_DECISIONS:
            group_report["predictions"][prediction] += 1
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
        criteria = _json_list(result.get("Criteria_JSON")) or []
        non_substantive = []
        for criterion in criteria:
            if not isinstance(criterion, dict):
                continue
            support = str(criterion.get("scope_support") or "MISSING").strip().upper()
            if support not in scope_support_counts:
                support = "MISSING"
            scope_support_counts[support] += 1
            if support in {"INCIDENTAL", "INSUFFICIENT"}:
                non_substantive.append({
                    "criterion_id": str(criterion.get("criterion_id") or ""),
                    "scope_support": support,
                    "verdict": str(criterion.get("verdict") or ""),
                })
        if prediction == "MAYBE" and non_substantive:
            non_substantive_maybe_rows.append({
                "source_row_index": key,
                "title": str(result.get("Title") or gold[key].get("Title", "")),
                "criteria": non_substantive,
                "reason": str(result.get("Reason") or ""),
            })
        raw_route = result.get("Critic_Route")
        raw_status = result.get("Verification_Status")
        critic_route = "" if pd.isna(raw_route) else str(raw_route or "").strip()
        verification_status = (
            "not_required" if pd.isna(raw_status)
            else str(raw_status or "not_required").strip()
        )
        if prediction in {"KEEP", "REJECT"} and critic_route and verification_status != "agreed":
            unverified_definitive_rows.append({
                "source_row_index": key,
                "title": str(result.get("Title") or ""),
                "decision": prediction,
                "critic_route": critic_route,
                "verification_status": verification_status,
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
        if label == "REJECT" and prediction == "MAYBE" and "near_miss" in group.casefold():
            unresolved_near_misses.append({
                "source_row_index": key,
                "title": result.get("Title", gold[key].get("Title", "")),
                "control_group": group,
                "reason": result.get("Reason", ""),
            })
        critic_count += int(_truthy(result.get("Escalated")))
        timeout_fallback_row_count += int(
            str(result.get("Failure_Class") or "").strip() == "transport_timeout"
        )
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
        "control_groups": control_groups,
        "false_keep_count": len(false_keeps),
        "positive_reject_count": len(positive_rejects),
        "unresolved_near_miss_count": len(unresolved_near_misses),
        "invalid_structured_count": len(invalid_rows),
        "unverified_definitive_count": len(unverified_definitive_rows),
        "scope_support_counts": scope_support_counts,
        "non_substantive_maybe_count": len(non_substantive_maybe_rows),
        "critic_count": critic_count,
        "retry_count": retry_count if retry_count is not None else diagnostics.get("retry_count"),
        "runtime_seconds": runtime_seconds if runtime_seconds is not None else diagnostics.get("runtime_seconds"),
        "timeout_fallback_count": diagnostics.get("timeout_fallback_count"),
        "timeout_fallback_row_count": timeout_fallback_row_count,
        "attempt_count": diagnostics.get("attempt_count"),
        "detector_outcomes": diagnostics.get("detector_outcomes", {}),
        "recovery_actions": diagnostics.get("recovery_actions", {}),
        "critic_route_counts": diagnostics.get("critic_route_counts", {}),
        "verification_outcomes": diagnostics.get("verification_outcomes", {}),
        "verified_reject_count": diagnostics.get("verified_reject_count"),
        "verification_fallback_count": diagnostics.get("verification_fallback_count"),
        "protocol_cache_version": diagnostics.get("protocol_cache_version"),
        "clean_chat_rotations": diagnostics.get("clean_chat_rotations"),
        "summed_processing_seconds": round(processing_seconds, 4),
        "false_keeps": false_keeps,
        "positive_rejects": positive_rejects,
        "unresolved_near_misses": unresolved_near_misses,
        "invalid_rows": invalid_rows,
        "unverified_definitive_rows": unverified_definitive_rows,
        "non_substantive_maybe_rows": non_substantive_maybe_rows,
        "gates": {
            "complete": complete,
            "all_structurally_valid": not invalid_rows and complete,
            "all_definitive_decisions_verified": not unverified_definitive_rows and complete,
            "no_clear_negative_kept": not false_keeps and complete,
            "no_clear_positive_rejected": not positive_rejects and complete,
        },
    }
    report["passed"] = all(report["gates"].values())
    return report


def compare_screening_runs(
    screened_a_path: str | Path,
    screened_b_path: str | Path,
    *,
    gold_path: str | Path | None = None,
) -> dict[str, Any]:
    rows_a = pd.read_csv(screened_a_path).to_dict(orient="records")
    rows_b = pd.read_csv(screened_b_path).to_dict(orient="records")
    by_a = {_source_key(row.get("Source_Row_Index")): row for row in rows_a}
    by_b = {_source_key(row.get("Source_Row_Index")): row for row in rows_b}
    gold = {}
    if gold_path:
        gold = {
            _source_key(row.get("Source_Row_Index")): row
            for row in pd.read_csv(gold_path).to_dict(orient="records")
        }
    shared = sorted(
        set(by_a) & set(by_b),
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    transitions = {
        first: {second: 0 for second in ORDERED_DECISIONS}
        for first in ORDERED_DECISIONS
    }
    disagreements: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []
    repeated_maybes: list[dict[str, Any]] = []
    transport_affected: list[dict[str, Any]] = []
    semantic_exact = 0
    semantic_shared = 0
    safety_contradictions: list[dict[str, Any]] = []
    exact = 0
    for key in shared:
        decision_a = str(by_a[key].get("Decision") or "").strip().upper()
        decision_b = str(by_b[key].get("Decision") or "").strip().upper()
        if decision_a in VALID_DECISIONS and decision_b in VALID_DECISIONS:
            transitions[decision_a][decision_b] += 1
        title = by_a[key].get("Title", by_b[key].get("Title", ""))
        item = {
            "source_row_index": key,
            "title": title,
            "decision_a": decision_a,
            "decision_b": decision_b,
            "gold_decision": str(gold.get(key, {}).get("Gold_Decision") or "").upper() or None,
            "control_group": gold.get(key, {}).get("Control_Group"),
        }
        if decision_a == decision_b:
            exact += 1
        else:
            disagreements.append(item)
        if {decision_a, decision_b} == {"KEEP", "REJECT"}:
            contradictions.append(item)
        if decision_a == decision_b == "MAYBE":
            repeated_maybes.append(item)
        transport_failure = any(
            str(row.get("Failure_Class") or "").strip() == "transport_timeout"
            for row in (by_a[key], by_b[key])
        )
        if transport_failure:
            transport_affected.append(item)
        else:
            semantic_shared += 1
            semantic_exact += int(decision_a == decision_b)
        if (
            {decision_a, decision_b} == {"KEEP", "REJECT"}
            and item["gold_decision"] in {"KEEP", "REJECT"}
        ):
            safety_contradictions.append(item)
    complete = len(shared) == len(rows_a) == len(rows_b)
    return {
        "shared_rows": len(shared),
        "run_a_rows": len(rows_a),
        "run_b_rows": len(rows_b),
        "complete": complete,
        "exact_agreement_count": exact,
        "exact_agreement_rate": round(exact / len(shared), 4) if shared else 0.0,
        "transport_affected_count": len(transport_affected),
        "semantic_shared_rows": semantic_shared,
        "semantic_exact_agreement_count": semantic_exact,
        "semantic_exact_agreement_rate": (
            round(semantic_exact / semantic_shared, 4) if semantic_shared else 0.0
        ),
        "transition_matrix": transitions,
        "disagreement_count": len(disagreements),
        "keep_reject_contradiction_count": len(contradictions),
        "safety_contradiction_count": len(safety_contradictions),
        "repeated_maybe_count": len(repeated_maybes),
        "disagreements": disagreements,
        "keep_reject_contradictions": contradictions,
        "safety_contradictions": safety_contradictions,
        "transport_affected_rows": transport_affected,
        "repeated_maybes": repeated_maybes,
        "passed": complete and not safety_contradictions,
    }


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
    compare = commands.add_parser("compare", help="Compare two repeated screening runs.")
    compare.add_argument("--screened-a", required=True)
    compare.add_argument("--screened-b", required=True)
    compare.add_argument("--gold")
    compare.add_argument("--report-out")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build":
        report = build_mixed_control(
            args.positive, args.negative, args.papers_out, args.gold_out,
            per_group=args.per_group,
        )
    elif args.command == "score":
        report = score_mixed_control(
            args.screened, args.gold,
            runtime_seconds=args.runtime_seconds, retry_count=args.retry_count,
            diagnostics_path=args.diagnostics,
        )
        if args.report_out:
            target = Path(args.report_out)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        report = compare_screening_runs(
            args.screened_a, args.screened_b, gold_path=args.gold,
        )
        if args.report_out:
            target = Path(args.report_out)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if args.command == "build" or report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
