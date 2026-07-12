from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("old_csv")
    parser.add_argument("new_csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    old = _normalize_columns(pd.read_csv(args.old_csv).fillna(""))
    new = _normalize_columns(pd.read_csv(args.new_csv).fillna(""))
    key = "title" if "title" in old.columns and "title" in new.columns else "row_index"
    merged = old.merge(new, on=key, suffixes=("_old", "_new"), how="outer", indicator=True)

    if len(old) != len(new) or (merged["_merge"] != "both").any():
        print("invalid_or_misaligned_inputs=true")
        return 2
    decision_flips = merged[
        merged.get("final_decision_old", "").astype(str)
        != merged.get("final_decision_new", "").astype(str)
    ]
    keep_lost = decision_flips[
        (decision_flips.get("final_decision_old", "") == "KEEP")
        & (decision_flips.get("final_decision_new", "") != "KEEP")
    ]
    new_keep = decision_flips[
        (decision_flips.get("final_decision_old", "") != "KEEP")
        & (decision_flips.get("final_decision_new", "") == "KEEP")
    ]
    reject_gained = decision_flips[
        (decision_flips.get("final_decision_old", "") != "REJECT")
        & (decision_flips.get("final_decision_new", "") == "REJECT")
    ]
    relation_changes = _changed(merged, "directional_relation")
    workflow_changes = _changed(merged, "workflow_use")
    external_changes = _changed(merged, "external_domain")
    adjudication_changes = _changed(merged, "final_adjudication_action")

    workflow_reject = new[_truthy(new.get("workflow_use", "")) & new.get("final_decision", "").astype(str).eq("REJECT")]
    external_keep = new[_truthy(new.get("external_domain", "")) & new.get("final_decision", "").astype(str).eq("KEEP")]
    suspicious_new_keeps = new_keep[
        _truthy(new_keep.get("external_domain_new", ""))
        | ~_truthy(new_keep.get("workflow_use_new", ""))
    ]
    output = StringIO()
    with redirect_stdout(output):
        print(f"rows_old={len(old)}")
        print(f"rows_new={len(new)}")
        print(f"decision_flips={len(decision_flips)}")
        print(f"keep_lost={len(keep_lost)}")
        print(f"new_keep={len(new_keep)}")
        print(f"reject_gained={len(reject_gained)}")
        print(f"workflow_flag_changes={len(workflow_changes)}")
        print(f"external_flag_changes={len(external_changes)}")
        print(f"final_adjudication_action_changes={len(adjudication_changes)}")
        print(f"suspicious_new_keeps={len(suspicious_new_keeps)}")
        print(f"workflow_reject_safety_violations={len(workflow_reject)}")
        print(f"external_keep_safety_violations={len(external_keep)}")
        _print_rows("decision_flips", decision_flips)
        _print_rows("keep_lost", keep_lost)
        _print_rows("new_keep", new_keep)
        _print_rows("reject_gained", reject_gained)
        _print_rows("suspicious_new_keeps", suspicious_new_keeps)
        _print_rows("workflow_reject_safety_violations", workflow_reject)
        _print_rows("external_keep_safety_violations", external_keep)
        _print_change_table(decision_flips)
    report = output.getvalue()
    print(report, end="")
    report_path = Path(args.output) if args.output else Path("outputs/benchmark_compare") / f"{datetime.now():%Y%m%d_%H%M%S}_current_vs_fast.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"report_path={report_path}")
    return 1 if len(workflow_reject) or len(external_keep) else 0


def _normalize_columns(df):
    aliases = {
        "Title": "title",
        "Decision": "final_decision",
        "stage1_directional_relation": "directional_relation",
        "stage1_directional_uses_ai_for_review_workflow": "workflow_use",
        "stage1_directional_is_review_about_ai_external_domain": "external_domain",
        "stage1_model_fusion_action": "model_fusion_action",
        "stage1_llm_route": "llm_route",
        "Reason": "final_reason",
    }
    renamed = df.rename(columns={source: target for source, target in aliases.items() if source in df.columns})
    if "row_index" not in renamed.columns:
        renamed = renamed.copy()
        renamed["row_index"] = range(len(renamed))
    return renamed


def _changed(df, column):
    old_col = f"{column}_old"
    new_col = f"{column}_new"
    if old_col not in df.columns or new_col not in df.columns:
        return df.iloc[0:0]
    return df[df[old_col].astype(str) != df[new_col].astype(str)]


def _truthy(values):
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _print_rows(label, rows):
    if rows.empty:
        return
    print(f"[{label}]")
    for _, row in rows.head(20).iterrows():
        title = row.get("title") or row.get("title_old") or row.get("title_new") or row.get("row_index")
        old_decision = row.get("final_decision_old", "")
        new_decision = row.get("final_decision_new", "")
        print(f"- {old_decision} -> {new_decision}: {title}")


def _print_change_table(rows):
    if rows.empty:
        return
    tracked = (
        "directional_relation",
        "workflow_use",
        "external_domain",
        "final_adjudication_action",
        "workflow_quote_valid",
        "external_domain_quote_valid",
        "model_fusion_action",
    )
    print("[change_table]")
    print("title | old_decision | new_decision | changed_field | likely_cause")
    for _, row in rows.head(40).iterrows():
        title = row.get("title") or row.get("title_old") or row.get("title_new") or row.get("row_index")
        old_decision = row.get("final_decision_old", "")
        new_decision = row.get("final_decision_new", "")
        changed = []
        for field in tracked:
            old_value = str(row.get(f"{field}_old", ""))
            new_value = str(row.get(f"{field}_new", ""))
            if old_value != new_value:
                changed.append(f"{field}: {old_value}->{new_value}")
        print(
            f"{title} | {old_decision} | {new_decision} | "
            f"{'; '.join(changed) if changed else 'decision_only'} | {_likely_cause(row)}"
        )


def _likely_cause(row):
    workflow_old = str(row.get("workflow_use_old", "")).lower() in {"true", "1", "yes"}
    workflow_new = str(row.get("workflow_use_new", "")).lower() in {"true", "1", "yes"}
    external_old = str(row.get("external_domain_old", "")).lower() in {"true", "1", "yes"}
    external_new = str(row.get("external_domain_new", "")).lower() in {"true", "1", "yes"}
    action_new = str(row.get("final_adjudication_action_new", "") or row.get("model_fusion_action_new", ""))
    if not workflow_old and workflow_new:
        return "workflow relation recovered"
    if workflow_old and not workflow_new:
        return "workflow relation lost"
    if external_old and not external_new:
        return "external-domain conflict cleared"
    if not external_old and external_new:
        return "external-domain conflict introduced"
    if "recall_guard" in action_new:
        return "recall guard changed final decision"
    if "promote" in action_new or "confirm_workflow" in action_new:
        return "workflow promotion/confirmation changed"
    return "inspect row diagnostics"


if __name__ == "__main__":
    raise SystemExit(main())
