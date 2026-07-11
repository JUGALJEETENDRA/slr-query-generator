from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--rq-type", default="")
    parser.add_argument("--write-report", default="")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    report = build_report(df, rq_type=args.rq_type)
    print(report)
    if args.write_report:
        Path(args.write_report).write_text(report + "\n", encoding="utf-8")
    return 0


def build_report(df: pd.DataFrame, rq_type: str = "") -> str:
    decision_col = "Decision" if "Decision" in df.columns else "stage1_decision"
    lines: list[str] = []
    _section(lines, "basic")
    lines.append(f"rows={len(df)}")
    _counts(lines, df, decision_col, "decision_counts")

    _section(lines, "runtime")
    for col in (
        "model_config_enable_model_judges",
        "model_config_model_judge_mode",
        "model_config_enable_hf_model_loading",
        "stage1_model_judges_enabled",
        "stage1_model_judge_mode",
        "stage1_model_judge_runtime_source",
        "stage1_model_profile",
        "stage1_model_real_models_loaded",
        "stage1_llm_directional_cache_source",
    ):
        _counts(lines, df, col, f"{col}_counts")

    _section(lines, "fusion")
    for col in (
        "stage1_model_fusion_action",
        "stage2_model_fusion_action",
        "final_adjudication_action",
        "final_relation",
        "task_object_type",
    ):
        _counts(lines, df, col, f"{col}_counts")

    _section(lines, "directional")
    stage1_workflow = _bool_col(df, "stage1_directional_uses_ai_for_review_workflow")
    stage1_external = _bool_col(df, "stage1_directional_is_review_about_ai_external_domain")
    stage2_workflow = _bool_col(df, "stage2_directional_uses_ai_for_review_workflow")
    stage2_external = _bool_col(df, "stage2_directional_is_review_about_ai_external_domain")
    final_workflow = _bool_col(df, "final_workflow_use")
    final_external = _bool_col(df, "final_external_domain")
    confidence = _num_col(df, "stage1_directional_confidence")
    final_confidence = _num_col(df, "final_relation_confidence")
    lines.append(f"stage1_workflow_use_true={int(stage1_workflow.sum())}")
    lines.append(f"stage1_external_domain_true={int(stage1_external.sum())}")
    lines.append(f"stage2_workflow_use_true={int(stage2_workflow.sum())}")
    lines.append(f"stage2_external_domain_true={int(stage2_external.sum())}")
    lines.append(f"final_workflow_use_true={int(final_workflow.sum())}")
    lines.append(f"final_external_domain_true={int(final_external.sum())}")
    lines.append(f"stage1_strong_directional={int((confidence >= 0.70).sum())}")
    lines.append(f"final_strong_directional={int((final_confidence >= 0.70).sum())}")

    _section(lines, "quality_checks")
    decision = df[decision_col].fillna("").astype(str).str.upper()
    fusion = _str_col(df, "stage1_model_fusion_action")
    adjudication = _str_col(df, "final_adjudication_action")
    warning = _str_col(df, "relation_validation_warning")
    suspicious = {
        "parse_error_rows": decision.eq("PARSE_ERROR"),
        "reject_with_validated_workflow": decision.eq("REJECT") & final_workflow & (final_confidence >= 0.70),
        "keep_with_validated_external_domain": decision.eq("KEEP") & final_external & (final_confidence >= 0.70),
        "maybe_with_strong_validated_workflow": decision.eq("MAYBE") & final_workflow & (final_confidence >= 0.80),
        "preserve_despite_strong_stage1_directional": fusion.eq("preserve") & (confidence >= 0.70),
        "adjudication_warning_rows": warning.ne(""),
        "demoted_by_final_adjudicator": adjudication.str.startswith("demote"),
        "promoted_or_rescued_by_final_adjudicator": adjudication.str.startswith("promote") | adjudication.str.startswith("rescue"),
    }
    if rq_type:
        lines.append(f"rq_type={rq_type}")
    for name, mask in suspicious.items():
        _suspicious(lines, df, name, mask)

    _section(lines, "stage_conflicts")
    _suspicious(lines, df, "stage1_workflow_stage2_external", stage1_workflow & stage2_external)
    _suspicious(lines, df, "stage1_external_stage2_workflow", stage1_external & stage2_workflow)
    _suspicious(lines, df, "final_relation_warning_keep", decision.eq("KEEP") & warning.ne(""))
    return "\n".join(lines)


def _section(lines: list[str], name: str) -> None:
    if lines:
        lines.append("")
    lines.append(f"[{name}]")


def _counts(lines: list[str], df: pd.DataFrame, col: str, label: str) -> None:
    if col not in df.columns:
        lines.append(f"{label}=<missing>")
        return
    lines.append(f"{label}:")
    counts = df[col].fillna("").astype(str).value_counts(dropna=False)
    if counts.empty:
        lines.append("  <empty>")
        return
    for value, count in counts.items():
        lines.append(f"  {value or '<blank>'}: {int(count)}")


def _suspicious(lines: list[str], df: pd.DataFrame, name: str, mask: pd.Series) -> None:
    count = int(mask.sum())
    lines.append(f"{name}={count}")
    if count and "Title" in df.columns:
        for title in df.loc[mask, "Title"].head(12):
            lines.append(f"  - {title}")


def _bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df[col].astype(str).str.lower().isin({"true", "1", "yes", "y", "on"})


def _num_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _str_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df[col].fillna("").astype(str)


if __name__ == "__main__":
    raise SystemExit(main())
