from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from bulk_screen import screen_csv


SLR_CONFIG = "benchmark_slr_automation_first100.json"
BLOCKCHAIN_CONFIG = "benchmark_blockchain_first100.json"
MEDICAL_CONFIG = "benchmark_heart_disease_first100.json"
FAILURE_RECALL_ROWS = "outputs/benchmark_rows/slr_current_final.csv"

WORKFLOW_RECALL_TERMS = (
    "systematic review automation",
    "systematic literature review automation",
    "literature review automation",
    "automating systematic reviews",
    "automating systematic literature reviews",
    "automate systematic reviews",
    "automate systematic literature reviews",
    "automating literature reviews",
    "automate literature reviews",
    "automate and facilitate the review process",
    "facilitate the review process",
    "automating the review process",
    "automate the review process",
    "review process automation",
    "title/abstract screening",
    "title abstract screening",
    "citation screening",
    "study selection",
    "selection of studies",
    "literature search",
    "search strategy",
    "review stages",
    "slr methodology",
    "llm-assisted methodology",
    "accelerating systematic reviews",
    "accelerate systematic reviews",
    "accelerating systematic literature reviews",
    "accelerate systematic literature reviews",
    "data extraction from studies",
    "screening sensitivity",
    "screening recall",
    "reviewer workload reduction",
    "screening phase",
    "identification of relevant articles",
)

SETS = {
    "obvious_workflow": [
        "Assessing Probability Rating Performance for Large Language Models in Systematic Literature Review Automation",
        "Toward the Conduction of Automated Systematic Literature Reviews: The Role of Large Language Models",
        "Cal-X: Enhancing Systematic Review Screening with LLMs and Next-Token Likelihood Calibration",
        "Large Language Models for Title/Abstract Screening in Systematic Literature Reviews",
        "Screening Automation for Systematic Reviews",
        "Implementation of a research assistant for literature review with Large Language Models",
        "SESR-Eval: Dataset for Evaluating LLMs in the Title-Abstract Screening of Systematic Reviews",
        "Efficient Citation Screening by Weak Classifier Ensemble",
        "Feature selection analysis of a scalable citation screening algorithm",
        "Enhancing Systematic Literature Reviews: Evaluating the Performance of LLM-Based Tools Across Key Systematic Literature Review Stages",
    ],
    "external_domain": [
        "Application of Large Language Models in Cybersecurity: A Systematic Literature Review",
        "Applications of Large Language Models in Breast Cancer Diagnosis: A Systematic Review",
        "Artificial Intelligence Methods in Software Refactoring: A Systematic Literature Review",
        "XAI in Fintech: A Systematic Literature Review",
        "Integrating Artificial Intelligence in Language Education: A Systematic Literature Review",
        "Systematic Literature Review on Application of Artificial Intelligence in Cancer Detection Using Image Processing",
        "A Systematic Literature Review of Large Language Model Applications in Industry",
        "Systematic Literature Review: Deep Learning and Machine Learning Analysis for Batik Peranakan Tionghoa Datasets",
    ],
    "borderline": [
        "Can AI be a Scholar? A Systematic Review on the Role of Generative AI in Systematic Literature Reviews",
        "Efficacy of Large Language Models for Systematic Reviews",
        "Can Machine Learning Support the Selection of Studies for Systematic Literature Review Updates?",
        "Leveraging the Potential of Generative AI to Accelerate Systematic Literature Reviews",
        "Towards Explainable AI in Agentic Retrieval-Augmented Generation: A Systematic Review",
        "Revolutionizing Literature Search: AI vs. Traditional Methods in Digital Divide Literature Screening and Reviewing",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=["all", "obvious_workflow", "external_domain", "borderline", "failure_recall", "first20", "blockchain10", "medical10"], default="all")
    parser.add_argument("--mode", choices=["current", "two_pass_fast"], default="current")
    parser.add_argument("--cache", choices=["off", "on"], default="off")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _set_env_defaults(args.mode, args.cache)
    selected = [
        "obvious_workflow",
        "external_domain",
        "borderline",
        "failure_recall",
        "first20",
        "blockchain10",
        "medical10",
    ] if args.set == "all" else [args.set]
    summaries = []
    failures = []
    for name in selected:
        summary, failed_rows = run_set(name, verbose=args.verbose)
        summaries.append(summary)
        failures.extend(failed_rows)
    failures.extend(_aggregate_quality_gate(selected))

    print("SET | rows | keep | maybe | reject | parse_error | passed | runtime")
    for row in summaries:
        print(
            f"{row['set']} | {row['rows']} | {row['keep']} | {row['maybe']} | "
            f"{row['reject']} | {row['parse_error']} | {row['passed']} | {row['runtime']:.2f}s"
        )
    if failures:
        print("[failures]")
        for failure in failures[:40]:
            print(json.dumps(failure, ensure_ascii=True))
        return 2
    return 0


def run_set(name: str, verbose=False):
    started = time.perf_counter()
    config_path = BLOCKCHAIN_CONFIG if name == "blockchain10" else MEDICAL_CONFIG if name == "medical10" else SLR_CONFIG
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    df = pd.read_csv(config["dataset_path"])
    if name == "first20":
        subset = df.head(20).copy()
    elif name == "blockchain10" or name == "medical10":
        subset = df.head(10).copy()
    elif name == "failure_recall":
        subset = _failure_recall_subset(df)
    else:
        subset = _select_titles(df, SETS[name])

    output_dir = Path("outputs") / "mini_regression"
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = output_dir / f"{name}_input.csv"
    output_path = output_dir / f"{name}_screened.csv"
    subset.to_csv(subset_path, index=False)
    result = screen_csv(
        csv_path=str(subset_path),
        research_question=config["research_question"],
        output_path=str(output_path),
        two_stage_enabled=False,
    )
    rows = pd.read_csv(output_path).fillna("").to_dict("records")
    failures = _evaluate(name, rows)
    if verbose or failures:
        for row in failures:
            print(json.dumps(row, ensure_ascii=True))
    summary = {
        "set": name,
        "rows": len(rows),
        "keep": result["keep"],
        "maybe": result["maybe"],
        "reject": result["reject"],
        "parse_error": result["parse_error"],
        "passed": not failures,
        "runtime": time.perf_counter() - started,
    }
    return summary, failures


def _select_titles(df, titles):
    title_col = _title_col(df)
    used = set()
    rows = []
    normalized_titles = df[title_col].astype(str).map(_norm)
    for title in titles:
        target = _norm(title)
        mask = normalized_titles.eq(target)
        if not mask.any():
            mask = normalized_titles.str.contains(_contains_safe(target), regex=False)
        if not mask.any():
            continue
        idx = int(df[mask].index[0])
        if idx not in used:
            rows.append(df.loc[idx])
            used.add(idx)
    return pd.DataFrame(rows)


def _evaluate(name, rows):
    failures = []
    decisions = [str(row.get("Decision", "")).upper() for row in rows]
    if any(decision == "PARSE_ERROR" for decision in decisions):
        failures.extend(_fail_rows(name, rows, "PARSE_ERROR is not allowed"))
    if name == "obvious_workflow":
        failures.extend(_fail_rows(name, [r for r in rows if r.get("Decision") == "REJECT"], "workflow row rejected"))
    elif name == "external_domain":
        failures.extend(_fail_rows(name, [r for r in rows if r.get("Decision") == "KEEP"], "external-domain row kept"))
    elif name == "borderline":
        for row in rows:
            workflow = _bool(row.get("stage1_directional_uses_ai_for_review_workflow"))
            external = _bool(row.get("stage1_directional_is_review_about_ai_external_domain"))
            if row.get("Decision") == "REJECT" and workflow:
                failures.append(_failure(name, row, "workflow evidence hard rejected"))
            if row.get("Decision") == "KEEP" and external:
                failures.append(_failure(name, row, "external-domain evidence kept"))
    elif name == "failure_recall":
        keep_count = decisions.count("KEEP")
        for row in rows:
            strong_workflow = _strong_workflow_row(row)
            external = _bool(row.get("stage1_directional_is_review_about_ai_external_domain"))
            if strong_workflow and row.get("Decision") == "REJECT":
                failures.append(_failure(name, row, "strong workflow recall row rejected"))
            if external and row.get("Decision") == "KEEP":
                failures.append(_failure(name, row, "external-domain row kept"))
        if rows and keep_count == 0:
            failures.append({"set": name, "title": "<set>", "expected": "some strong workflow candidates become KEEP", "decision": str(decisions)})
    elif name == "first20":
        if decisions.count("KEEP") == len(decisions) or decisions.count("REJECT") == len(decisions):
            failures.append({"set": name, "title": "<set>", "expected": "no decision collapse", "decision": ""})
        for row in rows:
            workflow = _bool(row.get("stage1_directional_uses_ai_for_review_workflow"))
            external = _bool(row.get("stage1_directional_is_review_about_ai_external_domain"))
            if row.get("Decision") == "REJECT" and workflow:
                failures.append(_failure(name, row, "obvious workflow rejected"))
            if row.get("Decision") == "KEEP" and external:
                failures.append(_failure(name, row, "external-domain kept"))
    elif name in {"blockchain10", "medical10"}:
        if decisions.count("KEEP") < 8:
            failures.append({"set": name, "title": "<set>", "expected": "mostly KEEP", "decision": str(decisions)})
    return failures


def _aggregate_quality_gate(selected):
    if not {"obvious_workflow", "borderline", "failure_recall"}.issubset(set(selected)):
        return []
    failures = []
    rows = []
    for name in ("obvious_workflow", "borderline", "failure_recall"):
        path = Path("outputs") / "mini_regression" / f"{name}_screened.csv"
        if path.exists():
            rows.extend(pd.read_csv(path).fillna("").to_dict("records"))

    strong_workflow = [row for row in rows if _strong_workflow_row(row)]
    workflow_rejects = [row for row in strong_workflow if str(row.get("Decision", "")).upper() == "REJECT"]
    external_keeps = [
        row for row in rows
        if _bool(row.get("stage1_directional_is_review_about_ai_external_domain"))
        and str(row.get("Decision", "")).upper() == "KEEP"
    ]
    failures.extend(_fail_rows("aggregate", workflow_rejects, "strong workflow rows must not be REJECT"))
    failures.extend(_fail_rows("aggregate", external_keeps, "external-domain rows must not be KEEP"))
    return failures


def _failure_recall_subset(df):
    row_path = Path(FAILURE_RECALL_ROWS)
    if not row_path.exists():
        return _select_titles(df, SETS["borderline"])

    failed = pd.read_csv(row_path).fillna("")
    title_col = _title_col(df)
    abstract_col = _abstract_col(df)
    source_text_by_title = {
        _norm(row.get(title_col)): f"{row.get(title_col, '')} {row.get(abstract_col, '')}"
        for _, row in df.iterrows()
    }
    selected_titles = []
    for _, row in failed.iterrows():
        title = str(row.get("title") or "")
        decision = str(row.get("final_decision") or "").upper()
        if decision == "KEEP":
            continue
        text = " ".join(
            str(row.get(name) or "")
            for name in (
                "title",
                "directional_relation",
                "model_fusion_action",
                "final_adjudication_action",
                "final_reason",
            )
        ).lower()
        text = f"{text} {source_text_by_title.get(_norm(title), '')}".lower()
        workflowish = _has_any(text, WORKFLOW_RECALL_TERMS)
        action = str(row.get("final_adjudication_action") or "").lower()
        workflow_direction = (
            _bool(row.get("workflow_use"))
            or "recall_guard" in action
            or "rescue" in action
        )
        if workflowish or workflow_direction:
            selected_titles.append(title)

    if not selected_titles:
        selected_titles = SETS["borderline"]
    subset = _select_titles(df, selected_titles)
    if len(subset) < 6:
        supplement = _select_titles(df, SETS["borderline"])
        subset = pd.concat([subset, supplement], ignore_index=True)
        subset = subset.drop_duplicates(subset=[title_col], keep="first")
    return subset.head(20).copy()


def _strong_workflow_row(row):
    text = " ".join(
        str(row.get(name) or "")
        for name in (
            "Title",
            "Abstract",
            "stage1_positive_workflow_candidate_terms",
            "stage1_workflow_evidence_quote",
            "stage1_llm_workflow_tasks_detected",
            "stage1_directional_reason",
            "stage1_llm_judge_reason",
            "stage1_paper_target_problem_or_task",
            "stage1_paper_application_context",
        )
    ).lower()
    return _has_any(text, WORKFLOW_RECALL_TERMS)


def _fail_rows(name, rows, expected):
    return [_failure(name, row, expected) for row in rows]


def _failure(name, row, expected):
    return {
        "set": name,
        "title": row.get("Title", ""),
        "decision": row.get("Decision", ""),
        "expected": expected,
        "directional_relation": row.get("stage1_directional_relation", ""),
        "workflow_use": row.get("stage1_directional_uses_ai_for_review_workflow", ""),
        "external_domain": row.get("stage1_directional_is_review_about_ai_external_domain", ""),
        "final_adjudication_action": row.get("final_adjudication_action", ""),
        "model_fusion_action": row.get("stage1_model_fusion_action", ""),
        "llm_route": row.get("stage1_llm_route", ""),
        "cache_source": row.get("stage1_llm_directional_cache_source", ""),
        "reason": str(row.get("Reason", ""))[:400],
    }


def _set_env_defaults(mode, cache):
    os.environ.setdefault("MODEL_JUDGE_MODE", "balanced")
    os.environ.setdefault("MODEL_JUDGE_PROFILE", "light")
    os.environ.setdefault("ENABLE_MODEL_JUDGES", "true")
    os.environ.setdefault("ENABLE_HF_MODEL_LOADING", "false")
    os.environ.setdefault("ENABLE_HF_MODEL_DOWNLOAD", "false")
    os.environ.setdefault("ENABLE_LLM_JUDGE", "true")
    os.environ["SCREENING_PIPELINE_MODE"] = mode
    os.environ["ENABLE_SEMANTIC_FRAME_CACHE"] = "true" if cache == "on" else "false"
    if mode == "current":
        os.environ.setdefault("ENABLE_AGGRESSIVE_LLM_GATING", "false")
        os.environ.setdefault("ENABLE_BATCH_LLM_JUDGE", "false")
    else:
        os.environ.setdefault("ENABLE_AGGRESSIVE_LLM_GATING", "true")


def _title_col(df):
    for col in ("Title", "Document Title", "title"):
        if col in df.columns:
            return col
    return df.columns[0]


def _abstract_col(df):
    for col in ("Abstract", "abstract", "Abstract Note", "Description"):
        if col in df.columns:
            return col
    return df.columns[1] if len(df.columns) > 1 else df.columns[0]


def _norm(value):
    return " ".join(str(value or "").lower().split())


def _contains_safe(value):
    return value[:80]


def _bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _has_any(text, terms):
    normalized = str(text or "").lower()
    return any(term in normalized for term in terms)


if __name__ == "__main__":
    raise SystemExit(main())
