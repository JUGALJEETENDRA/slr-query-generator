from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


POSITIVE_LABELS = {"KEEP", "INCLUDE", "RELEVANT", "1", "TRUE"}


def _source_id(row: dict[str, Any]) -> str:
    return str(row.get("Source_Row_Index", row.get("source_row_index", row.get("id", row.get("Title", "")))))


def _gold_positive(value: Any) -> bool:
    return str(value or "").strip().upper() in POSITIVE_LABELS


def evaluate_rows(screened_rows: list[dict], gold_rows: list[dict]) -> dict[str, Any]:
    screened = {_source_id(row): row for row in screened_rows}
    gold = {_source_id(row): row for row in gold_rows}
    shared = sorted(set(screened) & set(gold))
    tp_retrieved = relevant = definitive_keeps = correct_keeps = false_rejects = 0
    exact_evidence = total_evidence = invalid_definitive = validated = repairs = 0
    for key in shared:
        predicted = str(screened[key].get("Decision", "")).upper()
        label = gold[key].get("Gold_Decision", gold[key].get("label", gold[key].get("relevant")))
        is_relevant = _gold_positive(label)
        relevant += int(is_relevant)
        tp_retrieved += int(is_relevant and predicted in {"KEEP", "MAYBE"})
        false_rejects += int(is_relevant and predicted == "REJECT")
        definitive_keeps += int(predicted == "KEEP")
        correct_keeps += int(predicted == "KEEP" and is_relevant)
        invalid_definitive += int(
            predicted in {"KEEP", "REJECT"} and screened[key].get("Validation_Status") != "validated"
        )
        validated += int(screened[key].get("Validation_Status") == "validated")
        repairs += int(str(screened[key].get("Escalated", "")).lower() in {"1", "true"})
        try:
            evidence = json.loads(screened[key].get("Evidence_JSON") or "[]")
        except (json.JSONDecodeError, TypeError):
            evidence = []
        title = str(screened[key].get("Title", ""))
        abstract = str(screened[key].get("Abstract", ""))
        for span in evidence:
            total_evidence += 1
            source = title if span.get("source") == "title" else abstract
            exact_evidence += int(str(span.get("quote", "")) in source)
    recall = tp_retrieved / relevant if relevant else 0.0
    structural_rate = validated / len(shared) if shared else 0.0
    repair_rate = repairs / len(shared) if shared else 0.0
    return {
        "matched_rows": len(shared),
        "relevant_recall_keep_or_maybe": round(recall, 4),
        "false_reject_rate": round(false_rejects / relevant, 4) if relevant else 0.0,
        "definitive_keep_precision": round(correct_keeps / definitive_keeps, 4) if definitive_keeps else 0.0,
        "exact_evidence_rate": round(exact_evidence / total_evidence, 4) if total_evidence else 1.0,
        "invalid_definitive_count": invalid_definitive,
        "structurally_validated_rate": round(structural_rate, 4),
        "repair_call_rate": round(repair_rate, 4),
        "gates": {
            "recall_passed": recall >= 0.95,
            "false_reject_passed": (false_rejects / relevant if relevant else 0.0) <= 0.05,
            "keep_precision_passed": (correct_keeps / definitive_keeps if definitive_keeps else 0.0) >= 0.85,
            "evidence_passed": exact_evidence == total_evidence,
            "safety_passed": invalid_definitive == 0,
            "structure_passed": structural_rate >= 0.95,
            "repair_rate_passed": repair_rate <= 0.15,
        },
    }


def evaluate_files(gold_path: str, screened_path: str) -> dict[str, Any]:
    return evaluate_rows(
        pd.read_csv(screened_path).to_dict(orient="records"),
        pd.read_csv(gold_path).to_dict(orient="records"),
    )
