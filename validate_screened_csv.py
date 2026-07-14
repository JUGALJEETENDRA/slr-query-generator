from __future__ import annotations

import argparse
import json

import pandas as pd


def validate_screened_csv(path: str) -> dict:
    frame = pd.read_csv(path)
    required = {
        "Title", "Abstract", "Decision", "Reason", "Confidence", "Protocol_ID",
        "Evidence_JSON", "Criteria_JSON", "Validation_Status", "Schema_Version",
    }
    missing = sorted(required - set(frame.columns))
    invalid_decisions = int((~frame.get("Decision", pd.Series(dtype=str)).isin(["KEEP", "MAYBE", "REJECT"])).sum())
    invalid_definitive = 0
    invalid_quotes = 0
    evidence_count = 0
    for _, row in frame.iterrows():
        decision = str(row.get("Decision", ""))
        if decision in {"KEEP", "REJECT"} and row.get("Validation_Status") != "validated":
            invalid_definitive += 1
        try:
            evidence = json.loads(row.get("Evidence_JSON") or "[]")
        except (json.JSONDecodeError, TypeError):
            evidence = []
            invalid_quotes += 1
        for span in evidence:
            evidence_count += 1
            source = str(row.get("Title", "")) if span.get("source") == "title" else str(row.get("Abstract", ""))
            invalid_quotes += int(str(span.get("quote", "")) not in source)
    return {
        "rows": len(frame), "missing_columns": missing,
        "invalid_decisions": invalid_decisions,
        "invalid_definitive_decisions": invalid_definitive,
        "evidence_count": evidence_count,
        "invalid_evidence_quotes": invalid_quotes,
        "valid": not missing and not invalid_decisions and not invalid_definitive and not invalid_quotes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    print(json.dumps(validate_screened_csv(args.path), indent=2))
