from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import pandas as pd


GOLD_FIELDS = {
    "Gold_Decision", "Gold_Rationale", "Challenge_Type",
}


def prepare_adversarial_benchmark(
    *,
    adjudicated_csv: str | Path,
    papers_out: str | Path,
    gold_out: str | Path,
) -> dict:
    source = pd.read_csv(adjudicated_csv, keep_default_na=False)
    required = {"Title", "Abstract", *GOLD_FIELDS}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"adversarial source is missing columns: {sorted(missing)}")
    if not 20 <= len(source) <= 30:
        raise ValueError("adversarial benchmark must contain 20 to 30 independently adjudicated papers")
    decisions = source["Gold_Decision"].astype(str).str.strip().str.upper()
    if not decisions.isin({"KEEP", "MAYBE", "REJECT"}).all():
        raise ValueError("Gold_Decision must contain only KEEP, MAYBE, or REJECT")
    if source["Gold_Rationale"].astype(str).str.strip().eq("").any():
        raise ValueError("every adversarial gold label requires an adjudication rationale")
    challenge_types = source["Challenge_Type"].astype(str).str.strip()
    if challenge_types.eq("").any() or challenge_types.nunique() < 5:
        raise ValueError("adversarial benchmark requires at least five non-empty challenge types")

    screening_columns = [column for column in source.columns if column not in GOLD_FIELDS]
    papers = source[screening_columns].copy()
    forbidden = set(papers.columns) & GOLD_FIELDS
    if forbidden:
        raise ValueError(f"gold leakage into screening fixture: {sorted(forbidden)}")
    gold = pd.DataFrame({
        "Source_Row_Index": list(range(len(source))),
        "Gold_Decision": decisions,
        "Control_Group": challenge_types,
        "Gold_Rationale": source["Gold_Rationale"].astype(str),
        "Title": source["Title"].astype(str),
    })
    papers_path = Path(papers_out)
    gold_path = Path(gold_out)
    papers_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    papers.to_csv(papers_path, index=False)
    gold.to_csv(gold_path, index=False)
    return {
        "papers": str(papers_path),
        "gold": str(gold_path),
        "paper_count": len(papers),
        "challenge_type_count": int(challenge_types.nunique()),
        "labels_blinded": True,
        "screening_sha256": sha256(papers_path.read_bytes()).hexdigest(),
        "gold_sha256": sha256(gold_path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a blinded, independently adjudicated v2.4 adversarial benchmark."
    )
    parser.add_argument("--adjudicated", required=True)
    parser.add_argument("--papers-out", required=True)
    parser.add_argument("--gold-out", required=True)
    args = parser.parse_args()
    result = prepare_adversarial_benchmark(
        adjudicated_csv=args.adjudicated,
        papers_out=args.papers_out,
        gold_out=args.gold_out,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
