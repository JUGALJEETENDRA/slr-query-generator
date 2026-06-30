"""CSV screening orchestration for Local AI and Hybrid modes."""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List

import pandas as pd

from batch_builder import build_screening_prompt, make_batches
from csv_writer import write_screening_outputs
from embedding_screener import EmbeddingScreener
from gemini_browser import GeminiBrowser
from response_parser import parse_batch_response
from screener import screen_paper
from summary import write_summary


def _find_col(df: pd.DataFrame, candidates: List[str]):
    lower_map = {str(column).lower(): column for column in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _record(paper: Dict, result: Dict, stage: str) -> Dict:
    return {
        "_Paper_ID": paper["id"],
        "Title": paper["title"],
        "Abstract": paper["abstract"],
        "Decision": result["decision"],
        "Reason": result.get("reason", ""),
        "Required_Evidence": result.get("required_evidence", ""),
        "Paper_Contribution": result.get("paper_contribution", ""),
        "Similarity": paper.get("similarity", ""),
        "Screening_Stage": stage,
    }


def _load_papers(csv_path: str):
    df = pd.read_csv(csv_path)
    abstract_col = _find_col(df, [
        "Abstract", "AB", "Abstracts", "Summary", "Author Abstract",
        "abstract_note", "Description",
    ])
    title_col = _find_col(df, [
        "Title", "TI", "Article Title", "Document Title", "paper_title", "Name",
    ])
    if abstract_col is None:
        raise KeyError(f"No Abstract column found. Columns in your CSV: {list(df.columns)}")
    if title_col is None:
        raise KeyError(f"No Title column found. Columns in your CSV: {list(df.columns)}")

    papers = []
    for number, (_, row) in enumerate(df.iterrows(), start=1):
        title = "" if pd.isna(row[title_col]) else str(row[title_col]).strip()
        abstract = "" if pd.isna(row[abstract_col]) else str(row[abstract_col]).strip()
        papers.append({"id": f"paper_{number}", "title": title, "abstract": abstract})
    return papers


def _screen_local(papers: List[Dict], research_question: str, model: str) -> List[Dict]:
    records = []
    for index, paper in enumerate(papers, start=1):
        if not paper["title"]:
            result = {"decision": "REJECT", "reason": "MISSING_TITLE"}
        else:
            result = screen_paper(
                title=paper["title"],
                abstract=paper["abstract"],
                research_question=research_question,
                model=model,
            )
        records.append(_record(paper, result, "LOCAL_AI"))
        print(f"[{index}/{len(papers)}] {result['decision']}")
    return records


def _screen_hybrid(
    papers: List[Dict],
    research_question: str,
    threshold: float,
    batch_size: int,
    embedding_model: str,
    browser_factory: Callable[[], GeminiBrowser],
):
    records = []
    titled = [paper for paper in papers if paper["title"]]
    scores = EmbeddingScreener(model=embedding_model).score_titles(
        research_question, [paper["title"] for paper in titled]
    ) if titled else []
    for paper, score in zip(titled, scores):
        paper["similarity"] = round(score, 6)

    candidates = []
    embedding_rejected = 0
    for paper in papers:
        if not paper["title"]:
            records.append(_record(
                paper, {"decision": "REJECT", "reason": "MISSING_TITLE"}, "VALIDATION"
            ))
        elif paper["similarity"] < threshold:
            embedding_rejected += 1
            reason = (
                f"EMBEDDING_REJECT: similarity {paper['similarity']:.6f} "
                f"is below threshold {threshold:.6f}"
            )
            records.append(_record(
                paper, {"decision": "REJECT", "reason": reason}, "EMBEDDING_FILTER"
            ))
        else:
            candidates.append(paper)

    if candidates:
        with browser_factory() as browser:
            for batch_number, batch in enumerate(make_batches(candidates, batch_size), start=1):
                ids = [paper["id"] for paper in batch]
                try:
                    response = browser.submit(build_screening_prompt(research_question, batch))
                    parsed = parse_batch_response(response, ids)
                    by_id = {result["id"]: result for result in parsed}
                    for paper in batch:
                        records.append(_record(paper, by_id[paper["id"]], "GEMINI_BROWSER"))
                    print(f"[Gemini batch {batch_number}] screened {len(batch)} papers")
                except Exception as exc:
                    for paper in batch:
                        records.append(_record(paper, {
                            "decision": "PARSE_ERROR",
                            "reason": f"Gemini batch failed: {exc}",
                        }, "GEMINI_BROWSER"))

    order = {paper["id"]: index for index, paper in enumerate(papers)}
    records.sort(key=lambda item: order[item["_Paper_ID"]])
    return records, embedding_rejected, len(candidates)


def screen_csv(
    csv_path,
    research_question,
    output_path="outputs/screened.csv",
    mode="local",
    embedding_threshold=0.35,
    batch_size=10,
    embedding_model="nomic-embed-text",
    local_model="qwen2.5:7b",
    browser_factory=GeminiBrowser,
):
    """Screen a CSV and write the complete set of Version 2 outputs."""
    started = time.perf_counter()
    mode = str(mode).lower().strip()
    if mode not in {"local", "hybrid"}:
        raise ValueError("mode must be 'local' or 'hybrid'; Gemini API mode was removed")
    if not research_question or not str(research_question).strip():
        raise ValueError("research_question is required")
    if not -1.0 <= float(embedding_threshold) <= 1.0:
        raise ValueError("embedding_threshold must be between -1 and 1")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be at least 1")

    papers = _load_papers(csv_path)
    if mode == "local":
        records = _screen_local(papers, research_question, local_model)
        embedding_rejected = 0
        sent_to_gemini = 0
    else:
        records, embedding_rejected, sent_to_gemini = _screen_hybrid(
            papers, research_question, float(embedding_threshold), int(batch_size),
            embedding_model, browser_factory,
        )

    paths = write_screening_outputs(records, output_path)
    counts = {
        "keep": sum(record["Decision"] == "KEEP" for record in records),
        "maybe": sum(record["Decision"] == "MAYBE" for record in records),
        "reject": sum(record["Decision"] == "REJECT" for record in records),
        "parse_error": sum(record["Decision"] == "PARSE_ERROR" for record in records),
    }
    elapsed = round(time.perf_counter() - started, 3)
    metrics = {
        "Total papers": len(papers),
        "Screening mode": mode,
        "Embedding model": embedding_model if mode == "hybrid" else "N/A",
        "Embedding threshold": float(embedding_threshold) if mode == "hybrid" else "N/A",
        "Papers rejected by embedding": embedding_rejected,
        "Papers sent to Gemini": sent_to_gemini,
        "Included": counts["keep"],
        "Excluded": counts["reject"],
        "Maybe": counts["maybe"],
        "Parse errors": counts["parse_error"],
        "Execution time (seconds)": elapsed,
    }
    summary_path = write_summary(metrics, os.path.dirname(output_path) or ".")
    return {
        **counts,
        "total": len(papers),
        "embedding_rejected": embedding_rejected,
        "sent_to_gemini": sent_to_gemini,
        "execution_time_seconds": elapsed,
        "output_file": output_path,
        "summary_file": summary_path,
        "files": {**paths, "summary": summary_path},
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Screen an SLR CSV with Local AI or Hybrid mode")
    parser.add_argument("csv_path")
    parser.add_argument("research_question")
    parser.add_argument("--mode", choices=("local", "hybrid"), default="local")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--output", default="outputs/screened.csv")
    args = parser.parse_args()
    print(screen_csv(
        args.csv_path, args.research_question, args.output, args.mode,
        args.threshold, args.batch_size,
    ))
