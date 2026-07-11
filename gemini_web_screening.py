from __future__ import annotations

import os
import hashlib
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from gemini_web_automation import GeminiWebAutomation, GeminiWebConfig
from gemini_web_parser import GeminiResponseParseError, parse_gemini_screening_response
from gemini_web_prompt import ScreeningPaper, build_screening_prompt


GEMINI_WEB_ENGINE = "gemini_web"
DEFAULT_GEMINI_WEB_BATCH_SIZE = 5


@dataclass(frozen=True)
class GeminiWebScreeningOptions:
    batch_size: int = DEFAULT_GEMINI_WEB_BATCH_SIZE
    inclusion_criteria: str = ""
    exclusion_criteria: str = ""
    output_dir: str = "outputs"
    checkpoint_path: str | None = None
    max_retries: int = 1
    browser: GeminiWebConfig = GeminiWebConfig()


def is_gemini_web_engine(value: str | None) -> bool:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in {"gemini_web", "gemini_web_automation", "web_gemini"}


def screen_csv_with_gemini_web(
    *,
    csv_path: str,
    research_question: str,
    progress,
    screening_session,
    progress_job_id: str | None = None,
    options: GeminiWebScreeningOptions | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Run the Gemini Web batch screening workflow.

    This function intentionally does not start, finish, or fail the global
    progress job. The caller owns the job lifecycle; this function owns only
    Gemini Web's batch execution, checkpointing, parsing, and count updates.
    """
    options = options or GeminiWebScreeningOptions()
    batch_size = max(1, int(options.batch_size or DEFAULT_GEMINI_WEB_BATCH_SIZE))
    progress_job_id = progress_job_id or f"gemini-web-{uuid.uuid4()}"

    df = pd.read_csv(csv_path)
    input_total_rows = len(df)
    effective_row_limit = _normalize_row_limit(max_rows)

    title_col = _find_col(df, ["Title", "title", "TI", "Article Title", "Document Title", "paper_title", "Name"])

    abstract_col = _find_col(df, ["Abstract", "abstract", "AB", "Abstracts", "Summary", "Author Abstract", "abstract_note", "Description"])
    id_col = _find_col(df, ["Paper ID", "paper_id", "ID", "id", "DOI", "doi"])

    if title_col is None:
        raise KeyError(f"No Title column found. Columns in your CSV: {list(df.columns)}")
    if abstract_col is None:
        raise KeyError(f"No Abstract column found. Columns in your CSV: {list(df.columns)}")

    valid_rows = df[df[abstract_col].notna()].copy()
    valid_total_rows = len(valid_rows)
    row_limit_applied = effective_row_limit is not None
    if row_limit_applied:
        valid_rows = valid_rows.head(effective_row_limit)
    screened_total_rows = len(valid_rows)
    row_limit_value = effective_row_limit or ""

    print(f"Loaded rows: {input_total_rows}")
    print(f"Valid screening rows: {valid_total_rows}")
    print(f"Screening rows: {screened_total_rows}")
    print(f"Row limit applied: {'yes' if row_limit_applied else 'no'}")

    output_root = Path(options.output_dir)
    job_output_dir = output_root / f"gemini_web_{progress_job_id}"
    job_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(options.checkpoint_path) if options.checkpoint_path else job_output_dir / "screened_checkpoint.csv"

    started_at = time.perf_counter()
    results: list[dict[str, Any]] = []
    counts = {"KEEP": 0, "MAYBE": 0, "REJECT": 0}

    papers: list[ScreeningPaper] = []
    metadata_by_id: dict[str, dict[str, Any]] = {}
    used_paper_ids: set[str] = set()
    for i, (_, row) in enumerate(valid_rows.iterrows(), start=1):
        paper = _row_to_screening_paper(
            index=i,
            row=row,
            title_col=title_col,
            abstract_col=abstract_col,
            id_col=id_col,
            used_ids=used_paper_ids,
        )
        papers.append(paper)
        metadata_by_id[paper.paper_id] = _original_metadata(row)

    if not papers:
        _write_outputs(results, job_output_dir, checkpoint_path, started_at, total=0)
        screening_session.set_results(results)
        return {
            "keep": 0,
            "maybe": 0,
            "reject": 0,
            "parse_error": 0,
            "output_dir": str(job_output_dir),
            "output_file": str(job_output_dir / "screened_checkpoint.csv"),
            "screening_engine": GEMINI_WEB_ENGINE,
            "batch_size": batch_size,
            "total_papers": 0,
            "input_total_rows": input_total_rows,
            "screened_total_rows": 0,
            "row_limit_applied": row_limit_applied,
            "row_limit_value": row_limit_value,
        }

    try:
        with GeminiWebAutomation(options.browser) as browser:
            for batch_number, batch in enumerate(_chunks(papers, batch_size), start=1):
                response_text = _execute_batch_with_retries(
                    browser=browser,
                    batch=batch,
                    research_question=research_question,
                    inclusion_criteria=options.inclusion_criteria,
                    exclusion_criteria=options.exclusion_criteria,
                    max_retries=options.max_retries,
                )

                parsed = parse_gemini_screening_response(
                    response_text,
                    expected_ids={paper.paper_id for paper in batch},
                )

                by_id = {decision.paper_id: decision for decision in parsed}
                for paper in batch:
                    decision = by_id[paper.paper_id]
                    counts[decision.decision] += 1
                    original = metadata_by_id.get(paper.paper_id, {})
                    results.append(
                        {
                            **original,
                            "Paper_ID": paper.paper_id,
                            "Title": paper.title,
                            "Abstract": paper.abstract,
                            "Decision": decision.decision,
                            "Reason": decision.reason,
                            "Confidence": "",
                            "Required_Evidence": "",
                            "Paper_Contribution": "",
                            "processing_engine": GEMINI_WEB_ENGINE,
                            "screening_strategy": GEMINI_WEB_ENGINE,
                            "batch_number": batch_number,
                            "input_total_rows": input_total_rows,
                            "valid_total_rows": valid_total_rows,
                            "screened_total_rows": screened_total_rows,
                            "row_limit_applied": row_limit_applied,
                            "row_limit_value": row_limit_value,
                        }
                    )

                _write_outputs(results, job_output_dir, checkpoint_path, started_at, total=len(valid_rows))
                screening_session.set_results(results)
                progress.update_counts(
                    progress_job_id,
                    len(results),
                    counts["KEEP"],
                    counts["MAYBE"],
                    counts["REJECT"],
                )
    except Exception as exc:
        _write_outputs(results, job_output_dir, checkpoint_path, started_at, total=len(valid_rows), error=str(exc))
        if results:
            screening_session.set_results(results)
        raise

    return {
        "keep": counts["KEEP"],
        "maybe": counts["MAYBE"],
        "reject": counts["REJECT"],
        "parse_error": 0,
        "output_dir": str(job_output_dir),
        "output_file": str(checkpoint_path),
        "screening_engine": GEMINI_WEB_ENGINE,
        "batch_size": batch_size,
        "total_papers": len(valid_rows),
        "input_total_rows": input_total_rows,
        "screened_total_rows": screened_total_rows,
        "row_limit_applied": row_limit_applied,
        "row_limit_value": row_limit_value,
        "rows": results,
    }


def _execute_batch_with_retries(
    *,
    browser: GeminiWebAutomation,
    batch: list[ScreeningPaper],
    research_question: str,
    inclusion_criteria: str,
    exclusion_criteria: str,
    max_retries: int,
) -> str:
    prompt = build_screening_prompt(
        research_question=research_question,
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
        papers=batch,
    )
    attempts = max(1, max_retries + 1)
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            response_text = browser.submit_prompt_and_get_response(prompt)
            parse_gemini_screening_response(response_text, expected_ids={paper.paper_id for paper in batch})
            return response_text
        except (GeminiResponseParseError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            browser.wait_until_ready()
    raise RuntimeError(f"Gemini batch failed after {attempts} attempt(s): {last_error}")


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(column).lower(): column for column in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _normalize_row_limit(value) -> int | None:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return None
    return limit if limit > 0 else None


def _row_to_screening_paper(
    *,
    index: int,
    row,
    title_col: str,
    abstract_col: str,
    id_col: str | None,
    used_ids: set[str],
) -> ScreeningPaper:
    title = " ".join(str(row[title_col]).split())
    abstract = " ".join(str(row[abstract_col]).split())
    fingerprint = hashlib.sha1(
        f"{title}\n{abstract}".encode("utf-8", errors="ignore")
    ).hexdigest()[:10]
    paper_id = f"P{index:04d}_{fingerprint}"
    if id_col is not None:
        raw_id = str(row.get(id_col, "")).strip()
        if raw_id:
            safe_raw_id = _safe_id(raw_id)
            if safe_raw_id:
                paper_id = f"{safe_raw_id}_{fingerprint}"
    if paper_id in used_ids:
        paper_id = f"{paper_id}_{index}"
    used_ids.add(paper_id)
    return ScreeningPaper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
    )


def _safe_id(value: str) -> str:
    cleaned = "".join(char for char in value if char.isalnum() or char in {"_", ".", "-"})
    return cleaned[:80]


def _chunks(items: list[ScreeningPaper], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _original_metadata(row) -> dict[str, Any]:
    metadata = {}
    for key, value in row.to_dict().items():
        if pd.isna(value):
            metadata[str(key)] = ""
        else:
            metadata[str(key)] = value
    return metadata


def _write_outputs(
    results: list[dict[str, Any]],
    output_dir: Path,
    checkpoint_path: Path,
    started_at: float,
    *,
    total: int,
    error: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(results)
    if result_df.empty:
        result_df = pd.DataFrame(columns=["Paper_ID", "Title", "Abstract", "Decision", "Reason"])

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_dir / "screened_checkpoint.csv", index=False)
    if checkpoint_path != output_dir / "screened_checkpoint.csv":
        result_df.to_csv(checkpoint_path, index=False)
    result_df[result_df["Decision"] == "KEEP"].to_csv(output_dir / "included.csv", index=False)
    result_df[result_df["Decision"] == "REJECT"].to_csv(output_dir / "excluded.csv", index=False)
    result_df[result_df["Decision"] == "MAYBE"].to_csv(output_dir / "maybe.csv", index=False)

    elapsed = time.perf_counter() - started_at
    completed = len(results)
    remaining = max(0, total - completed)
    estimated_remaining = (elapsed / completed * remaining) if completed else None

    summary_lines = [
        "Gemini Web Automation Screening Summary",
        f"Completed papers: {completed}",
        f"Remaining papers: {remaining}",
        f"Included: {int((result_df['Decision'] == 'KEEP').sum()) if not result_df.empty else 0}",
        f"Maybe: {int((result_df['Decision'] == 'MAYBE').sum()) if not result_df.empty else 0}",
        f"Excluded: {int((result_df['Decision'] == 'REJECT').sum()) if not result_df.empty else 0}",
        f"Elapsed seconds: {elapsed:.2f}",
        f"Estimated remaining seconds: {estimated_remaining:.2f}" if estimated_remaining is not None else "Estimated remaining seconds: unknown",
    ]
    if error:
        summary_lines.append(f"Last error: {error}")
    (output_dir / "screening_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
