import os
import time
import pandas as pd
from screener import screen_paper
from fast_title_screener import title_score
from semantic_frame import extract_semantic_frame
from config import DEFAULT_MODEL
from concurrent.futures import ThreadPoolExecutor, as_completed  # Added for parallel processing

# ---------- PROGRESS dictionary for UI feedback ----------
PROGRESS = {
    "status": "idle",
    "current": 0,
    "total": 0,
    "keep": 0,
    "maybe": 0,
    "reject": 0,
}
# ---------------------------------------------------------


def _find_col(df, candidates):
    """Return the first column name from candidates that exists in df (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


# ---------- NEW: process_paper function (replaces per-paper logic) ----------
def process_paper(
    row,
    title_col,
    abstract_col,
    research_question,
    rq_frame,
    mode,
    model,
):
    title = row[title_col]
    abstract = str(row[abstract_col])

    score = title_score(
        title=title,
        research_question=research_question,
        model=model,
    )

    if score < 10:
        return {
            "decision": "REJECT",
            "title": title,
            "abstract": abstract,
            "reason": "TITLE_GATE_REJECT",
            "required_evidence": "",
            "paper_contribution": "",
        }

    result = screen_paper(
        title=title,
        abstract=abstract,
        research_question=research_question,
        rq_frame=rq_frame,
        mode=mode,
        model=model,
    )

    result["title"] = title
    result["abstract"] = abstract

    return result
# ---------------------------------------------------------------------------


def screen_csv(
    csv_path,
    research_question,
    output_path="outputs/screened.csv",
    mode="local",
    model=DEFAULT_MODEL,
):

    df = pd.read_csv(csv_path)

    # ---- PERFORMANCE TIMER START ----
    overall_start = time.perf_counter()
    # ---------------------------------

    # Auto-detect Abstract and Title columns across database export formats
    abstract_col = _find_col(df, [
        "Abstract", "abstract", "AB", "Abstracts", "Summary",
        "Author Abstract", "abstract_note", "Description"
    ])
    title_col = _find_col(df, [
        "Title", "title", "TI", "Article Title", "Document Title",
        "paper_title", "Name"
    ])

    if abstract_col is None:
        raise KeyError(
            f"No Abstract column found. Columns in your CSV: {list(df.columns)}"
        )
    if title_col is None:
        raise KeyError(
            f"No Title column found. Columns in your CSV: {list(df.columns)}"
        )

    valid_rows = df[df[abstract_col].notna()].head(10)

    # ---------- Update PROGRESS after valid_rows ----------
    PROGRESS["status"] = "running"
    PROGRESS["current"] = 0
    PROGRESS["total"] = len(valid_rows)
    PROGRESS["keep"] = 0
    PROGRESS["maybe"] = 0
    PROGRESS["reject"] = 0
    # -------------------------------------------------------

    rq_frame = extract_semantic_frame(
        title=research_question,
        abstract="",
        model=model,
    )

    keep_count = 0
    maybe_count = 0
    reject_count = 0
    parse_error_count = 0

    results = []
    included_results = []
    maybe_results = []
    excluded_results = []

    # ---------- REPLACED the original for-loop with ThreadPoolExecutor ----------
    with ThreadPoolExecutor(max_workers=4) as executor:

        futures = [
            executor.submit(
                process_paper,
                row,
                title_col,
                abstract_col,
                research_question,
                rq_frame,
                mode,
                model,
            )
            for _, row in valid_rows.iterrows()
        ]

        for i, future in enumerate(as_completed(futures), start=1):

            result = future.result()

            title = result["title"]
            abstract = result["abstract"]
            decision = result["decision"]
            reason = result["reason"]

            if decision == "KEEP":
                keep_count += 1
            elif decision == "MAYBE":
                maybe_count += 1
            elif decision == "REJECT":
                reject_count += 1

            PROGRESS["current"] = i
            PROGRESS["keep"] = keep_count
            PROGRESS["maybe"] = maybe_count
            PROGRESS["reject"] = reject_count

            results.append({
                "Title": title,
                "Abstract": abstract,
                "Decision": decision,
                "Reason": reason,
                "Required_Evidence": result.get("required_evidence", ""),
                "Paper_Contribution": result.get("paper_contribution", ""),
            })

            if decision == "KEEP":
                included_results.append({
                    "Title": title,
                    "Abstract": abstract,
                    "Reason": reason,
                })

            elif decision == "MAYBE":
                maybe_results.append({
                    "Title": title,
                    "Abstract": abstract,
                    "Reason": reason,
                })

            elif decision == "REJECT":
                excluded_results.append({
                    "Title": title,
                    "Abstract": abstract,
                    "Reason": reason,
                })
    # ---------------------------------------------------------------------------

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)

    # Save the three category-specific CSV files in the same directory as output_path
    output_dir = os.path.dirname(output_path)
    if output_dir:
        included_path = os.path.join(output_dir, "included_studies.csv")
        maybe_path = os.path.join(output_dir, "maybe_studies.csv")
        excluded_path = os.path.join(output_dir, "excluded_studies.csv")
    else:
        included_path = "included_studies.csv"
        maybe_path = "maybe_studies.csv"
        excluded_path = "excluded_studies.csv"

    if included_results:
        pd.DataFrame(included_results).to_csv(included_path, index=False)
    if maybe_results:
        pd.DataFrame(maybe_results).to_csv(maybe_path, index=False)
    if excluded_results:
        pd.DataFrame(excluded_results).to_csv(excluded_path, index=False)

    PROGRESS["status"] = "finished"

    # ---- PERFORMANCE SUMMARY ----
    overall_time = time.perf_counter() - overall_start
    print("\n========== PERFORMANCE ==========")
    print(f"Total runtime: {overall_time:.2f} s")
    print(f"Average per paper: {overall_time / len(valid_rows):.2f} s")
    print("=================================\n")
    # -----------------------------

    return {
        "keep": keep_count,
        "maybe": maybe_count,
        "reject": reject_count,
        "parse_error": parse_error_count,
        "output_file": output_path
    }


if __name__ == "__main__":
    summary = screen_csv(
        csv_path="uploads/LitSync_Clean_Dataset_2026-06-07.csv",
        research_question="Can large language models help automate systematic literature reviews?"
    )
    print(summary)