import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from functools import partial
import pandas as pd

from semantic_comparator import compare_semantic_frames
from semantic_frame import extract_semantic_frame

CSV_PATH = "uploads/LitSync_Clean_Dataset_2026-06-07.csv"
OUTPUT_PATH = "outputs/comparator_results.csv"

# Research question text – will be passed through the semantic extractor
RQ_TEXT = "large language models for automating systematic literature reviews"


def _find_col(df, candidates):
    lower_map = {column.lower(): column for column in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def process_paper(paper_data, title_col, abstract_col, rq_frame):
    """Processes a single paper row to extract and compare semantic frames."""
    index, row = paper_data
    title = str(row[title_col])
    abstract = str(row[abstract_col])

    paper_frame = extract_semantic_frame(
        title=title,
        abstract=abstract,
    )
    comparison = compare_semantic_frames(
        rq_frame,
        paper_frame,
    )

    return {
        "index": index,
        "title": title,
        "paper_frame": paper_frame,
        "comparison": comparison,
    }


def run_benchmark(
    max_workers: int,
    csv_path=CSV_PATH,
    output_path=OUTPUT_PATH,
):
    """Runs the semantic screening benchmark on a CSV file."""
    start_time = time.time()

    df = pd.read_csv(csv_path)

    abstract_col = _find_col(df, [
        "Abstract", "abstract", "AB", "Abstracts", "Summary",
        "Author Abstract", "abstract_note", "Description",
    ])
    title_col = _find_col(df, [
        "Title", "title", "TI", "Article Title", "Document Title",
        "paper_title", "Name",
    ])

    if abstract_col is None:
        raise KeyError(f"No Abstract column found. Columns in CSV: {list(df.columns)}")
    if title_col is None:
        raise KeyError(f"No Title column found. Columns in CSV: {list(df.columns)}")

    # Extract the RQ semantic frame once, using the same extractor as for papers
    rq_frame = extract_semantic_frame(
        title=RQ_TEXT,
        abstract="",          # No abstract needed for a research question
    )

    # Process all rows with non‑null abstracts (no head limit)
    valid_rows = df[df[abstract_col].notna()]

    # Log configuration at startup
    print(f"Workers: {max_workers}")
    print(f"Papers: {len(valid_rows)}")

    results = []
    counts = {
        "KEEP": 0,
        "MAYBE": 0,
        "REJECT": 0,
    }
    errors = 0

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    process_func = partial(process_paper, title_col=title_col, abstract_col=abstract_col, rq_frame=rq_frame,)

    # --- Progress counter fix ---
    completed = 0  # tracks total number of papers processed (success + error)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(process_func, (i, row)): i for i, (_, row) in enumerate(valid_rows.iterrows())}
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                comparison = result["comparison"]
                paper_frame = result["paper_frame"]
                decision = comparison["decision"]
                counts[decision] += 1

                result_row = {
                    "Title": result["title"],
                    "technology_match": comparison["technology_match"],
                    "task_match": comparison["task_match"],
                    "task_subject_match": comparison.get("task_subject_match", 0.0),
                    "task_role_match": comparison.get("task_role_match", 0.0),
                    "canonical_task_left": comparison.get("canonical_task_left", ""),
                    "canonical_task_right": comparison.get("canonical_task_right", ""),
                    "task_identity_match": comparison.get("task_identity_match", False),
                    "subject_match": comparison.get("subject_match", 0.0),
                    "context_match": comparison["context_match"],
                    "study_role_match": comparison.get("study_role_match", False),
                    "review_role_match": comparison.get("review_role_match", False),
                    "decision": decision,
                    # Additional diagnostic columns from the paper's semantic frame
                    "intervention_or_method": paper_frame.get("intervention_or_method", ""),
                    "target_problem_or_task": paper_frame.get("target_problem_or_task", ""),
                    "study_role": paper_frame.get("study_role", ""),
                    "application_context": paper_frame.get("application_context", ""),
                    "_index": result["index"],  # Internal index for sorting
                }
                results.append(result_row)

                # --- Success progress update ---
                completed += 1
                print(f"[{completed}/{len(valid_rows)}] {decision}", flush=True)

            except Exception as exc:
                # --- Error progress update ---
                completed += 1
                print(f"[{completed}/{len(valid_rows)}] ERROR", flush=True)
                errors += 1
                print(f"Paper at index {index} generated an exception: {exc}")

    # Write all results to CSV once after processing all papers
    # Sort results by the original index to preserve the input file's order
    if results:
        results.sort(key=lambda x: x["_index"])
        # Remove internal index before writing to CSV
        for r in results:
            del r["_index"]
        pd.DataFrame(results).to_csv(output_path, index=False)

    runtime = time.time() - start_time

    # Use the total number of papers (valid_rows) for throughput calculation
    processed_count = len(valid_rows)
    papers_per_sec = processed_count / runtime if runtime > 0 else 0
    avg_per_paper = runtime / processed_count if processed_count > 0 else 0

    print("\n" + "="*30)
    print("Benchmark Complete")
    print("="*30)
    print(f"KEEP count: {counts['KEEP']}")
    print(f"MAYBE count: {counts['MAYBE']}")
    print(f"REJECT count: {counts['REJECT']}")
    print(f"Errors: {errors}")
    print("\n--- Performance ---")
    print(f"Total Runtime: {runtime:.2f}s")
    print(f"Papers Processed: {processed_count}")
    print(f"Average per paper: {avg_per_paper:.2f}s")
    print(f"Papers per second: {papers_per_sec:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LitSync Screener benchmark.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers to use.")
    args = parser.parse_args()
    run_benchmark(max_workers=args.workers)
