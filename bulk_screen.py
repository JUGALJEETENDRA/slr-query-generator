import os
import pandas as pd
from screener import screen_paper


def screen_csv(
    csv_path,
    research_question,
    output_path="outputs/screened.csv",
    mode="local"
):
    df = pd.read_csv(csv_path)

    valid_rows = df[df["Abstract"].notna()].head(200)

    keep_count = 0
    maybe_count = 0
    reject_count = 0
    parse_error_count = 0

    results = []
    included_results = []
    maybe_results = []
    excluded_results = []

    for i, (_, row) in enumerate(valid_rows.iterrows(), start=1):
        title = row["Title"]
        abstract = str(row["Abstract"])

        result = screen_paper(
            title=title,
            abstract=abstract,
            research_question=research_question,
            mode=mode
        )

        decision = result["decision"]
        reason = result["reason"]

        if decision == "KEEP":
            keep_count += 1
        elif decision == "MAYBE":
            maybe_count += 1
        elif decision == "REJECT":
            reject_count += 1
        else:
            parse_error_count += 1

        results.append({
            "Title": title,
            "Abstract": abstract,
            "Decision": decision,
            "Reason": reason,
            "Required_Evidence": result.get(
                "required_evidence",
                ""
            ),
            "Paper_Contribution": result.get(
                "paper_contribution",
                ""
            )
        })

        if decision == "KEEP":
            included_results.append({
                "Title": title,
                "Abstract": abstract,
                "Reason": reason
            })
        elif decision == "MAYBE":
            maybe_results.append({
                "Title": title,
                "Abstract": abstract,
                "Reason": reason
            })
        elif decision == "REJECT":
            excluded_results.append({
                "Title": title,
                "Abstract": abstract,
                "Reason": reason
            })

        print(f"[{i}/{len(valid_rows)}]")

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