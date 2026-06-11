import pandas as pd
import json
from ollama_client import ask_ollama

RQ = "Can large language models help automate systematic literature reviews?"

df = pd.read_csv("data/LitSync_Clean_Dataset_2026-06-07.csv")

# Use all rows with non‑null abstracts (full dataset)
valid_rows = df[df["Abstract"].notna()]

# Initialize counters
keep_count = 0
maybe_count = 0
reject_count = 0
parse_error_count = 0

results = []

for i, (_, row) in enumerate(valid_rows.iterrows(), start=1):
    title = row["Title"]
    abstract = str(row["Abstract"])

    prompt = f"""
You are conducting title and abstract screening for a systematic literature review.

Research Question:
{RQ}

Paper Title:
{title}

Paper Abstract:
{abstract}

Determine whether this paper provides DIRECT evidence for answering the research question.

KEEP:
The paper directly studies, evaluates, benchmarks, compares, or applies large language models for:
- literature reviews
- systematic reviews
- evidence synthesis
- citation screening
- study selection
- abstract screening
- systematic review automation
- review assistance

MAYBE:
The paper studies large language models but NOT specifically for literature review automation.

REJECT:
The paper does not provide evidence relevant to the research question.

Return ONLY valid JSON.

Example:
{{"decision":"KEEP","reason":"Directly evaluates LLMs for systematic review automation."}}

Do not output markdown.
Do not output explanations.
Do not output anything except JSON.
"""

    response = ask_ollama(
        prompt,
        model="qwen2.5:7b"
    )

    print(f"[{i}/{len(valid_rows)}]")

    try:
        parsed = json.loads(response)
        # Use .get() with defaults to avoid KeyError when fields are missing
        decision = parsed.get(
            "decision",
            "PARSE_ERROR"
        ).strip().upper()
        reason = parsed.get(
            "reason",
            "No reason provided."
        )

        # Update counters based on decision
        if decision == "KEEP":
            keep_count += 1
        elif decision == "MAYBE":
            maybe_count += 1
        elif decision == "REJECT":
            reject_count += 1

        results.append({
            "Title": title,
            "Decision": decision,
            "Reason": reason
        })
    except Exception:
        parse_error_count += 1
        results.append({
            "Title": title,
            "Decision": "PARSE_ERROR",
            "Reason": response.strip()
        })

result_df = pd.DataFrame(results)

result_df.to_csv(
    "outputs/screened.csv",
    index=False
)

# Print decision summary
print()
print("KEEP =", keep_count)
print("MAYBE =", maybe_count)
print("REJECT =", reject_count)
print("PARSE_ERROR =", parse_error_count)

print("\nDone.")