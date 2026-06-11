import json

from ollama_client import ask_ollama


def screen_paper(
    title,
    abstract,
    research_question,
    model="qwen2.5:7b"
):
    prompt = f"""
You are conducting title and abstract screening for a systematic literature review.

Research Question:
{research_question}

Paper Title:
{title}

Paper Abstract:
{abstract}

Determine whether this paper provides DIRECT evidence for answering the research question.

KEEP:
The paper directly studies, evaluates, benchmarks, compares, or applies methods relevant to answering the research question.

MAYBE:
The paper is partially related.

REJECT:
The paper is not relevant.

Return ONLY valid JSON.

Example:
{{"decision":"KEEP","reason":"Directly answers the research question."}}

Do not output markdown.
Do not output explanations.
Do not output anything except JSON.
"""

    response = ask_ollama(
        prompt,
        model=model
    )

    try:
        parsed = json.loads(response)

        decision = parsed.get(
            "decision",
            "PARSE_ERROR"
        ).strip().upper()

        reason = parsed.get(
            "reason",
            "No reason provided."
        )

        return {
            "decision": decision,
            "reason": reason
        }

    except Exception:
        return {
            "decision": "PARSE_ERROR",
            "reason": response.strip()
        }