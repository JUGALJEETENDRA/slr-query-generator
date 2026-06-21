import json

from ollama_client import ask_ollama
from gemini_client import ask_gemini  # NEW: online mode support


def screen_paper(
    title,
    abstract,
    research_question,
    model="qwen2.5:7b",
    mode="local"  # NEW: 'local' (Ollama) or 'online' (Gemini)
):
    prompt = f"""
You are conducting title and abstract screening for a systematic literature review.

Research Question:
{research_question}

Paper Title:
{title}

Paper Abstract:
{abstract}

Analysis Process:

1. Determine the evidence required to answer the research question.

2. Determine the primary contribution of the paper.

3. Compare the paper contribution against the required evidence.

4. Include the analyses in:

- required_evidence
- paper_contribution

5. Make the final decision.

Decision Rules:

KEEP:
The paper's contribution directly provides evidence relevant to answering the research question.

MAYBE:
The title and abstract do not provide enough information to determine whether the contribution is relevant.

REJECT:
The paper's contribution does not provide evidence relevant to answering the research question.

A paper should not be kept merely because it contains similar keywords.

A paper should not be kept merely because it discusses the same technology.

A paper should not be kept merely because it is a systematic review.

Focus on the paper's actual contribution and whether that contribution helps answer the research question.

Use MAYBE only when information is genuinely insufficient.

When uncertain between MAYBE and REJECT, choose REJECT.

A paper should be KEEP only when the paper contribution directly satisfies the required evidence.

Evidence that large language models are useful in a different domain does NOT count as evidence for automating systematic literature reviews.

Do not generalize from one application domain to another.

If the required evidence and the paper contribution concern different tasks, objectives, or domains, choose REJECT.

Evidence about machine learning, traditional NLP, text classification, or artificial intelligence methods does not count as evidence about large language models unless large language models are explicitly evaluated, compared, or used.

Return ONLY valid JSON in this format:

{{
  "required_evidence":"...",
  "paper_contribution":"...",
  "decision":"KEEP",
  "reason":"..."
}}

Do not output markdown.
Do not output explanations.
Do not output anything except JSON.
"""

    # NEW: choose backend based on mode
    if mode == "online":
        response = ask_gemini(prompt)
    else:
        response = ask_ollama(prompt, model=model)

    try:
        parsed = json.loads(response)

        # New robust parsing logic
        if "decision" in parsed:
            decision = parsed["decision"].strip().upper()
            reason = parsed.get("reason", "No reason provided.")

            required_evidence = parsed.get(
                "required_evidence",
                ""
            )

            paper_contribution = parsed.get(
                "paper_contribution",
                ""
            )

        elif "KEEP" in parsed:
            decision = "KEEP"
            reason = parsed["KEEP"]
            required_evidence = ""
            paper_contribution = ""

        elif "MAYBE" in parsed:
            decision = "MAYBE"
            reason = parsed["MAYBE"]
            required_evidence = ""
            paper_contribution = ""

        elif "REJECT" in parsed:
            decision = "REJECT"
            reason = parsed["REJECT"]
            required_evidence = ""
            paper_contribution = ""

        else:
            decision = "PARSE_ERROR"
            reason = str(parsed)
            required_evidence = ""
            paper_contribution = ""

        return {
            "decision": decision,
            "reason": reason,
            "required_evidence": required_evidence,
            "paper_contribution": paper_contribution
        }

    except Exception as e:
        return {
            "decision": "PARSE_ERROR",
            "reason": str(e),
            "required_evidence": "",
            "paper_contribution": ""
        }