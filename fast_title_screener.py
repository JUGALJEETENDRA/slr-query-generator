import json
from ollama_client import ask_ollama

def title_score(
    title,
    research_question,
    model="qwen2.5:7b"
):
    prompt = f"""
Research Question:
{research_question}

Paper Title:
{title}

Rate how likely this title is relevant.

Return ONLY JSON:

{{
  "score": 0-100
}}

No explanation.
"""

    response = ask_ollama(prompt, model=model)

    try:
        return json.loads(response)["score"]
    except Exception:
        return 100