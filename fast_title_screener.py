import json
from ollama_client import ask_ollama

def title_score(
    title,
    research_question,
    model="qwen2.5:3b",
    inference_engine=None,
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

    ask = inference_engine.ask if inference_engine is not None else ask_ollama
    response = ask(prompt, model=model)

    try:
        return json.loads(response)["score"]
    except Exception:
        return 100
