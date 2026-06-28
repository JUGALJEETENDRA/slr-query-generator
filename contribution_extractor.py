import json
from ollama_client import ask_ollama

def extract_contribution(
    title,
    abstract,
    model="qwen2.5:3b"
):
    prompt = f"""
Paper Title:
{title}

Paper Abstract:
{abstract}

Identify:

1. What is the paper ABOUT?
2. What does the paper CONTRIBUTE?
3. What task is being automated, evaluated, or studied?

Return ONLY JSON:

{{
  "paper_topic":"",
  "paper_contribution":"",
  "paper_task":""
}}

No explanation.
No markdown.
"""

    response = ask_ollama(
        prompt,
        model=model
    )

    return json.loads(response)