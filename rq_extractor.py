import json
from ollama_client import ask_ollama

def extract_rq(
    research_question,
    model="qwen2.5:7b"
):
    prompt = f"""
Research Question:
{research_question}

Extract:

1. Technology being studied
2. Task being automated/evaluated
3. Evidence being sought

Return ONLY JSON:

{{
  "technology":"",
  "task":"",
  "evidence":""
}}

No explanation.
No markdown.
"""

    response = ask_ollama(
        prompt,
        model=model
    )

    return json.loads(response)