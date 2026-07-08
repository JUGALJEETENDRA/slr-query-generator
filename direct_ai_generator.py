"""
Direct AI query generation for the production query API.

Generates a complete Boolean query directly from a research question using the
existing one-shot LLM prompt.
"""

from openai import OpenAI

from config import DEFAULT_MODEL


client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


SYSTEM_PROMPT = """
You are an expert academic researcher specializing in Systematic Literature Reviews (SLRs).

Your task is to generate a publication-quality Boolean search query suitable for Scopus.

Rules:

1. Preserve the exact concepts from the research question.
2. Expand only with academically valid synonyms.
3. Do NOT broaden the scope.
4. Do NOT introduce unrelated concepts.
5. Do NOT replace specific research tasks with evaluation metrics.
6. Use quoted phrases whenever appropriate.
7. Join synonyms using OR.
8. Join concept groups using AND.
9. Produce concise, high-precision search strings.

Return ONLY the Boolean query.

Example:

Research Question:
What machine learning techniques have been developed for software defect prediction?

Output:

("machine learning" OR "ML")
AND
("software defect" OR "software bug")
AND
("prediction" OR "predictive modeling")
"""


def generate_query(question: str, model: str = DEFAULT_MODEL) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content.strip()
