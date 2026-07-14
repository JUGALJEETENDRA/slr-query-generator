"""Local Boolean query generation used by the existing query endpoint."""

from local_ai.hardware import resolve_runtime_profile
from ollama_client import ask_ollama


SYSTEM_PROMPT = """
You are an expert systematic-review search strategist. Convert the research question into a concise,
high-precision Boolean search query. Preserve its concepts and relationships, add only academically valid
synonyms, quote phrases, join synonyms with OR, and join concept groups with AND. Return only the query.
""".strip()


def generate_query(question: str, model: str | None = None) -> str:
    model = model or resolve_runtime_profile().fast_model
    return ask_ollama(f"{SYSTEM_PROMPT}\n\nResearch question:\n{question}", model=model).strip()
