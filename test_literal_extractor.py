from openai import OpenAI

from config import DEFAULT_MODEL
from extractor import extract_literal_spans

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

question = "What machine learning techniques have been developed for software defect prediction?"

result = extract_literal_spans(
    client,
    DEFAULT_MODEL,
    question
)

print("\nExtracted phrases")
print("-----------------")

for phrase in result.phrases:
    print("-", phrase)