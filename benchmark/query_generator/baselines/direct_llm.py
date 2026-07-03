
import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

sys.path.insert(0, PROJECT_ROOT)

from direct_ai_generator import generate_query


if __name__ == "__main__":

    question = (
        "What machine learning techniques have been developed "
        "for software defect prediction?"
    )

    print("=" * 80)
    print("Direct LLM Baseline")
    print("=" * 80)
    print()

    print("Question:")
    print(question)
    print()

    print("Generated Query:")
    print("-" * 80)
    print(generate_query(question))
