"""One-off end-to-end gpt-oss:20b smoke test for the supplied LitSync CSV."""

import os

from bulk_screen import screen_csv


os.environ.update(
    {
        "LOCAL_MODEL": "gpt-oss:20b",
        "OLLAMA_NUM_CTX": "8192",
        "OLLAMA_NUM_PREDICT": "768",
        "OLLAMA_NUM_THREAD": "8",
        "OLLAMA_TEMPERATURE": "0.1",
        "OLLAMA_KEEP_ALIVE": "30m",
        "OLLAMA_REQUEST_TIMEOUT_SECONDS": "900",
        "OLLAMA_MAX_CONCURRENT": "1",
        "ENABLE_PARALLEL_SCREENING": "false",
        "SCREENING_WORKERS": "1",
    }
)

result = screen_csv(
    csv_path=r"C:\Users\xyz\Downloads\LitSync_Clean_Dataset_2026-07-27 (3).csv",
    research_question="What factors influence energy efficiency in edge computing systems?",
    output_path="outputs/gptoss_20b_energy_efficiency_10_papers.csv",
    mode="local",
    model="gpt-oss:20b",
    max_rows=10,
)
print(result)
