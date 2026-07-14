"""Compatibility client for non-screening features such as query generation."""

import os

import requests


def ask_ollama(prompt, model="qwen3:8b"):
    response = requests.post(
        os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": "5m",
            "options": {"temperature": 0.1, "num_ctx": 4096},
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"]
