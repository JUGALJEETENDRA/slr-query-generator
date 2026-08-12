"""Small, dependency-light adapter for a locally running Ollama server."""

from __future__ import annotations

import os
from typing import Any

import requests


DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def _ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")


def local_model_name(default: str = "qwen2.5:3b") -> str:
    """Return the configured local model without forcing a model download."""
    return os.getenv("LOCAL_MODEL", os.getenv("GPT_OSS_MODEL", default)).strip() or default


def _generation_options() -> dict[str, Any]:
    """Options shared by all local requests; unset values use Ollama defaults."""
    options: dict[str, Any] = {}
    numeric_options = {
        "num_ctx": "OLLAMA_NUM_CTX",
        "num_predict": "OLLAMA_NUM_PREDICT",
        "num_thread": "OLLAMA_NUM_THREAD",
    }
    for option, env_name in numeric_options.items():
        value = os.getenv(env_name)
        if value:
            try:
                options[option] = int(value)
            except ValueError:
                pass
    temperature = os.getenv("OLLAMA_TEMPERATURE")
    if temperature:
        try:
            options["temperature"] = float(temperature)
        except ValueError:
            pass
    return options


def ask_ollama(prompt: str, model: str | None = None) -> str:
    """Send one JSON-constrained request to Ollama and return its text response."""
    selected_model = model or local_model_name()
    payload: dict[str, Any] = {
        "model": selected_model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": _generation_options(),
    }
    keep_alive = os.getenv("OLLAMA_KEEP_ALIVE")
    if keep_alive:
        payload["keep_alive"] = keep_alive

    timeout = float(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "900"))
    response = requests.post(
        f"{_ollama_host()}/api/generate",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if not data.get("response"):
        raise RuntimeError("Ollama returned no response text.")
    return data["response"]


def ollama_status(model: str | None = None, timeout: float = 3.0) -> dict[str, Any]:
    """Read-only readiness check used before an end-to-end screening run."""
    selected_model = model or local_model_name()
    try:
        response = requests.get(f"{_ollama_host()}/api/tags", timeout=timeout)
        response.raise_for_status()
        models = response.json().get("models", [])
        installed = [item.get("name", "") for item in models]
        return {
            "reachable": True,
            "host": _ollama_host(),
            "selected_model": selected_model,
            "model_installed": selected_model in installed,
            "installed_models": installed,
        }
    except requests.RequestException as exc:
        return {
            "reachable": False,
            "host": _ollama_host(),
            "selected_model": selected_model,
            "model_installed": False,
            "installed_models": [],
            "error": str(exc),
        }
