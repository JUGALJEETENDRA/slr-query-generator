from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from config import ENABLE_EXTERNAL_ENGINES
from ollama_client import ask_ollama


LOCAL_ENGINE = "local"
GEMINI_API_ENGINE = "gemini_api"
GEMINI_WEB_ENGINE = "gemini_web"
DEFAULT_PROCESSING_ENGINE = LOCAL_ENGINE
SUPPORTED_PROCESSING_ENGINES = {LOCAL_ENGINE, GEMINI_API_ENGINE, GEMINI_WEB_ENGINE}


class InferenceEngine(Protocol):
    engine_id: str
    def ask(self, prompt: str, model: str = "qwen3:8b") -> str: ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc, tb): ...


def normalize_processing_engine(engine: str | None) -> str:
    value = str(engine or LOCAL_ENGINE).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"ollama": LOCAL_ENGINE, "gemini": GEMINI_API_ENGINE, "online": GEMINI_API_ENGINE}
    value = aliases.get(value, value)
    return value if value in SUPPORTED_PROCESSING_ENGINES else LOCAL_ENGINE


@dataclass
class LocalInferenceEngine:
    engine_id: str = LOCAL_ENGINE
    def ask(self, prompt: str, model: str = "qwen3:8b") -> str:
        return ask_ollama(prompt, model)
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return None


class _LazyExternalEngine:
    def __init__(self, engine_id: str, **options):
        self.engine_id = engine_id
        self.options = options
        self._delegate = None
    def __enter__(self):
        if not ENABLE_EXTERNAL_ENGINES:
            raise RuntimeError("External AI engines are disabled; set ENABLE_EXTERNAL_ENGINES=true explicitly.")
        if self.engine_id == GEMINI_API_ENGINE:
            from gemini_client import ask_gemini
            self._delegate = lambda prompt, model: ask_gemini(
                prompt, model=model if str(model).startswith("gemini") else "gemini-2.5-flash",
                api_key=self.options.get("gemini_api_key"),
            )
        else:
            from gemini_web_automation import GeminiWebAutomation, GeminiWebConfig
            browser = GeminiWebAutomation(self.options.get("gemini_web_config") or GeminiWebConfig())
            browser.__enter__()
            self._browser = browser
            self._delegate = lambda prompt, model: browser.submit_prompt_and_get_response(prompt)
        return self
    def ask(self, prompt: str, model: str = "") -> str:
        if self._delegate is None:
            raise RuntimeError("External engine must be used as a context manager")
        return self._delegate(prompt, model)
    def __exit__(self, exc_type, exc, tb):
        if getattr(self, "_browser", None):
            self._browser.__exit__(exc_type, exc, tb)


def resolve_processing_engine(engine: str | None, **options):
    normalized = normalize_processing_engine(engine)
    if normalized == LOCAL_ENGINE:
        return LocalInferenceEngine()
    return _LazyExternalEngine(normalized, **options)
