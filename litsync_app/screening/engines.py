from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from litsync_app.config import ENABLE_EXTERNAL_ENGINES
from litsync_app.screening.ollama import ask_ollama


LOCAL_ENGINE = "local"
GEMINI_API_ENGINE = "gemini_api"
GEMINI_WEB_V24_ENGINE = "gemini_web_v24"
DEFAULT_PROCESSING_ENGINE = LOCAL_ENGINE
SUPPORTED_PROCESSING_ENGINES = {
    LOCAL_ENGINE, GEMINI_API_ENGINE, GEMINI_WEB_V24_ENGINE,
}


class InferenceEngine(Protocol):
    engine_id: str
    def ask(self, prompt: str, model: str = "qwen3:8b") -> str: ...
    def ask_structured(self, prompt: str, schema, model: str = "") -> str: ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc, tb): ...


def normalize_processing_engine(engine: str | None) -> str:
    value = str(engine or LOCAL_ENGINE).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ollama": LOCAL_ENGINE,
        "gemini": GEMINI_API_ENGINE,
        "online": GEMINI_API_ENGINE,
        "gemini_web_v2_4": GEMINI_WEB_V24_ENGINE,
        "gemini_web_v2.4": GEMINI_WEB_V24_ENGINE,
    }
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
        if not ENABLE_EXTERNAL_ENGINES and not self.options.get("explicit_opt_in"):
            raise RuntimeError("External AI engines are disabled; set ENABLE_EXTERNAL_ENGINES=true explicitly.")
        if self.engine_id == GEMINI_API_ENGINE:
            from litsync_app.integrations.gemini_api import GeminiAPIClient
            self._client = GeminiAPIClient(api_key=self.options.get("gemini_api_key"))
            self._delegate = lambda prompt, model: self._client.generate(prompt, model=model)
        elif self.engine_id == GEMINI_WEB_V24_ENGINE:
            from litsync_app.integrations.gemini_web_v24_automation import GeminiWebV24Automation, GeminiWebV24Config
            browser = GeminiWebV24Automation(
                self.options.get("gemini_web_v24_config") or GeminiWebV24Config()
            )
            browser.__enter__()
            self._browser = browser
            self._delegate = lambda prompt, model: browser.submit_prompt_and_get_response(prompt)
        else:
            raise ValueError(f"Unsupported external engine: {self.engine_id}")
        return self
    def ask(self, prompt: str, model: str = "") -> str:
        if self._delegate is None:
            raise RuntimeError("External engine must be used as a context manager")
        return self._delegate(prompt, model)
    def ask_structured(self, prompt: str, schema, model: str = "") -> str:
        if self._delegate is None:
            raise RuntimeError("External engine must be used as a context manager")
        if self.engine_id == GEMINI_API_ENGINE:
            return self._client.generate(prompt, model=model, schema=schema)
        return self._delegate(prompt, model)
    def __exit__(self, exc_type, exc, tb):
        if getattr(self, "_browser", None):
            self._browser.__exit__(exc_type, exc, tb)
        if getattr(self, "_client", None):
            self._client.close()


def resolve_processing_engine(engine: str | None, **options):
    normalized = normalize_processing_engine(engine)
    if normalized == LOCAL_ENGINE:
        return LocalInferenceEngine()
    return _LazyExternalEngine(normalized, **options)
