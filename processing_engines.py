from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ollama_client import ask_ollama

try:
    from gemini_web_automation import GeminiWebConfig
except Exception:
    GeminiWebConfig = None


LOCAL_ENGINE = "local"
GEMINI_API_ENGINE = "gemini_api"
GEMINI_WEB_ENGINE = "gemini_web"
DEFAULT_PROCESSING_ENGINE = LOCAL_ENGINE
SUPPORTED_PROCESSING_ENGINES = {
    LOCAL_ENGINE,
    GEMINI_API_ENGINE,
    GEMINI_WEB_ENGINE,
}


class InferenceEngine(Protocol):
    engine_id: str

    def ask(self, prompt: str, model: str = "qwen2.5:3b") -> str:
        ...

    def __enter__(self) -> "InferenceEngine":
        ...

    def __exit__(self, exc_type, exc, tb) -> None:
        ...


def normalize_processing_engine(engine: str | None) -> str:
    value = str(engine or DEFAULT_PROCESSING_ENGINE).strip().lower()
    value = value.replace("-", "_").replace(" ", "_")
    aliases = {
        "ollama": LOCAL_ENGINE,
        "qwen": LOCAL_ENGINE,
        "online": GEMINI_API_ENGINE,
        "gemini": GEMINI_API_ENGINE,
        "gemini_api": GEMINI_API_ENGINE,
        "google_gemini_api": GEMINI_API_ENGINE,
        "gemini_web": GEMINI_WEB_ENGINE,
        "gemini_web_automation": GEMINI_WEB_ENGINE,
        "web_gemini": GEMINI_WEB_ENGINE,
    }
    normalized = aliases.get(value, value)
    if normalized not in SUPPORTED_PROCESSING_ENGINES:
        return DEFAULT_PROCESSING_ENGINE
    return normalized


def resolve_processing_engine(
    engine: str | None,
    *,
    gemini_web_config=None,
    gemini_api_key: str | None = None,
) -> InferenceEngine:
    normalized = normalize_processing_engine(engine)
    if normalized == GEMINI_API_ENGINE:
        return GeminiApiEngine(api_key=gemini_api_key)
    if normalized == GEMINI_WEB_ENGINE:
        return GeminiWebInferenceEngine(config=gemini_web_config)
    return LocalInferenceEngine()


@dataclass
class LocalInferenceEngine:
    engine_id: str = LOCAL_ENGINE

    def ask(self, prompt: str, model: str = "qwen2.5:3b") -> str:
        return ask_ollama(prompt, model=model)

    def __enter__(self) -> "LocalInferenceEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


@dataclass
class GeminiApiEngine:
    engine_id: str = GEMINI_API_ENGINE
    api_key: str | None = None

    def ask(self, prompt: str, model: str = "qwen2.5:3b") -> str:
        from gemini_client import ask_gemini

        gemini_model = model if str(model or "").startswith("gemini") else "gemini-2.5-flash"
        return ask_gemini(prompt, model=gemini_model, api_key=self.api_key)

    def __enter__(self) -> "GeminiApiEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class GeminiWebInferenceEngine:
    engine_id = GEMINI_WEB_ENGINE

    def __init__(self, config=None):
        if config is None:
            from gemini_web_automation import GeminiWebConfig as RuntimeGeminiWebConfig

            config = RuntimeGeminiWebConfig()
        self.config = config
        self._automation: GeminiWebAutomation | None = None

    def __enter__(self) -> "GeminiWebInferenceEngine":
        from gemini_web_automation import GeminiWebAutomation

        self._automation = GeminiWebAutomation(self.config)
        self._automation.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._automation is not None:
            self._automation.__exit__(exc_type, exc, tb)
            self._automation = None

    def ask(self, prompt: str, model: str = "qwen2.5:3b") -> str:
        if self._automation is None:
            from gemini_web_automation import GeminiWebAutomation

            with GeminiWebAutomation(self.config) as browser:
                return browser.submit_prompt_and_get_response(prompt)
        return self._automation.submit_prompt_and_get_response(prompt)
