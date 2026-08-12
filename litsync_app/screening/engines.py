from __future__ import annotations


LOCAL_ENGINE = "local"
GEMINI_WEB_ENGINE = "gemini_web"
DEFAULT_PROCESSING_ENGINE = LOCAL_ENGINE
SUPPORTED_PROCESSING_ENGINES = {LOCAL_ENGINE, GEMINI_WEB_ENGINE}


def normalize_processing_engine(engine: str | None) -> str:
    """Return one of the two public AI engines, defaulting safely to Local AI."""
    value = str(engine or LOCAL_ENGINE).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ollama": LOCAL_ENGINE,
        "gemini": GEMINI_WEB_ENGINE,
        "online": GEMINI_WEB_ENGINE,
    }
    value = aliases.get(value, value)
    return value if value in SUPPORTED_PROCESSING_ENGINES else LOCAL_ENGINE
