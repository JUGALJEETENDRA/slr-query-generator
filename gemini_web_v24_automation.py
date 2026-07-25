from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from gemini_web_automation import GeminiWebAutomation, GeminiWebConfig


def _bounded(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class GeminiWebV24Config:
    diagnostic_sink: Callable[[dict[str, Any]], None] | None = None
    max_chat_submissions: int = field(
        default_factory=lambda: _bounded("GEMINI_WEB_V24_MAX_CHAT_SUBMISSIONS", 5, 1, 50)
    )
    max_browser_submissions: int = field(
        default_factory=lambda: _bounded("GEMINI_WEB_V24_MAX_BROWSER_SUBMISSIONS", 10, 1, 100)
    )
    recovery_backoff_ms: int = field(
        default_factory=lambda: _bounded("GEMINI_WEB_V24_RECOVERY_BACKOFF_MS", 2000, 0, 10000)
    )

    def transport_config(self) -> GeminiWebConfig:
        return GeminiWebConfig(
            max_chat_submissions=self.max_chat_submissions,
            max_browser_submissions=self.max_browser_submissions,
            recovery_backoff_ms=self.recovery_backoff_ms,
            diagnostic_sink=self.diagnostic_sink,
        )


class GeminiWebV24Automation(GeminiWebAutomation):
    def __init__(self, config: GeminiWebV24Config | None = None):
        self.v24_config = config or GeminiWebV24Config()
        super().__init__(self.v24_config.transport_config())
