from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path


GEMINI_URL = "https://gemini.google.com/app"


@dataclass(frozen=True)
class GeminiWebConfig:
    profile_dir: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_WEB_PROFILE_DIR",
            os.path.join("browser_profiles", "gemini"),
        )
    )
    headless: bool = False
    ready_timeout_ms: int = 120_000
    response_timeout_ms: int = 180_000


class GeminiWebAutomation:
    def __init__(self, config: GeminiWebConfig | None = None):
        self.config = config or GeminiWebConfig()
        self._playwright = None
        self._context = None
        self._page = None

    def __enter__(self) -> "GeminiWebAutomation":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Gemini Web Automation requires Playwright. Install it with "
                "`pip install playwright` and run `playwright install chromium`."
            ) from exc

        Path(self.config.profile_dir).mkdir(parents=True, exist_ok=True)
        self._playwright = sync_playwright().start()
        self._context = self._playwright.chromium.launch_persistent_context(
            user_data_dir=self.config.profile_dir,
            headless=self.config.headless,
            viewport={"width": 1400, "height": 900},
            args=["--disable-blink-features=AutomationControlled"],
        )
        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        self._page.goto(GEMINI_URL, wait_until="domcontentloaded")
        self.wait_until_ready()

    def close(self) -> None:
        if self._context is not None:
            self._context.close()
            self._context = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None
        self._page = None

    def wait_until_ready(self) -> None:
        page = self._require_page()
        deadline = time.monotonic() + (self.config.ready_timeout_ms / 1000)
        last_error = None

        while time.monotonic() < deadline:
            try:
                if self._find_prompt_box() is not None:
                    return
            except Exception as exc:
                last_error = exc
            time.sleep(1)

        raise RuntimeError(
            "Gemini did not become ready. Open the persistent browser profile, "
            "finish Google/Gemini login, then retry. "
            f"Profile directory: {Path(self.config.profile_dir).resolve()}. "
            f"Last error: {last_error}"
        )

    def submit_prompt_and_get_response(self, prompt: str) -> str:
        page = self._require_page()
        before_text = self._latest_response_text()
        box = self._find_prompt_box()
        if box is None:
            page.reload(wait_until="domcontentloaded")
            self.wait_until_ready()
            box = self._find_prompt_box()
        if box is None:
            raise RuntimeError("Could not find Gemini prompt input.")

        box.click()
        box.fill(prompt)
        self._submit_prompt()
        return self._wait_for_new_response(before_text)

    def _require_page(self):
        if self._page is None:
            raise RuntimeError("Gemini browser is not started.")
        return self._page

    def _find_prompt_box(self):
        page = self._require_page()
        selectors = [
            "div[contenteditable='true'][role='textbox']",
            "rich-textarea div[contenteditable='true']",
            "textarea",
            "[role='textbox']",
        ]
        for selector in selectors:
            locator = page.locator(selector).last
            try:
                if locator.count() and locator.is_visible() and locator.is_enabled():
                    return locator
            except Exception:
                continue
        return None

    def _submit_prompt(self) -> None:
        page = self._require_page()
        submit_selectors = [
            "button[aria-label*='Send']",
            "button[aria-label*='Submit']",
            "button:has(mat-icon:has-text('send'))",
        ]
        for selector in submit_selectors:
            locator = page.locator(selector).last
            try:
                if locator.count() and locator.is_visible() and locator.is_enabled():
                    locator.click()
                    return
            except Exception:
                continue
        page.keyboard.press("Control+Enter")

    def _wait_for_new_response(self, before_text: str) -> str:
        deadline = time.monotonic() + (self.config.response_timeout_ms / 1000)
        stable_since = None
        last_text = ""

        while time.monotonic() < deadline:
            text = self._latest_response_text()
            if text and text != before_text:
                if text == last_text:
                    stable_since = stable_since or time.monotonic()
                    if time.monotonic() - stable_since >= 3:
                        return text
                else:
                    stable_since = None
                    last_text = text
            time.sleep(1)

        raise TimeoutError("Timed out waiting for Gemini response.")

    def _latest_response_text(self) -> str:
        page = self._require_page()
        selectors = [
            "message-content",
            ".model-response-text",
            "[data-response-index]",
            "div.markdown",
        ]
        for selector in selectors:
            locator = page.locator(selector)
            try:
                count = locator.count()
                if count:
                    text = locator.nth(count - 1).inner_text(timeout=2_000).strip()
                    if text:
                        return text
            except Exception:
                continue
        return ""
