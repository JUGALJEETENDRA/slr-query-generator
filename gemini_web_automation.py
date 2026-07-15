from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple


GEMINI_URL = "https://gemini.google.com/app"


class _ResponseSnapshot(NamedTuple):
    selector: str
    count: int
    text: str


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
        self._submission_count = 0

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
        try:
            self._context = self._playwright.chromium.launch_persistent_context(
                user_data_dir=self.config.profile_dir,
                headless=self.config.headless,
                viewport={"width": 1400, "height": 900},
                args=["--disable-blink-features=AutomationControlled"],
            )
            self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
            self.start_new_job_chat()
        except Exception as exc:
            message = str(exc).lower()
            self.close()
            if "executable doesn't exist" in message or "browser executable" in message:
                raise RuntimeError(
                    "Gemini Web needs its Chromium browser. Run `python -m playwright install chromium` once, then retry."
                ) from exc
            if "processsingleton" in message or "profile" in message and "lock" in message:
                raise RuntimeError(
                    "The dedicated LitSync Gemini browser is already open. Close that browser window, then retry."
                ) from exc
            if "gemini did not become ready" in message:
                raise RuntimeError(str(exc)) from exc
            raise RuntimeError(
                "Could not open the dedicated Gemini Web browser. Check the browser window and internet connection, then retry."
            ) from exc

    def close(self) -> None:
        if self._context is not None:
            self._context.close()
            self._context = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None
        self._page = None
        self._submission_count = 0

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
            "Gemini did not become ready within two minutes. In the browser window that LitSync opens, "
            "sign in to Google, accept any Gemini welcome or consent screen, and wait until the message box appears. "
            "Then retry Screen Papers; the dedicated login is saved for future runs. "
            f"Browser profile: {Path(self.config.profile_dir).resolve()}."
        )

    def submit_prompt_and_get_response(self, prompt: str) -> str:
        page = self._require_page()
        before = self._response_snapshot()
        box = self._find_prompt_box()
        if box is None:
            page.reload(wait_until="domcontentloaded")
            self.wait_until_ready()
            box = self._find_prompt_box()
        if box is None:
            raise RuntimeError("Could not find Gemini prompt input.")

        box.click()
        box.fill(
            prompt
            + "\n\nWEB AUTOMATION OUTPUT RULE: Return exactly one JSON object matching the requested schema. "
              "Do not use Markdown fences, headings, commentary, or text before or after the JSON."
        )
        self._submit_prompt()
        response = self._wait_for_new_response(before)
        self._submission_count += 1
        return response

    def start_new_job_chat(self) -> None:
        """Open one clean chat for a screening job, not for every request."""
        page = self._require_page()
        page.goto(GEMINI_URL, wait_until="domcontentloaded")
        self.wait_until_ready()
        self._submission_count = 0

    def recover_job_chat(self) -> None:
        """Recover after a broken page; batch prompts carry the full protocol."""
        self.start_new_job_chat()

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
            "button[data-test-id='send-button']",
            "button[aria-label='Send message']",
            "button[aria-label*='Send']",
            "button[aria-label*='Submit']",
            "button.send-button",
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
        # Gemini's composer submits with Enter; Shift+Enter is the newline action.
        page.keyboard.press("Enter")

    def _wait_for_new_response(self, before: _ResponseSnapshot) -> str:
        deadline = time.monotonic() + (self.config.response_timeout_ms / 1000)
        stable_since = None
        last_text = ""

        while time.monotonic() < deadline:
            current = self._response_snapshot(preferred_selector=before.selector)
            text = current.text
            is_new = bool(text) and (
                current.count > before.count
                or (current.count == before.count and text != before.text)
                or (current.selector != before.selector and text != before.text)
            )
            if is_new:
                if text == last_text:
                    stable_since = stable_since or time.monotonic()
                    if time.monotonic() - stable_since >= 4 and not self._is_generating():
                        return text
                else:
                    stable_since = None
                    last_text = text
            time.sleep(1)

        raise TimeoutError(
            "Gemini Web did not produce a new completed response in time. "
            "Check the opened Gemini window for login, consent, quota, or network messages."
        )

    def _latest_response_text(self) -> str:
        return self._response_snapshot().text

    def _response_snapshot(self, preferred_selector: str = "") -> _ResponseSnapshot:
        page = self._require_page()
        selectors = [
            preferred_selector,
            "model-response",
            "message-content",
            "[data-test-id='model-response']",
            ".model-response-text",
            "[data-response-index]",
            "div.markdown.markdown-main-panel",
        ]
        for selector in selectors:
            if not selector:
                continue
            locator = page.locator(selector)
            try:
                count = locator.count()
                if count:
                    latest = locator.nth(count - 1)
                    if not latest.is_visible():
                        continue
                    text = latest.inner_text(timeout=2_000).strip()
                    if text:
                        return _ResponseSnapshot(selector, count, text)
            except Exception:
                continue
        return _ResponseSnapshot("", 0, "")

    def _is_generating(self) -> bool:
        page = self._require_page()
        selectors = [
            "button[aria-label*='Stop response']",
            "button[aria-label*='Stop generating']",
            "button[data-test-id='stop-button']",
            "button.stop-button",
        ]
        for selector in selectors:
            try:
                locator = page.locator(selector)
                if locator.count() and locator.last.is_visible():
                    return True
            except Exception:
                continue
        return False
