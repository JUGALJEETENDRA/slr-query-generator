"""Gemini website automation using a reusable Playwright browser profile."""

from __future__ import annotations

import os
import time
from pathlib import Path


class GeminiBrowserError(RuntimeError):
    pass


class GeminiBrowser:
    def __init__(
        self,
        profile_dir: str | None = None,
        headless: bool = False,
        timeout_ms: int = 240_000,
    ) -> None:
        default_profile = Path.home() / ".slr-query-generator" / "gemini-profile"
        self.profile_dir = str(Path(profile_dir or os.getenv("GEMINI_PROFILE_DIR", default_profile)))
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._playwright = None
        self.context = None
        self.page = None

    def __enter__(self):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise GeminiBrowserError(
                "Playwright is not installed. Run: pip install -r requirements.txt, "
                "then: playwright install chromium"
            ) from exc

        self._playwright = sync_playwright().start()
        try:
            self.context = self._playwright.chromium.launch_persistent_context(
                self.profile_dir,
                headless=self.headless,
                viewport={"width": 1400, "height": 900},
            )
            self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
            self.page.set_default_timeout(self.timeout_ms)
            self.page.goto("https://gemini.google.com/app", wait_until="domcontentloaded")
            self._wait_for_prompt_box()
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc, traceback):
        if self.context is not None:
            self.context.close()
            self.context = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def _prompt_box(self):
        selectors = [
            "div[contenteditable='true'][role='textbox']",
            "rich-textarea div[contenteditable='true']",
            ".ql-editor[contenteditable='true']",
            "textarea",
        ]
        for selector in selectors:
            locator = self.page.locator(selector)
            if locator.count() and locator.last.is_visible():
                return locator.last
        return None

    def _wait_for_prompt_box(self):
        # Do not use page.wait_for_function here. Gemini enforces Trusted Types,
        # which prevents Playwright from evaluating a JavaScript string.
        deadline = time.monotonic() + (self.timeout_ms / 1000)
        last_error = None
        while time.monotonic() < deadline:
            try:
                if self._prompt_box() is not None:
                    return
            except Exception as exc:
                last_error = exc
            self.page.wait_for_timeout(500)
        raise GeminiBrowserError(
            "Gemini's prompt box was not found. Complete Google sign-in in the opened "
            "browser, then run Hybrid screening again."
        ) from last_error

    def submit(self, prompt: str) -> str:
        """Submit one batch and return the newest model response text."""
        response_selector = "message-content, [data-message-author-role='model']"
        previous_count = self.page.locator(response_selector).count()
        box = self._prompt_box()
        if box is None:
            self._wait_for_prompt_box()
            box = self._prompt_box()

        box.click()
        box.fill(prompt)
        send_button = self.page.locator(
            "button[aria-label*='Send'], button[aria-label*='send'], button.send-button"
        )
        if send_button.count() and send_button.last.is_enabled():
            send_button.last.click()
        else:
            box.press("Enter")

        try:
            deadline = time.monotonic() + (self.timeout_ms / 1000)
            responses = self.page.locator(response_selector)
            while time.monotonic() < deadline and responses.count() <= previous_count:
                self.page.wait_for_timeout(500)
            if responses.count() <= previous_count:
                raise GeminiBrowserError("Gemini did not create a response message.")

            newest = responses.last
            # Gemini streams into the newest response. Consider it complete once its
            # text has stayed unchanged across five checks.
            last_text = ""
            stable_checks = 0
            while time.monotonic() < deadline:
                self.page.wait_for_timeout(1000)
                current = newest.inner_text().strip()
                if current and current == last_text:
                    stable_checks += 1
                    if stable_checks >= 5:
                        return current
                else:
                    stable_checks = 0
                    last_text = current
        except GeminiBrowserError:
            raise
        except Exception as exc:
            raise GeminiBrowserError("Timed out while waiting for Gemini's response.") from exc
        raise GeminiBrowserError("Gemini's response did not finish before the timeout.")
