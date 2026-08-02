from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


GEMINI_URL = "https://gemini.google.com/app"


class GeminiWebFastTransportError(RuntimeError):
    def __init__(self, stage: str, cause: Exception, active_tabs: int) -> None:
        self.stage = stage
        self.exception_type = type(cause).__name__
        self.active_tabs = max(0, int(active_tabs))
        super().__init__(f"Gemini Web Fast transport failed during {stage} ({self.exception_type}).")


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class GeminiWebFastBrowserConfig:
    profile_dir: str = field(default_factory=lambda: os.getenv(
        "GEMINI_WEB_PROFILE_DIR", os.path.join("browser_profiles", "gemini"),
    ))
    headless: bool = False
    ready_timeout_ms: int = field(default_factory=lambda: _bounded_int(
        "GEMINI_WEB_FAST_READY_TIMEOUT_MS", 60_000, 10_000, 120_000,
    ))
    response_timeout_ms: int = field(default_factory=lambda: _bounded_int(
        "GEMINI_WEB_FAST_RESPONSE_TIMEOUT_MS", 120_000, 30_000, 240_000,
    ))
    poll_interval_ms: int = 400
    stable_ms: int = 700


class GeminiWebFastBrowser:
    """One persistent async context; every request uses and closes a fresh page."""

    def __init__(
        self,
        config: GeminiWebFastBrowserConfig | None = None,
        *,
        playwright_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config or GeminiWebFastBrowserConfig()
        self._playwright_factory = playwright_factory
        self._playwright = None
        self._context = None
        self._anchor_page = None
        self.active_pages = 0
        self.peak_active_pages = 0
        self.pages_opened = 0
        self.pages_closed = 0
        self.transport_events: list[dict[str, Any]] = []
        self.activity_callback: Callable[[int], None] | None = None

    async def start(self) -> None:
        Path(self.config.profile_dir).mkdir(parents=True, exist_ok=True)
        if self._playwright_factory is None:
            try:
                from playwright.async_api import async_playwright
            except ImportError as exc:
                raise RuntimeError("Gemini Web Fast requires Playwright Chromium.") from exc
            self._playwright_factory = async_playwright
        self._playwright = await self._playwright_factory().start()
        try:
            self._context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=self.config.profile_dir,
                headless=self.config.headless,
                viewport={"width": 1400, "height": 900},
                args=["--disable-blink-features=AutomationControlled"],
            )
            pages = list(self._context.pages)
            if pages:
                self._anchor_page = pages[0]
                await self._anchor_page.goto(
                    "about:blank",
                    wait_until="commit",
                    timeout=min(self.config.ready_timeout_ms, 10_000),
                )
            else:
                self._anchor_page = await self._context.new_page()
            for page in pages[1:]:
                await page.close()
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        context, playwright = self._context, self._playwright
        self._context = None
        self._playwright = None
        self._anchor_page = None
        if context is not None:
            try:
                await context.close()
            finally:
                self.active_pages = 0
        if playwright is not None:
            await playwright.stop()

    async def submit_fresh(self, prompt: str, *, timeout_seconds: float | None = None) -> str:
        if self._context is None:
            raise RuntimeError("Gemini Web Fast browser is not started.")
        page = None
        failure: Exception | None = None
        stage = "page_creation"
        try:
            page = await self._context.new_page()
            self.pages_opened += 1
            self.active_pages += 1
            self.peak_active_pages = max(self.peak_active_pages, self.active_pages)
            if self.activity_callback:
                self.activity_callback(self.active_pages)
            timeout_ms = self.config.response_timeout_ms
            if timeout_seconds is not None:
                timeout_ms = min(timeout_ms, max(1, int(timeout_seconds * 1000)))
            page.set_default_timeout(min(self.config.ready_timeout_ms, timeout_ms))
            stage = "navigation_readiness"
            await page.goto(GEMINI_URL, wait_until="domcontentloaded", timeout=timeout_ms)
            box = await self._find_prompt_box(page, min(timeout_ms, self.config.ready_timeout_ms))
            if box is None:
                body = (await page.locator("body").inner_text(timeout=2_000)).casefold()
                if any(marker in body for marker in ("usage limit", "quota", "rate limit")):
                    stage = "quota_rate_limiting"
                    raise RuntimeError("Gemini quota or usage limit prevented submission.")
                if any(marker in body for marker in ("sign in", "consent")):
                    raise RuntimeError("Gemini login, consent, or quota screen prevented submission.")
                raise RuntimeError("Gemini prompt composer was not available.")
            stage = "response_baseline"
            before = await self._response_texts(page)
            stage = "prompt_submission"
            await box.click()
            await box.fill(
                prompt
                + "\n\nWEB AUTOMATION OUTPUT RULE: Return exactly one JSON object. "
                  "Do not use Markdown fences, headings, commentary, or surrounding text."
            )
            await self._submit(page)
            stage = "response_waiting"
            try:
                return await self._wait_for_response(page, before, timeout_ms)
            except TimeoutError as exc:
                if "complete JSON" in str(exc):
                    stage = "response_capture"
                raise
        except Exception as exc:
            if isinstance(exc, GeminiWebFastTransportError):
                failure = exc
            else:
                failure = GeminiWebFastTransportError(stage, exc, self.active_pages)
                self.transport_events.append({
                    "failure_stage": failure.stage,
                    "exception_type": failure.exception_type,
                    "active_tabs": failure.active_tabs,
                })
            raise failure from exc
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception as exc:
                    if failure is None:
                        closure_error = GeminiWebFastTransportError(
                            "page_closure", exc, self.active_pages,
                        )
                        self.transport_events.append({
                            "failure_stage": closure_error.stage,
                            "exception_type": closure_error.exception_type,
                            "active_tabs": closure_error.active_tabs,
                        })
                        raise closure_error from exc
                finally:
                    self.pages_closed += 1
                self.active_pages = max(0, self.active_pages - 1)
                if self.activity_callback:
                    self.activity_callback(self.active_pages)

    async def _find_prompt_box(self, page, timeout_ms: int):
        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            for selector in (
                "div[contenteditable='true'][role='textbox']",
                "rich-textarea div[contenteditable='true']",
                "textarea",
                "[role='textbox']",
            ):
                locator = page.locator(selector).last
                try:
                    if await locator.count() and await locator.is_visible() and await locator.is_enabled():
                        return locator
                except Exception:
                    continue
            await asyncio.sleep(min(.25, max(0, deadline - time.monotonic())))
        return None

    async def _submit(self, page) -> None:
        for selector in (
            "button[data-test-id='send-button']",
            "button[aria-label='Send message']",
            "button[aria-label*='Send']",
            "button[aria-label*='Submit']",
            "button.send-button",
        ):
            locator = page.locator(selector).last
            try:
                if await locator.count() and await locator.is_visible() and await locator.is_enabled():
                    await locator.click()
                    return
            except Exception:
                continue
        await page.keyboard.press("Enter")

    async def _response_texts(self, page) -> tuple[str, ...]:
        values: list[str] = []
        for selector in (
            "model-response",
            "[data-test-id='model-response']",
            "[data-message-author-role='model']",
            "[data-author='model']",
            ".model-response-text",
            "div.markdown.markdown-main-panel",
            "message-content",
        ):
            locator = page.locator(selector)
            try:
                for index in range(await locator.count()):
                    candidate = locator.nth(index)
                    if await candidate.is_visible():
                        text = (await candidate.inner_text(timeout=2_000)).strip()
                        if text:
                            values.append(text)
            except Exception:
                continue
        return tuple(values)

    async def _wait_for_response(self, page, before: tuple[str, ...], timeout_ms: int) -> str:
        deadline = time.monotonic() + timeout_ms / 1000
        last = ""
        stable_since: float | None = None
        while time.monotonic() < deadline:
            current = await self._response_texts(page)
            new_values = current[len(before):] if current[:len(before)] == before else tuple(
                value for value in current if value not in before
            )
            candidate = new_values[-1] if new_values else ""
            if candidate and candidate == last:
                stable_since = stable_since or time.monotonic()
                if self._complete_json(candidate) and (time.monotonic() - stable_since) * 1000 >= self.config.stable_ms:
                    return candidate
            else:
                last, stable_since = candidate, None
            await asyncio.sleep(self.config.poll_interval_ms / 1000)
        if last:
            raise TimeoutError("Gemini Web Fast returned content but not a complete JSON response in time.")
        raise TimeoutError("Gemini Web Fast did not return a response in time.")

    @staticmethod
    def _complete_json(value: str) -> bool:
        text = value.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.I | re.S)
        if fenced:
            text = fenced.group(1).strip()
        try:
            return isinstance(json.loads(text), dict)
        except (TypeError, json.JSONDecodeError):
            return False
