from __future__ import annotations

import asyncio

import pytest

from litsync_app.integrations.gemini_web_fast_browser import (
    GeminiWebFastBrowser,
    GeminiWebFastTransportError,
)


class Locator:
    def __init__(self, page, kind): self.page, self.kind = page, kind
    @property
    def last(self): return self
    def nth(self, index): return self
    async def count(self): return 1 if self.kind in {"box", "response", "send"} else 0
    async def is_visible(self): return True
    async def is_enabled(self): return True
    async def click(self): pass
    async def fill(self, value): self.page.prompt = value
    async def inner_text(self, timeout=None):
        return '{"items": []}' if self.kind == "response" and self.page.submitted else ""


class Keyboard:
    def __init__(self, page): self.page = page
    async def press(self, key): self.page.submitted = True


class Page:
    def __init__(self):
        self.prompt = ""; self.submitted = False; self.closed = 0; self.keyboard = Keyboard(self)
    def set_default_timeout(self, value): self.timeout = value
    async def goto(self, *args, **kwargs): self.goto_args = (args, kwargs)
    def locator(self, selector):
        if "contenteditable" in selector: return Locator(self, "box")
        if "send" in selector.casefold(): return Locator(self, "none")
        if selector == "model-response": return Locator(self, "response")
        return Locator(self, "none")
    async def close(self): self.closed += 1


class Context:
    def __init__(self): self.pages = []; self.created = []; self.closed = 0
    async def new_page(self):
        page = Page(); self.created.append(page); return page
    async def close(self): self.closed += 1


def test_start_preserves_one_blank_anchor_page_for_persistent_context():
    anchor, restored = Page(), Page()
    context = Context()
    context.pages = [anchor, restored]

    class Chromium:
        async def launch_persistent_context(self, **kwargs):
            return context

    class Playwright:
        chromium = Chromium()
        async def stop(self): pass

    class Factory:
        async def start(self): return Playwright()

    browser = GeminiWebFastBrowser(playwright_factory=Factory)

    async def run():
        await browser.start()
        assert browser._anchor_page is anchor
        assert anchor.goto_args[0] == ("about:blank",)
        assert anchor.closed == 0
        assert restored.closed == 1
        batch_page = await context.new_page()
        assert batch_page is not anchor
        await browser.close()

    asyncio.run(run())
    assert context.closed == 1


def test_each_submission_uses_and_closes_a_fresh_page(monkeypatch):
    browser = GeminiWebFastBrowser()
    context = Context()
    browser._context = context
    monkeypatch.setattr(browser, "_wait_for_response", lambda page, before, timeout: asyncio.sleep(0, result='{"items": []}'))
    async def run():
        return (
            await browser.submit_fresh("one", timeout_seconds=1),
            await browser.submit_fresh("two", timeout_seconds=1),
        )
    first, second = asyncio.run(run())
    assert first == second == '{"items": []}'
    assert len(context.created) == 2
    assert context.created[0] is not context.created[1]
    assert all(page.closed == 1 for page in context.created)
    assert browser.pages_opened == browser.pages_closed == 2
    assert browser.active_pages == 0
    assert all(page.prompt.count("WEB AUTOMATION OUTPUT RULE") == 1 for page in context.created)


def test_page_closes_when_submission_fails(monkeypatch):
    browser = GeminiWebFastBrowser()
    context = Context()
    browser._context = context
    async def fail(*args): raise TimeoutError("no response")
    monkeypatch.setattr(browser, "_wait_for_response", fail)
    with pytest.raises(GeminiWebFastTransportError) as captured:
        asyncio.run(browser.submit_fresh("prompt", timeout_seconds=1))
    assert captured.value.stage == "response_waiting"
    assert captured.value.exception_type == "TimeoutError"
    assert context.created[0].closed == 1
    assert browser.active_pages == 0
    assert browser.pages_closed == 1


def test_browser_uses_async_playwright_and_dedicated_profile_source():
    import inspect
    import litsync_app.integrations.gemini_web_fast_browser as module
    source = inspect.getsource(module)
    assert "playwright.async_api" in source
    assert 'browser_profiles", "gemini' in source
    assert "GeminiWebV24" not in source


def test_prompt_composer_readiness_is_bounded_and_allows_ui_settlement(monkeypatch):
    browser = GeminiWebFastBrowser()
    page = Page()
    calls = {"count": 0}
    original = page.locator
    def delayed(selector):
        locator = original(selector)
        if "contenteditable" in selector:
            async def count():
                calls["count"] += 1
                return 1 if calls["count"] >= 2 else 0
            locator.count = count
        return locator
    page.locator = delayed
    assert asyncio.run(browser._find_prompt_box(page, 1000)) is not None
    assert calls["count"] >= 2
