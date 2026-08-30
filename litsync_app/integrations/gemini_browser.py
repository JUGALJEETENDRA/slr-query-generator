from __future__ import annotations

import json
import os
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple


GEMINI_URL = "https://gemini.google.com/app"


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


class _ResponseSnapshot(NamedTuple):
    selector: str
    count: int
    index: int
    text: str


class _CapturedResponse(NamedTuple):
    text: str
    snapshot: _ResponseSnapshot
    complete_json: bool
    stable_duration_ms: int


@dataclass(frozen=True)
class GeminiWebConfig:
    profile_dir: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_WEB_PROFILE_DIR",
            os.path.join("browser_profiles", "gemini"),
        )
    )
    headless: bool = False
    ready_timeout_ms: int = field(default_factory=lambda: int(os.getenv("GEMINI_WEB_READY_TIMEOUT_MS", "120000")))
    response_timeout_ms: int = field(default_factory=lambda: int(os.getenv("GEMINI_WEB_RESPONSE_TIMEOUT_MS", "120000")))
    no_container_timeout_ms: int = field(
        default_factory=lambda: _bounded_env_int(
            "GEMINI_WEB_NO_CONTAINER_TIMEOUT_MS", 60000, 30000, 120000
        )
    )
    response_stable_ms: int = 750
    poll_interval_ms: int = 500
    max_chat_submissions: int = field(
        default_factory=lambda: _bounded_env_int("GEMINI_WEB_MAX_CHAT_SUBMISSIONS", 6, 1, 50)
    )
    max_browser_submissions: int = field(
        default_factory=lambda: _bounded_env_int("GEMINI_WEB_MAX_BROWSER_SUBMISSIONS", 12, 1, 100)
    )
    recovery_backoff_ms: int = field(
        default_factory=lambda: _bounded_env_int("GEMINI_WEB_RECOVERY_BACKOFF_MS", 2000, 0, 10000)
    )
    diagnostic_sink: Callable[[dict[str, Any]], None] | None = None
    raw_debug_capture: bool = field(
        default_factory=lambda: os.getenv("GEMINI_WEB_CAPTURE_RAW_DEBUG", "").lower() in {"1", "true", "yes"}
    )
    raw_debug_dir: str = field(
        default_factory=lambda: os.path.join(tempfile.gettempdir(), "litsync-gemini-web-debug")
    )

    def __post_init__(self) -> None:
        bounded = max(30000, min(120000, int(self.no_container_timeout_ms)))
        object.__setattr__(
            self,
            "no_container_timeout_ms",
            min(bounded, max(0, int(self.response_timeout_ms))),
        )


class GeminiWebAutomation:
    def __init__(self, config: GeminiWebConfig | None = None):
        self.config = config or GeminiWebConfig()
        self._playwright = None
        self._context = None
        self._page = None
        self._submission_count = 0
        self._browser_submission_count = 0
        self._attempt_context: dict[str, Any] = {}
        self._last_wait_metadata: dict[str, Any] = {}
        self._last_response_capture_metadata: dict[str, Any] = {}
        self._raw_debug_file: Path | None = None

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
        self._browser_submission_count = 0
        self._attempt_context = {}
        self._last_wait_metadata = {}
        self._last_response_capture_metadata = {}

    @property
    def last_response_capture_metadata(self) -> dict[str, Any]:
        return dict(self._last_response_capture_metadata)

    def _clear_response_capture_metadata(self) -> None:
        self._last_response_capture_metadata = {}
        for field_name in (
            "response_return_reason",
            "response_complete_json_at_capture",
            "response_generation_detected_at_capture",
            "response_stable_duration_ms",
            "response_utf8_bytes_at_capture",
            "response_selector_at_capture",
            "response_container_count_at_capture",
        ):
            self._last_wait_metadata.pop(field_name, None)

    def set_attempt_context(self, *, stage: str, retry_number: int) -> None:
        self._attempt_context = {"stage": str(stage), "retry_number": int(retry_number)}

    def note_recovery(self, action: str) -> None:
        self._emit_diagnostic(
            outcome="recovery", recovery_action=str(action),
            response_state="not_applicable", generation_detected=False,
        )

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
        self._last_wait_metadata = {}
        self._last_response_capture_metadata = {}
        started = time.perf_counter()
        try:
            with self._runtime_observation("preparation_before_submission"):
                self._prepare_for_submission()
            page = self._require_page()
            with self._runtime_observation("prompt_box_discovery_reload_readiness"):
                before = self._response_snapshots()
                box = self._find_prompt_box()
                if box is None:
                    page.reload(wait_until="domcontentloaded")
                    self.wait_until_ready()
                    box = self._find_prompt_box()
                if box is None:
                    raise RuntimeError("Could not find Gemini prompt input.")

            with self._runtime_observation("prompt_fill_and_submit"):
                box.click()
                box.fill(
                    prompt
                    + "\n\nWEB AUTOMATION OUTPUT RULE: Return exactly one JSON object matching the requested schema. "
                      "Do not use Markdown fences, headings, commentary, or text before or after the JSON."
                )
                self._submit_prompt()
            with self._runtime_observation("response_wait"):
                response = self._wait_for_new_response(before)
            self._submission_count += 1
            self._browser_submission_count += 1
            with self._runtime_observation("final_response_capture_diagnostics"):
                self._record_raw_response(response)
                self._emit_diagnostic(
                    outcome="completed", attempt_duration_ms=round((time.perf_counter() - started) * 1000),
                    fallback_reason="", **self._last_wait_metadata,
                )
            return response
        except TimeoutError as exc:
            self._clear_response_capture_metadata()
            self._emit_diagnostic(
                outcome="timeout", attempt_duration_ms=round((time.perf_counter() - started) * 1000),
                fallback_reason=str(exc), **self._last_wait_metadata,
            )
            raise
        except RuntimeError as exc:
            self._clear_response_capture_metadata()
            self._emit_diagnostic(
                outcome="browser_error", attempt_duration_ms=round((time.perf_counter() - started) * 1000),
                fallback_reason=str(exc), **self._last_wait_metadata,
            )
            raise
        except Exception as exc:
            # Playwright's DOM/navigation errors do not inherit RuntimeError;
            # normalize them so the screening retry policy can recover the chat.
            error = RuntimeError("Gemini Web browser interaction failed.")
            self._clear_response_capture_metadata()
            self._emit_diagnostic(
                outcome="browser_error", attempt_duration_ms=round((time.perf_counter() - started) * 1000),
                fallback_reason=str(error), **self._last_wait_metadata,
            )
            raise error from exc

    def start_new_job_chat(self) -> None:
        """Open one clean chat for a screening job, not for every request."""
        with self._runtime_observation("new_chat_creation"):
            page = self._require_page()
            page.goto(GEMINI_URL, wait_until="domcontentloaded")
            self.wait_until_ready()
            self._submission_count = 0

    def _prepare_for_submission(self) -> None:
        if self._browser_submission_count >= self.config.max_browser_submissions:
            self.recycle_browser_context("proactive_browser_recycle", backoff=False)
        elif self._submission_count >= self.config.max_chat_submissions:
            self.note_recovery("proactive_new_job_chat")
            self.start_new_job_chat()

    def recycle_browser_context(self, action: str, *, backoff: bool = True) -> None:
        with self._runtime_observation("browser_context_recycle"):
            attempt_context = dict(self._attempt_context)
            self.note_recovery(action)
            if backoff and self.config.recovery_backoff_ms:
                with self._runtime_observation("recovery_backoff_sleep"):
                    time.sleep(self.config.recovery_backoff_ms / 1000)
            self.close()
            self.start()
            self._attempt_context = attempt_context

    def recover_transport_failure(self, *, exhausted: bool = False) -> None:
        with self._runtime_observation("browser_recovery"):
            no_container_timeout = (
                self._last_wait_metadata.get("timeout_stage") in {
                    "timeout_final_sweep",
                    "stalled_generation_no_container",
                }
                and self._last_wait_metadata.get("response_state") == "no_new_response"
                and int(self._last_wait_metadata.get("response_container_count") or 0) == 0
            )
            if exhausted or no_container_timeout:
                action = (
                    "browser_recycle_after_exhausted_retry"
                    if exhausted else "browser_recycle_after_no_container_timeout"
                )
                self.recycle_browser_context(action)
                return
            self.note_recovery("new_job_chat")
            self.recover_job_chat()

    def recover_job_chat(self) -> None:
        """Recover after a broken page; batch prompts carry the full protocol."""
        try:
            self.start_new_job_chat()
        except Exception:
            # A renderer/browser crash needs a fresh persistent context, while
            # retaining the saved profile and login.
            self.close()
            self.start()

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

    def _wait_for_new_response(self, before: tuple[_ResponseSnapshot, ...]) -> str:
        deadline = time.monotonic() + (self.config.response_timeout_ms / 1000)
        stalled_generation_since: float | None = None
        stable_since: dict[tuple[str, int], float] = {}
        last_text: dict[tuple[str, int], str] = {}

        while time.monotonic() < deadline:
            current = self._new_response_snapshots(before, self._response_snapshots())
            generating = self._is_generating()
            completed = self._stable_completed_response(current, last_text, stable_since, generating)
            self._last_wait_metadata = self._wait_metadata(current, generating, "polling")
            if completed is not None:
                self._publish_response_capture(
                    completed,
                    response_return_reason=(
                        "complete_json_stable"
                        if completed.complete_json
                        else "incomplete_response_stable_generation_stopped"
                    ),
                    generating=generating,
                    timeout_stage="polling",
                )
                return completed.text
            if generating and not current:
                stalled_now = time.monotonic()
                if stalled_generation_since is None:
                    stalled_generation_since = stalled_now
                elif (
                    (stalled_now - stalled_generation_since) * 1000
                    >= self.config.no_container_timeout_ms
                ):
                    self._last_wait_metadata = self._wait_metadata(
                        current,
                        generating,
                        "stalled_generation_no_container",
                    )
                    raise TimeoutError(
                        "Gemini Web did not produce a new completed response in time. "
                        "Check the opened Gemini window for login, consent, quota, or network messages."
                    )
            else:
                stalled_generation_since = None
            time.sleep(self.config.poll_interval_ms / 1000)

        current = self._new_response_snapshots(before, self._response_snapshots())
        generating = self._is_generating()
        completed = self._completed_json_response(current, generating)
        self._last_wait_metadata = self._wait_metadata(current, generating, "timeout_final_sweep")
        if completed is not None:
            self._publish_response_capture(
                completed,
                response_return_reason="timeout_final_sweep_complete",
                generating=generating,
                timeout_stage="timeout_final_sweep",
            )
            return completed.text
        raise TimeoutError(
            "Gemini Web did not produce a new completed response in time. "
            "Check the opened Gemini window for login, consent, quota, or network messages."
        )

    @staticmethod
    def _is_complete_json(text: str) -> bool:
        candidate = text.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.IGNORECASE | re.DOTALL)
        if fenced:
            candidate = fenced.group(1).strip()
        try:
            return isinstance(json.loads(candidate), (dict, list))
        except (json.JSONDecodeError, TypeError):
            return False

    def _latest_response_text(self) -> str:
        snapshots = self._response_snapshots()
        return snapshots[-1].text if snapshots else ""

    def _response_snapshots(self) -> tuple[_ResponseSnapshot, ...]:
        page = self._require_page()
        selectors = [
            "model-response",
            "[data-test-id='model-response']",
            "[data-message-author-role='model']",
            "[data-author='model']",
            ".model-response-text",
            "[data-response-index]",
            "div.markdown.markdown-main-panel",
            "message-content",
        ]
        snapshots: list[_ResponseSnapshot] = []
        for selector in selectors:
            locator = page.locator(selector)
            try:
                count = locator.count()
                for index in range(count):
                    candidate = locator.nth(index)
                    if not candidate.is_visible():
                        continue
                    text = candidate.inner_text(timeout=2_000).strip()
                    if text:
                        snapshots.append(_ResponseSnapshot(selector, count, index, text))
            except Exception:
                continue
        return tuple(snapshots)

    @staticmethod
    def _new_response_snapshots(
        before: tuple[_ResponseSnapshot, ...], current: tuple[_ResponseSnapshot, ...],
    ) -> tuple[_ResponseSnapshot, ...]:
        baseline = {(item.selector, item.index): item.text for item in before}
        return tuple(
            item for item in current
            if baseline.get((item.selector, item.index)) != item.text
        )

    def _stable_completed_response(
        self,
        snapshots: tuple[_ResponseSnapshot, ...],
        last_text: dict[tuple[str, int], str],
        stable_since: dict[tuple[str, int], float],
        generating: bool,
    ) -> _CapturedResponse | None:
        for snapshot in reversed(snapshots):
            key = (snapshot.selector, snapshot.index)
            if last_text.get(key) != snapshot.text:
                last_text[key] = snapshot.text
                stable_since[key] = time.monotonic()
                continue
            complete_json = self._is_complete_json(snapshot.text)
            required_ms = self.config.response_stable_ms if complete_json else 4_000
            stable_ms = (time.monotonic() - stable_since[key]) * 1000
            if stable_ms >= required_ms and not generating:
                return _CapturedResponse(
                    snapshot.text,
                    snapshot,
                    complete_json,
                    max(0, round(stable_ms)),
                )
        return None

    def _completed_json_response(
        self, snapshots: tuple[_ResponseSnapshot, ...], generating: bool,
    ) -> _CapturedResponse | None:
        if generating:
            return None
        for snapshot in reversed(snapshots):
            if self._is_complete_json(snapshot.text):
                return _CapturedResponse(snapshot.text, snapshot, True, 0)
        return None

    def _publish_response_capture(
        self,
        completed: _CapturedResponse,
        *,
        response_return_reason: str,
        generating: bool,
        timeout_stage: str,
    ) -> None:
        try:
            capture = {
                "response_return_reason": str(response_return_reason)[:120],
                "response_complete_json_at_capture": bool(completed.complete_json),
                "response_generation_detected_at_capture": bool(generating),
                "response_stable_duration_ms": int(completed.stable_duration_ms),
                "response_utf8_bytes_at_capture": len(
                    completed.text.encode("utf-8")
                ),
                "response_selector_at_capture": completed.snapshot.selector[:120],
                "response_container_count_at_capture": int(
                    completed.snapshot.count
                ),
            }
            self._last_response_capture_metadata = capture
            self._last_wait_metadata = {
                **self._wait_metadata(
                    (completed.snapshot,),
                    generating,
                    timeout_stage,
                ),
                **capture,
            }
        except Exception:
            # Capture diagnostics cannot alter the response selected for return.
            self._last_response_capture_metadata = {}

    @staticmethod
    def _wait_metadata(
        snapshots: tuple[_ResponseSnapshot, ...], generating: bool, timeout_stage: str,
    ) -> dict[str, Any]:
        latest = snapshots[-1] if snapshots else None
        return {
            "response_selector": latest.selector if latest else "",
            "response_container_count": latest.count if latest else 0,
            "response_state": "new_response" if latest else "no_new_response",
            "generation_detected": bool(generating),
            "timeout_stage": timeout_stage,
        }

    def _emit_diagnostic(self, *, outcome: str, **metadata: Any) -> None:
        sink = self.config.diagnostic_sink
        if sink is None:
            return
        event = {
            "event": "gemini_web_attempt",
            "submission_number": self._submission_count + 1,
            "stage": self._attempt_context.get("stage", "unknown"),
            "retry_number": self._attempt_context.get("retry_number", 0),
            "outcome": outcome,
            "recovery_action": metadata.pop("recovery_action", ""),
            "attempt_duration_ms": metadata.pop("attempt_duration_ms", 0),
            "response_selector": metadata.pop("response_selector", ""),
            "response_container_count": metadata.pop("response_container_count", 0),
            "response_state": metadata.pop("response_state", "unknown"),
            "generation_detected": bool(metadata.pop("generation_detected", False)),
            "timeout_stage": metadata.pop("timeout_stage", ""),
            "fallback_reason": metadata.pop("fallback_reason", ""),
            "response_return_reason": metadata.pop("response_return_reason", ""),
            "response_complete_json_at_capture": metadata.pop(
                "response_complete_json_at_capture", None
            ),
            "response_generation_detected_at_capture": metadata.pop(
                "response_generation_detected_at_capture", None
            ),
            "response_stable_duration_ms": metadata.pop(
                "response_stable_duration_ms", 0
            ),
            "response_utf8_bytes_at_capture": metadata.pop(
                "response_utf8_bytes_at_capture", 0
            ),
            "response_selector_at_capture": metadata.pop(
                "response_selector_at_capture", ""
            ),
            "response_container_count_at_capture": metadata.pop(
                "response_container_count_at_capture", 0
            ),
        }
        try:
            sink(event)
        except Exception:
            # Diagnostics cannot replace a browser response or browser error.
            return

    @contextmanager
    def _runtime_observation(self, metric: str) -> Iterator[None]:
        try:
            started = time.perf_counter()
        except Exception:
            started = None
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            if started is not None:
                try:
                    duration = max(0.0, time.perf_counter() - started)
                    sink = self.config.diagnostic_sink
                    if sink is not None:
                        sink({
                            "event": "gemini_web_runtime",
                            "runtime_metric": str(metric),
                            "runtime_family": "browser_transport",
                            "duration_seconds": duration,
                            "stage": self._attempt_context.get("stage", "unknown"),
                            "retry_number": self._attempt_context.get("retry_number", 0),
                            "attempt_type": "",
                            "outcome": "completed" if succeeded else "failed",
                        })
                except Exception:
                    # Timing and metrics emission are strictly best effort.
                    pass

    def _record_raw_response(self, response: str) -> None:
        if not self.config.raw_debug_capture:
            return
        if self._raw_debug_file is None:
            directory = Path(self.config.raw_debug_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self._raw_debug_file = directory / f"gemini-web-{int(time.time())}-{os.getpid()}.jsonl"
        with self._raw_debug_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"response": response}, ensure_ascii=False) + "\n")

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
