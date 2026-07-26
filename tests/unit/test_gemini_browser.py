from __future__ import annotations

from litsync_app.integrations.gemini_browser import (
    GeminiWebAutomation,
    GeminiWebConfig,
    _ResponseSnapshot,
)


def _snapshot(selector: str, count: int, index: int, text: str) -> _ResponseSnapshot:
    return _ResponseSnapshot(selector, count, index, text)


def test_watcher_accepts_completed_json_from_alternate_container(monkeypatch):
    browser = GeminiWebAutomation(GeminiWebConfig(response_stable_ms=0, poll_interval_ms=0))
    before = (_snapshot("model-response", 1, 0, "old response"),)
    alternate = _snapshot("[data-message-author-role='model']", 2, 1, '{"items": []}')
    snapshots = iter([(alternate,), (alternate,)])
    monkeypatch.setattr(browser, "_response_snapshots", lambda: next(snapshots))
    monkeypatch.setattr(browser, "_is_generating", lambda: False)
    monkeypatch.setattr("litsync_app.integrations.gemini_browser.time.sleep", lambda _: None)

    assert browser._wait_for_new_response(before) == '{"items": []}'


def test_timeout_boundary_final_sweep_accepts_completed_json(monkeypatch):
    browser = GeminiWebAutomation(GeminiWebConfig(response_timeout_ms=0, poll_interval_ms=0))
    before = (_snapshot("model-response", 1, 0, "old response"),)
    final = _snapshot("model-response", 2, 1, '{"items": []}')
    monkeypatch.setattr(browser, "_response_snapshots", lambda: (final,))
    monkeypatch.setattr(browser, "_is_generating", lambda: False)

    assert browser._wait_for_new_response(before) == '{"items": []}'
    assert browser._last_wait_metadata["timeout_stage"] == "timeout_final_sweep"


def test_watcher_rejects_json_while_generation_is_visible(monkeypatch):
    browser = GeminiWebAutomation(GeminiWebConfig(response_timeout_ms=0, poll_interval_ms=0))
    before = (_snapshot("model-response", 1, 0, "old response"),)
    final = _snapshot("model-response", 2, 1, '{"items": []}')
    monkeypatch.setattr(browser, "_response_snapshots", lambda: (final,))
    monkeypatch.setattr(browser, "_is_generating", lambda: True)

    try:
        browser._wait_for_new_response(before)
    except TimeoutError:
        pass
    else:
        raise AssertionError("Generating response must not be accepted as complete")


def test_raw_debug_capture_is_disabled_by_default(tmp_path):
    browser = GeminiWebAutomation(
        GeminiWebConfig(raw_debug_capture=False, raw_debug_dir=str(tmp_path))
    )
    browser._record_raw_response('{"items": []}')
    assert not list(tmp_path.iterdir())


def test_lifecycle_rotates_chat_at_configured_limit(monkeypatch):
    browser = GeminiWebAutomation(
        GeminiWebConfig(max_chat_submissions=6, max_browser_submissions=12)
    )
    browser._submission_count = 6
    browser._browser_submission_count = 6
    actions = []
    monkeypatch.setattr(browser, "note_recovery", actions.append)
    monkeypatch.setattr(browser, "start_new_job_chat", lambda: actions.append("start_new_chat"))

    browser._prepare_for_submission()

    assert actions == ["proactive_new_job_chat", "start_new_chat"]
