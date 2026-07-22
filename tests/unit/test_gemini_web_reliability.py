import json

import pandas as pd

from gemini_web_automation import GeminiWebAutomation, GeminiWebConfig, _ResponseSnapshot
from gemini_web_screening import GeminiWebDiagnostics, TRANSPORT_TIMEOUT_FAILURE, _resume_rows


def _snapshot(selector, count, index, text):
    return _ResponseSnapshot(selector, count, index, text)


def test_watcher_accepts_json_from_new_alternate_response_container(monkeypatch):
    browser = GeminiWebAutomation(GeminiWebConfig(response_stable_ms=0, poll_interval_ms=0))
    before = (_snapshot("model-response", 1, 0, "old response"),)
    alternate = _snapshot("[data-message-author-role='model']", 2, 1, '{"items": []}')
    snapshots = iter([(alternate,), (alternate,)])
    monkeypatch.setattr(browser, "_response_snapshots", lambda: next(snapshots))
    monkeypatch.setattr(browser, "_is_generating", lambda: False)
    monkeypatch.setattr("gemini_web_automation.time.sleep", lambda _: None)

    assert browser._wait_for_new_response(before) == '{"items": []}'


def test_timeout_boundary_final_sweep_accepts_completed_json(monkeypatch):
    browser = GeminiWebAutomation(GeminiWebConfig(response_timeout_ms=0, poll_interval_ms=0))
    before = (_snapshot("model-response", 1, 0, "old response"),)
    final = _snapshot("model-response", 2, 1, '{"items": []}')
    monkeypatch.setattr(browser, "_response_snapshots", lambda: (final,))
    monkeypatch.setattr(browser, "_is_generating", lambda: False)

    assert browser._wait_for_new_response(before) == '{"items": []}'
    assert browser._last_wait_metadata["timeout_stage"] == "timeout_final_sweep"


def test_watcher_does_not_accept_json_while_generation_is_visible(monkeypatch):
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


def test_diagnostics_allow_only_metadata_fields(tmp_path):
    target = tmp_path / "diagnostics.jsonl"
    diagnostics = GeminiWebDiagnostics(target)
    diagnostics.record({
        "event": "gemini_web_attempt", "outcome": "completed",
        "prompt": "must not persist", "response": "must not persist",
        "response_selector": "model-response", "attempt_duration_ms": 42,
    })

    event = json.loads(target.read_text(encoding="utf-8"))
    assert event["response_selector"] == "model-response"
    assert "prompt" not in event
    assert "response" not in event


def test_raw_debug_capture_is_disabled_by_default(tmp_path):
    browser = GeminiWebAutomation(GeminiWebConfig(raw_debug_capture=False, raw_debug_dir=str(tmp_path)))

    browser._record_raw_response('{"items": []}')

    assert not list(tmp_path.iterdir())


def test_raw_debug_capture_requires_explicit_opt_in_and_uses_local_path(tmp_path):
    local_debug = tmp_path / "local-debug"
    browser = GeminiWebAutomation(GeminiWebConfig(raw_debug_capture=True, raw_debug_dir=str(local_debug)))

    browser._record_raw_response('{"items": []}')

    files = list(local_debug.glob("*.jsonl"))
    assert len(files) == 1
    assert files[0].parent == local_debug


def test_resume_requeues_transport_fallback_but_keeps_normal_maybe(tmp_path):
    checkpoint = tmp_path / "checkpoint.csv"
    rows = [
        {
            "Source_Row_Index": 0, "Protocol_ID": "protocol", "Prompt_Version": "gemini-web-batched-v2",
            "Layer_Trace_JSON": "[]", "Decision": "MAYBE", "Validation_Status": "validated",
            "Criteria_JSON": "[]", "Evidence_JSON": "[]", "Failure_Class": TRANSPORT_TIMEOUT_FAILURE,
            "Reason": "Gemini Web browser request failed after retry: timeout",
        },
        {
            "Source_Row_Index": 1, "Protocol_ID": "protocol", "Prompt_Version": "gemini-web-batched-v2",
            "Layer_Trace_JSON": "[]", "Decision": "MAYBE", "Validation_Status": "validated",
            "Criteria_JSON": "[]", "Evidence_JSON": "[]", "Failure_Class": "",
            "Reason": "Title and abstract leave a material eligibility criterion unclear.",
        },
    ]
    pd.DataFrame(rows).to_csv(checkpoint, index=False)

    resumed = _resume_rows(checkpoint, "protocol", {"0", "1"})

    assert set(resumed) == {"1"}
