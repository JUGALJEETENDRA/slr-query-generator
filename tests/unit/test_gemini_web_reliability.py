import json

import pandas as pd

from gemini_web_automation import GeminiWebAutomation, GeminiWebConfig, _ResponseSnapshot
from gemini_web_screening import (
    GEMINI_WEB_LEGACY_PROTOCOL_VERSION, GEMINI_WEB_PROTOCOL_CACHE_VERSION,
    GEMINI_WEB_VERSION, GeminiWebDiagnostics, TRANSPORT_TIMEOUT_FAILURE,
    _load_cached_protocol, _protocol_hash, _resume_rows,
)
from local_ai.contracts import ReviewProtocol


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
            "Source_Row_Index": 0, "Protocol_ID": "protocol", "Prompt_Version": GEMINI_WEB_VERSION,
            "Layer_Trace_JSON": "[]", "Decision": "MAYBE", "Validation_Status": "validated",
            "Criteria_JSON": "[]", "Evidence_JSON": "[]", "Failure_Class": TRANSPORT_TIMEOUT_FAILURE,
            "Critic_Route": "inclusion_only_reject", "Verification_Status": "failed",
            "Reason": "Gemini Web browser request failed after retry: timeout",
        },
        {
            "Source_Row_Index": 1, "Protocol_ID": "protocol", "Prompt_Version": GEMINI_WEB_VERSION,
            "Layer_Trace_JSON": "[]", "Decision": "MAYBE", "Validation_Status": "validated",
            "Criteria_JSON": "[]", "Evidence_JSON": "[]", "Failure_Class": "",
            "Critic_Route": "", "Verification_Status": "not_required",
            "Reason": "Title and abstract leave a material eligibility criterion unclear.",
        },
    ]
    pd.DataFrame(rows).to_csv(checkpoint, index=False)

    resumed = _resume_rows(checkpoint, "protocol", {"0", "1"})

    assert set(resumed) == {"1"}


def test_v22_decision_checkpoint_is_not_resumed_by_v23(tmp_path):
    checkpoint = tmp_path / "checkpoint.csv"
    pd.DataFrame([{
        "Source_Row_Index": 0, "Protocol_ID": "protocol",
        "Prompt_Version": "gemini-web-batched-v2.2", "Layer_Trace_JSON": "[]",
        "Decision": "KEEP", "Validation_Status": "validated", "Criteria_JSON": "[]",
        "Evidence_JSON": "[]", "Failure_Class": "", "Critic_Route": "",
        "Verification_Status": "not_required", "Reason": "Legacy decision.",
    }]).to_csv(checkpoint, index=False)

    assert _resume_rows(checkpoint, "protocol", {"0"}) == {}


def test_protocol_cache_migrates_v21_without_recompilation(tmp_path):
    cache_root = tmp_path / "cache" / "gemini_web"
    protocols = cache_root / "protocols"
    protocols.mkdir(parents=True)
    values = ("Stable question?", "", "", "")
    protocol = ReviewProtocol.model_validate({
        "schema_version": "2.0", "protocol_id": "", "research_question": values[0],
        "objective": "Assess stable evidence.", "scope_interpretation": "Use stable scope.",
        "criteria": [{
            "id": "inc1", "kind": "inclusion", "description": "Required relationship",
            "required": True, "expected_evidence": "Direct evidence", "source": "research_question",
        }],
        "prompt_version": GEMINI_WEB_LEGACY_PROTOCOL_VERSION, "model": "gemini-web",
    }).with_identity()
    legacy = protocols / f"{_protocol_hash(*values, version=GEMINI_WEB_LEGACY_PROTOCOL_VERSION)}.json"
    legacy.write_text(protocol.model_dump_json(), encoding="utf-8")

    loaded, canonical = _load_cached_protocol(cache_root, *values)

    assert loaded == protocol
    assert canonical.name == f"{_protocol_hash(*values)}.json"
    assert canonical.exists()
    assert GEMINI_WEB_PROTOCOL_CACHE_VERSION not in canonical.read_text(encoding="utf-8")


def test_lifecycle_rotates_chat_before_seventh_submission(monkeypatch):
    browser = GeminiWebAutomation(GeminiWebConfig(max_chat_submissions=6, max_browser_submissions=12))
    browser._submission_count = 6
    browser._browser_submission_count = 6
    actions = []
    monkeypatch.setattr(browser, "note_recovery", actions.append)
    monkeypatch.setattr(browser, "start_new_job_chat", lambda: actions.append("start_new_chat"))

    browser._prepare_for_submission()

    assert actions == ["proactive_new_job_chat", "start_new_chat"]


def test_lifecycle_recycles_browser_before_thirteenth_submission(monkeypatch):
    browser = GeminiWebAutomation(GeminiWebConfig(max_chat_submissions=6, max_browser_submissions=12))
    browser._submission_count = 2
    browser._browser_submission_count = 12
    actions = []
    monkeypatch.setattr(
        browser, "recycle_browser_context",
        lambda action, backoff=True: actions.append((action, backoff)),
    )

    browser._prepare_for_submission()

    assert actions == [("proactive_browser_recycle", False)]


def test_no_container_timeout_recycles_with_bounded_backoff(monkeypatch):
    browser = GeminiWebAutomation(GeminiWebConfig(recovery_backoff_ms=10000))
    browser._last_wait_metadata = {
        "timeout_stage": "timeout_final_sweep",
        "response_state": "no_new_response",
        "response_container_count": 0,
    }
    actions = []
    monkeypatch.setattr(
        browser, "recycle_browser_context",
        lambda action, backoff=True: actions.append((action, backoff)),
    )

    browser.recover_transport_failure()

    assert actions == [("browser_recycle_after_no_container_timeout", True)]
