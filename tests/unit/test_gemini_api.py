from __future__ import annotations

import json

import pandas as pd
import pytest

from litsync_app.screening.gemini_api import (
    ARCHITECTURE_VERSION,
    GeminiApiAuthenticationError,
    GeminiApiInterruptedError,
    GeminiApiQuotaError,
    _request_once,
    assess_paper,
    screen_csv_with_gemini_api,
)


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.ok = 200 <= status_code < 300

    def json(self):
        return self._payload


class FakeProgress:
    def __init__(self):
        self.resumed = 0
        self.retries = 0
        self.count_updates = []
        self.finished = False

    def set_resumed_count(self, _job_id, count):
        self.resumed = count

    def begin_batches(self, *_args):
        pass

    def update_counts(self, _job_id, current, keep, maybe, reject):
        self.count_updates.append((current, keep, maybe, reject))

    def update_batch(self, *_args):
        pass

    def record_retry(self, _job_id):
        self.retries += 1

    def set_screening_final_metadata(self, *_args, **_kwargs):
        pass

    def finish(self, _job_id):
        self.finished = True


class FakeSession:
    def __init__(self):
        self.rows = []
        self.metadata = {}

    @staticmethod
    def counts(rows):
        return {
            "total": len(rows),
            "keep": sum(row.get("Decision") == "KEEP" for row in rows),
            "maybe": sum(row.get("Decision") == "MAYBE" for row in rows),
            "reject": sum(row.get("Decision") == "REJECT" for row in rows),
        }

    def set_results(self, rows, **metadata):
        self.rows = [dict(row) for row in rows]
        self.metadata = metadata


def gemini_payload(decision="KEEP", evidence="Paper one"):
    content = json.dumps({
        "decision": decision,
        "confidence": 0.91,
        "reason": "The supplied title and abstract support this decision.",
        "evidence_quote": evidence,
    })
    return {
        "candidates": [{"content": {"parts": [{"text": content}]}}],
        "usageMetadata": {"totalTokenCount": 123},
    }


def test_request_uses_key_only_in_header():
    calls = []
    secret = "private-user-key"

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(200, gemini_payload())

    result = _request_once("prompt", api_key=secret, post=post)
    url, kwargs = calls[0]
    assert secret not in url
    assert secret not in json.dumps(kwargs["json"])
    assert kwargs["headers"]["x-goog-api-key"] == secret
    assert kwargs["json"]["generationConfig"]["responseMimeType"] == "application/json"
    assert kwargs["json"]["generationConfig"]["responseJsonSchema"]["additionalProperties"] is False
    assert result["total_tokens"] == 123


def test_authentication_error_is_not_retried():
    calls = []

    def generate(_prompt, **_kwargs):
        calls.append(1)
        raise GeminiApiAuthenticationError("rejected")

    with pytest.raises(GeminiApiAuthenticationError):
        assess_paper(
            question="RQ", context="", inclusion="", exclusion="",
            title="Paper", abstract="Abstract", api_key="bad-key",
            generate=generate, sleep=lambda _seconds: None,
        )
    assert len(calls) == 1


def test_valid_decision_has_exact_evidence_and_no_key_in_result():
    secret = "private-user-key"

    def generate(_prompt, **kwargs):
        assert kwargs["api_key"] == secret
        return {"content": json.dumps({
            "decision": "REJECT",
            "confidence": 0.8,
            "reason": "Outside the supplied scope.",
            "evidence_quote": "Exact abstract evidence",
        }), "total_tokens": 30}

    result = assess_paper(
        question="RQ", context="Context", inclusion="Include studies",
        exclusion="Exclude reviews", title="Title",
        abstract="Exact abstract evidence appears here.", api_key=secret,
        generate=generate, sleep=lambda _seconds: None,
    )
    assert result["decision"] == "REJECT"
    assert result["validation_status"] == "validated"
    assert result["evidence_quote"] == "Exact abstract evidence"
    assert secret not in json.dumps(result)


def test_quota_interruption_saves_partial_then_resumes_with_different_key(tmp_path):
    frame = pd.DataFrame({
        "Title": ["Paper one", "Paper two", "Paper three"],
        "Abstract": ["Abstract one", "Abstract two", "Abstract three"],
        "Authors": ["A", "B", "C"],
    })
    first_key = "first-private-key"
    first_calls = []

    def quota_after_one(_prompt, **kwargs):
        first_calls.append(kwargs["api_key"])
        if len(first_calls) == 1:
            return {"content": json.dumps({
                "decision": "KEEP", "confidence": 0.9,
                "reason": "Fits.", "evidence_quote": "Paper one",
            }), "total_tokens": 20}
        raise GeminiApiQuotaError("quota")

    first_output = tmp_path / "runs" / "first.csv"
    first_progress = FakeProgress()
    first_session = FakeSession()
    with pytest.raises(GeminiApiInterruptedError, match="after 1 of 3 papers"):
        screen_csv_with_gemini_api(
            frame=frame, valid=frame, title_col="Title", abstract_col="Abstract",
            research_question="RQ", research_context="Context",
            inclusion_criteria="Include relevant studies",
            exclusion_criteria="Exclude irrelevant studies",
            output_path=str(first_output), job_id="first-job",
            input_fingerprint="same-input", resume=True,
            progress=first_progress, screening_session=first_session,
            api_key=first_key, generate=quota_after_one,
            sleep=lambda _seconds: None,
        )
    partial = pd.read_csv(first_output, dtype=str, keep_default_na=False)
    assert partial["Title"].tolist() == ["Paper one"]
    assert first_session.rows[0]["Decision"] == "KEEP"
    assert first_progress.retries == 3
    assert first_key not in first_output.read_text(encoding="utf-8-sig")
    checkpoint_files = list((tmp_path / "cache" / "gemini_api").glob("*.csv"))
    assert len(checkpoint_files) == 1
    assert first_key not in checkpoint_files[0].read_text(encoding="utf-8-sig")

    second_key = "replacement-private-key"
    second_calls = []

    def complete_remaining(prompt, **kwargs):
        second_calls.append(kwargs["api_key"])
        title = "Paper two" if "Title: Paper two\n" in prompt else "Paper three"
        decision = "REJECT" if title == "Paper two" else "MAYBE"
        return {"content": json.dumps({
            "decision": decision, "confidence": 0.7,
            "reason": "Assessed from supplied evidence.",
            "evidence_quote": title if decision != "MAYBE" else "",
        }), "total_tokens": 25}

    second_progress = FakeProgress()
    second_session = FakeSession()
    summary = screen_csv_with_gemini_api(
        frame=frame, valid=frame, title_col="Title", abstract_col="Abstract",
        research_question="RQ", research_context="Context",
        inclusion_criteria="Include relevant studies",
        exclusion_criteria="Exclude irrelevant studies",
        output_path=str(tmp_path / "runs" / "second.csv"), job_id="second-job",
        input_fingerprint="same-input", resume=True,
        progress=second_progress, screening_session=second_session,
        api_key=second_key, generate=complete_remaining,
        sleep=lambda _seconds: None,
    )
    assert second_progress.resumed == 1
    assert second_progress.finished is True
    assert second_calls == [second_key, second_key]
    assert summary["resumed_count"] == 1
    assert (summary["keep"], summary["reject"], summary["maybe"]) == (1, 1, 1)
    assert all(row["Architecture_Version"] == ARCHITECTURE_VERSION for row in second_session.rows)
    combined = json.dumps(second_session.rows) + json.dumps(summary)
    assert first_key not in combined
    assert second_key not in combined
