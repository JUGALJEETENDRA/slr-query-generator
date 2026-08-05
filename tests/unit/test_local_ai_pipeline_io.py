from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient
from pydantic import BaseModel

from litsync_app import app as server
from litsync_app.screening.bulk import _resume_rows
from litsync_app.screening.local.engine import (
    LocalAIOutputError,
    OllamaStructuredEngine,
)
from litsync_app.screening.local_v2.assessor import (
    _model_assessment_envelope_schema_for_count,
)


class _ReadyOutput(BaseModel):
    status: str


def _http_response(status: int, payload: dict) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.url = "http://localhost:11434/api/generate"
    response._content = json.dumps(payload).encode("utf-8")
    return response


def test_ollama_grammar_failure_falls_back_to_validated_json(monkeypatch):
    calls = []
    responses = [
        _http_response(400, {
            "error": '{"error":{"message":"Failed to initialize samplers: failed to parse grammar"}}'
        }),
        _http_response(200, {
            "response": '{"status":"ready"}', "eval_count": 3,
            "eval_duration": 1000000, "total_duration": 2000000,
        }),
    ]

    def post(url, json, timeout):
        calls.append(json)
        return responses.pop(0)

    monkeypatch.setattr("litsync_app.screening.local.engine.requests.post", post)
    OllamaStructuredEngine._schema_grammar_support.clear()
    profile = SimpleNamespace(keep_alive="5m", num_ctx=4096)
    result = OllamaStructuredEngine(profile).generate("local-model", "Return ready.", _ReadyOutput)
    assert result.value == {"status": "ready"}
    assert isinstance(calls[0]["format"], dict)
    assert calls[1]["format"] == "json"
    assert OllamaStructuredEngine._schema_grammar_support[
        "http://localhost:11434|_ReadyOutput"
    ] is False


def test_grammar_fallback_is_isolated_to_the_failing_schema(monkeypatch):
    class _OtherOutput(BaseModel):
        value: str

    seen_formats = []
    responses = [
        _http_response(400, {"error": "failed to parse grammar"}),
        _http_response(200, {"response": '{"status":"ready"}'}),
        _http_response(200, {"response": '{"value":"ok"}'}),
    ]

    def post(url, json, timeout):
        seen_formats.append(json["format"])
        return responses.pop(0)

    monkeypatch.setattr("litsync_app.screening.local.engine.requests.post", post)
    OllamaStructuredEngine._schema_grammar_support.clear()
    profile = SimpleNamespace(keep_alive="5m", num_ctx=4096)
    engine = OllamaStructuredEngine(profile)
    engine.generate("local-model", "first", _ReadyOutput)
    engine.generate("local-model", "second", _OtherOutput)
    assert seen_formats[1] == "json"
    assert isinstance(seen_formats[2], dict)


def test_batch_wire_schema_is_compact_but_keeps_required_enums():
    from litsync_app.screening.local.three_layer import TriageBatch

    wire = OllamaStructuredEngine._ollama_format_schema(TriageBatch)
    item = wire["properties"]["items"]["items"]
    assert item["required"] == ["p", "d", "k", "b", "e"]
    assert item["properties"]["d"]["enum"] == ["KEEP", "MAYBE", "REJECT"]
    assert item["properties"]["e"]["maxItems"] == 2
    assert "additionalProperties" not in item


def test_resume_only_reuses_validated_rows(tmp_path):
    path = tmp_path / "checkpoint.csv"
    pd.DataFrame([
        {
            "Source_Row_Index": 1, "Protocol_ID": "p1", "Validation_Status": "validated",
            "Processing_Seconds": 12.5,
        },
        {"Source_Row_Index": 2, "Protocol_ID": "p1", "Validation_Status": "unresolved"},
        {"Source_Row_Index": 3, "Protocol_ID": "other", "Validation_Status": "validated"},
    ]).to_csv(path, index=False)
    resumed = _resume_rows(str(path), "p1")
    assert set(resumed) == {"1"}
    assert resumed["1"]["Processing_Seconds"] == 0.0
    assert resumed["1"]["Original_Processing_Seconds"] == 12.5
    assert resumed["1"]["Cache_Hit"] is True


def test_screen_endpoint_returns_v2_contract(monkeypatch):
    monkeypatch.setattr(server, "screen_candidate", lambda **kwargs: {
        "schema_version": "2.0", "decision": "MAYBE", "reason": "Insufficient abstract evidence.",
        "confidence": 0.5, "protocol_id": "p1", "criteria": [], "evidence": [],
        "uncertainty": ["missing detail"], "validation_status": "validated",
        "model_tier": "compact", "resource_profile": "eco", "model": "qwen2.5:3b",
    })
    response = TestClient(server.app).post("/screen", json={
        "question": "Does the method address the task?",
        "title": "Paper", "abstract": "Short abstract.",
        "model_tier": "compact", "resource_profile": "eco",
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["schema_version"] == "2.0"
    assert payload["decision"] == "MAYBE"



def test_dynamic_local_v2_schema_gets_criterion_scaled_output_budget(monkeypatch):
    calls = []

    def post(url, json, timeout):
        calls.append(json)
        return _http_response(
            200,
            {
                "response": (
                    '{"protocol_id":"p","paper_id":"x","assessments":[]}'
                ),
                "eval_count": 8,
                "eval_duration": 1000000,
                "total_duration": 2000000,
            },
        )

    monkeypatch.delenv("LOCAL_AI_MAX_OUTPUT_TOKENS", raising=False)
    schema = _model_assessment_envelope_schema_for_count(6)
    monkeypatch.delenv(
        f"LOCAL_AI_MAX_OUTPUT_TOKENS_{schema.__name__.upper()}",
        raising=False,
    )
    monkeypatch.setattr("litsync_app.screening.local.engine.requests.post", post)
    OllamaStructuredEngine._schema_grammar_support.clear()

    profile = SimpleNamespace(keep_alive="5m", num_ctx=4096)
    OllamaStructuredEngine(profile).generate(
        "local-model",
        "Assess six criteria.",
        schema,
    )

    budget = OllamaStructuredEngine._default_output_tokens_for_schema(schema)
    assert budget >= 1600
    assert calls[0]["options"]["num_predict"] == budget


def test_dynamic_assessment_budget_scales_with_criterion_count():
    three = _model_assessment_envelope_schema_for_count(3)
    six = _model_assessment_envelope_schema_for_count(6)

    three_budget = OllamaStructuredEngine._default_output_tokens_for_schema(three)
    six_budget = OllamaStructuredEngine._default_output_tokens_for_schema(six)

    assert three_budget > 700
    assert six_budget > three_budget
    assert six_budget <= 4096


def test_malformed_json_error_preserves_elapsed_time(monkeypatch):
    def post(url, json, timeout):
        return _http_response(
            200,
            {
                "response": '{"protocol_id":"p","paper_id":"x","assessments":[',
                "eval_count": 700,
                "eval_duration": 1000000,
                "total_duration": 2000000,
            },
        )

    monkeypatch.setattr("litsync_app.screening.local.engine.requests.post", post)
    OllamaStructuredEngine._schema_grammar_support.clear()
    profile = SimpleNamespace(keep_alive="5m", num_ctx=4096)

    with pytest.raises(LocalAIOutputError) as exc_info:
        OllamaStructuredEngine(profile).generate(
            "local-model",
            "Return structured JSON.",
            _model_assessment_envelope_schema_for_count(6),
        )

    assert exc_info.value.elapsed_seconds > 0
