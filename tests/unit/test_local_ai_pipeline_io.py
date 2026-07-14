from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import requests
from fastapi.testclient import TestClient
from pydantic import BaseModel

import server
from bulk_screen import _resume_rows
from evaluation.local_ai_benchmark import evaluate_rows
from local_ai.engine import OllamaStructuredEngine


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

    monkeypatch.setattr("local_ai.engine.requests.post", post)
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

    monkeypatch.setattr("local_ai.engine.requests.post", post)
    OllamaStructuredEngine._schema_grammar_support.clear()
    profile = SimpleNamespace(keep_alive="5m", num_ctx=4096)
    engine = OllamaStructuredEngine(profile)
    engine.generate("local-model", "first", _ReadyOutput)
    engine.generate("local-model", "second", _OtherOutput)
    assert seen_formats[1] == "json"
    assert isinstance(seen_formats[2], dict)


def test_batch_wire_schema_is_compact_but_keeps_required_enums():
    from local_ai.three_layer import TriageBatch

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


def test_gold_evaluator_enforces_recall_precision_and_evidence():
    screened = [
        {
            "Source_Row_Index": 1, "Title": "Relevant", "Abstract": "Direct evidence.",
            "Decision": "KEEP", "Validation_Status": "validated",
            "Evidence_JSON": json.dumps([{"source": "abstract", "quote": "Direct evidence."}]),
        },
        {
            "Source_Row_Index": 2, "Title": "Irrelevant", "Abstract": "Other topic.",
            "Decision": "REJECT", "Validation_Status": "validated", "Evidence_JSON": "[]",
        },
    ]
    gold = [
        {"Source_Row_Index": 1, "Gold_Decision": "KEEP"},
        {"Source_Row_Index": 2, "Gold_Decision": "REJECT"},
    ]
    metrics = evaluate_rows(screened, gold)
    assert metrics["relevant_recall_keep_or_maybe"] == 1.0
    assert metrics["definitive_keep_precision"] == 1.0
    assert all(metrics["gates"].values())


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
