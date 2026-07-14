from __future__ import annotations

import json

import pandas as pd
from fastapi.testclient import TestClient

import server
from bulk_screen import _resume_rows
from evaluation.local_ai_benchmark import evaluate_rows


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
