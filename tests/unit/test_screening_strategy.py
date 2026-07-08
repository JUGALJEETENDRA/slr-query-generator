import json

import screening_strategies
from processing_engines import GEMINI_API_ENGINE, GEMINI_WEB_ENGINE, LOCAL_ENGINE
from screening_strategies import (
    DEFAULT_SCREENING_STRATEGY,
    DIRECT_AI,
    LITSYNC_WORKFLOW,
    normalize_screening_strategy,
    screen_candidate,
)


class FakeInferenceEngine:
    def __init__(self, engine_id):
        self.engine_id = engine_id
        self.calls = 0

    def ask(self, *args, **kwargs):
        self.calls += 1
        return json.dumps({"decision": "REJECT", "reason": "Direct reason.", "confidence": 0.8})


def test_litsync_strategy_runs_only_litsync(monkeypatch):
    calls = {"litsync": 0, "direct": 0}

    def fake_screen_paper(**kwargs):
        calls["litsync"] += 1
        return {
            "decision": "KEEP",
            "reason": "LitSync reason.",
            "technology_match": 1.0,
            "task_role_match": 1.0,
            "context_match": 1.0,
            "review_role_match": True,
        }

    def fake_ask_ollama(*args, **kwargs):
        calls["direct"] += 1
        return json.dumps({"decision": "REJECT", "reason": "Direct reason.", "confidence": 0.1})

    monkeypatch.setattr(screening_strategies, "screen_paper", fake_screen_paper)
    monkeypatch.setattr(screening_strategies, "ask_ollama", fake_ask_ollama)

    result = screen_candidate(
        title="Title",
        abstract="Abstract",
        research_question="Question",
        strategy=LITSYNC_WORKFLOW,
        rq_frame={},
    )

    assert result["decision"] == "KEEP"
    assert result["metadata"]["screening_strategy"] == LITSYNC_WORKFLOW
    assert calls == {"litsync": 1, "direct": 0}


def test_non_public_strategies_fall_back_to_litsync(monkeypatch):
    calls = {"litsync": 0, "direct": 0}

    def fake_screen_paper(**kwargs):
        calls["litsync"] += 1
        return {"decision": "REJECT", "reason": "LitSync reason."}

    def fake_ask_ollama(*args, **kwargs):
        calls["direct"] += 1
        return json.dumps({"decision": "MAYBE", "reason": "Direct reason.", "confidence": 0.6})

    monkeypatch.setattr(screening_strategies, "screen_paper", fake_screen_paper)
    monkeypatch.setattr(screening_strategies, "ask_ollama", fake_ask_ollama)

    result = screen_candidate(
        title="Title",
        abstract="Abstract",
        research_question="Question",
        strategy=DIRECT_AI,
        rq_frame={},
    )

    assert normalize_screening_strategy(DIRECT_AI) == DEFAULT_SCREENING_STRATEGY
    assert normalize_screening_strategy("removed-workflow") == DEFAULT_SCREENING_STRATEGY
    assert result["decision"] == "REJECT"
    assert result["reason"] == "LitSync reason."
    assert result["metadata"]["screening_strategy"] == LITSYNC_WORKFLOW
    assert calls == {"litsync": 1, "direct": 0}


def test_workflow_does_not_change_selected_processing_engine(monkeypatch):
    calls = []

    def fake_screen_paper(**kwargs):
        calls.append(
            {
                "workflow": "litsync",
                "mode": kwargs.get("mode"),
                "engine_id": kwargs.get("inference_engine").engine_id,
            }
        )
        return {
            "decision": "KEEP",
            "reason": "LitSync included it.",
            "technology_match": 0.9,
            "task_role_match": 0.9,
            "context_match": 0.8,
            "review_role_match": True,
        }

    monkeypatch.setattr(screening_strategies, "screen_paper", fake_screen_paper)

    for engine_id in (LOCAL_ENGINE, GEMINI_API_ENGINE, GEMINI_WEB_ENGINE):
        engine = FakeInferenceEngine(engine_id)
        result = screen_candidate(
            title="Title",
            abstract="Abstract",
            research_question="Question",
            strategy=LITSYNC_WORKFLOW,
            rq_frame={},
            mode=engine_id,
            inference_engine=engine,
        )

        assert result["metadata"]["screening_strategy"] == LITSYNC_WORKFLOW
        assert calls[-1]["mode"] == engine_id
        assert calls[-1]["engine_id"] == engine_id
        assert engine.calls == 0
