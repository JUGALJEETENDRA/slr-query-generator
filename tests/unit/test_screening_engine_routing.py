import pandas as pd
import threading
import json
import re
from pathlib import Path

import bulk_screen
import gemini_web_screening
from gemini_web_parser import GeminiResponseParseError
from processing_engines import GEMINI_API_ENGINE, GEMINI_WEB_ENGINE, LOCAL_ENGINE
import processing_engines
from screening_strategies import LITSYNC_WORKFLOW


class FakeEngine:
    def __init__(self, engine_id):
        self.engine_id = engine_id

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def ask(self, prompt, model=""):
        raise AssertionError("screen_csv should use patched title and strategy calls in this test")


class FakeGeminiWebAutomation:
    prompts = []

    def __init__(self, config=None):
        self.config = config

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def submit_prompt_and_get_response(self, prompt):
        self.prompts.append(prompt)
        ids = _paper_ids_from_prompt(prompt)
        return json.dumps({
            "decisions": [
                {
                    "id": paper_id,
                    "decision": "Include" if index == 0 else "Exclude",
                    "reason": "Relevant." if index == 0 else "Outside scope.",
                }
                for index, paper_id in enumerate(ids)
            ]
        })

    def wait_until_ready(self):
        return None


def _paper_ids_from_prompt(prompt):
    match = re.search(r"Papers:\s*(\[.*?\])\s*For each paper", prompt, re.DOTALL)
    assert match, prompt
    return [paper["id"] for paper in json.loads(match.group(1))]


def test_screen_csv_matrix_preserves_selected_processing_engine(monkeypatch):
    csv_path = Path("test_screening_engine_routing_input.csv")
    pd.DataFrame(
        [
            {"Title": "Paper A", "Abstract": "Abstract A"},
            {"Title": "Paper B", "Abstract": "Abstract B"},
        ]
    ).to_csv(csv_path, index=False)

    resolved_engines = []
    screened = []

    def fake_resolve_processing_engine(engine, **kwargs):
        resolved_engines.append(engine)
        return FakeEngine(engine)

    def fake_screen_candidate(**kwargs):
        screened.append(
            {
                "strategy": kwargs["strategy"],
                "mode": kwargs["mode"],
                "engine_id": kwargs["inference_engine"].engine_id,
            }
        )
        return {
            "decision": "KEEP",
            "reason": "Relevant.",
            "confidence": 0.9,
        }

    monkeypatch.setattr(bulk_screen, "resolve_processing_engine", fake_resolve_processing_engine)
    monkeypatch.setattr(bulk_screen, "extract_semantic_frame", lambda **kwargs: {})
    monkeypatch.setattr(bulk_screen, "screen_candidate", fake_screen_candidate)
    monkeypatch.setattr(gemini_web_screening, "GeminiWebAutomation", FakeGeminiWebAutomation)

    try:
        for engine in (LOCAL_ENGINE, GEMINI_API_ENGINE, GEMINI_WEB_ENGINE):
            for workflow in (LITSYNC_WORKFLOW,):
                resolved_engines.clear()
                screened.clear()
                FakeGeminiWebAutomation.prompts.clear()

                summary = bulk_screen.screen_csv(
                    csv_path=str(csv_path),
                    research_question="Question",
                    mode=engine,
                    screening_engine=engine,
                    semantic_strategy=workflow,
                    two_stage_enabled=False,
                )

                assert summary["semantic_strategy"] == workflow
                assert summary["total_papers"] == 2
                if engine == GEMINI_WEB_ENGINE:
                    assert summary["screening_engine"] == GEMINI_WEB_ENGINE
                    assert resolved_engines == []
                    assert screened == []
                    assert len(FakeGeminiWebAutomation.prompts) == 1
                else:
                    assert resolved_engines == [engine]
                    assert screened
                    assert all(call["strategy"] == workflow for call in screened)
                    assert all(call["mode"] == engine for call in screened)
                    assert all(call["engine_id"] == engine for call in screened)
    finally:
        if csv_path.exists():
            csv_path.unlink()


def test_gemini_web_screen_csv_uses_batch_workflow_without_generic_engine(monkeypatch):
    csv_path = Path("test_gemini_web_thread_input.csv")
    checkpoint_path = Path("test_gemini_web_checkpoint.csv")
    pd.DataFrame([{"Title": "Paper A", "Abstract": "Abstract A"}]).to_csv(csv_path, index=False)

    browser_thread_ids = []
    process_thread_ids = []

    def fake_resolve_processing_engine(engine, **kwargs):
        raise AssertionError("Gemini Web batch screening must not use the generic engine resolver")

    def fake_process_paper(*args, **kwargs):
        process_thread_ids.append(threading.get_ident())
        raise AssertionError("Gemini Web batch screening must not use per-paper screening")

    class SinglePaperGeminiWebAutomation(FakeGeminiWebAutomation):
        def submit_prompt_and_get_response(self, prompt):
            browser_thread_ids.append(threading.get_ident())
            paper_id = _paper_ids_from_prompt(prompt)[0]
            return json.dumps({
                "decisions": [
                    {"id": paper_id, "decision": "Include", "reason": "Relevant."}
                ]
            })

    monkeypatch.setattr(bulk_screen, "resolve_processing_engine", fake_resolve_processing_engine)
    monkeypatch.setattr(bulk_screen, "extract_semantic_frame", lambda **kwargs: {})
    monkeypatch.setattr(bulk_screen, "process_paper", fake_process_paper)
    monkeypatch.setattr(gemini_web_screening, "GeminiWebAutomation", SinglePaperGeminiWebAutomation)

    try:
        bulk_screen.screen_csv(
            csv_path=str(csv_path),
            research_question="Question",
            output_path=str(checkpoint_path),
            mode=GEMINI_WEB_ENGINE,
            screening_engine=GEMINI_WEB_ENGINE,
            semantic_strategy=LITSYNC_WORKFLOW,
            two_stage_enabled=False,
        )

        assert browser_thread_ids
        assert process_thread_ids == []
        assert checkpoint_path.exists()
    finally:
        if csv_path.exists():
            csv_path.unlink()
        if checkpoint_path.exists():
            checkpoint_path.unlink()


def test_gemini_web_content_bound_ids_reject_stale_numeric_response():
    row = pd.Series({"Title": "Real Paper", "Abstract": "Real Abstract"})
    paper = gemini_web_screening._row_to_screening_paper(
        index=1,
        row=row,
        title_col="Title",
        abstract_col="Abstract",
        id_col=None,
        used_ids=set(),
    )
    assert paper.paper_id != "1"
    try:
        from gemini_web_parser import parse_gemini_screening_response

        parse_gemini_screening_response(
            '{"decisions":[{"id":"1","decision":"Include","reason":"Relevant."}]}',
            {paper.paper_id},
        )
    except GeminiResponseParseError:
        return
    raise AssertionError("stale numeric Gemini response should not validate")


def test_gemini_api_engine_uses_user_supplied_key(monkeypatch):
    captured = {}

    def fake_ask_gemini(prompt, model="", api_key=None):
        captured["prompt"] = prompt
        captured["model"] = model
        captured["api_key"] = api_key
        return '{"decision": "KEEP"}'

    import gemini_client

    monkeypatch.setattr(gemini_client, "ask_gemini", fake_ask_gemini)

    engine = processing_engines.resolve_processing_engine(
        GEMINI_API_ENGINE,
        gemini_api_key="test-key",
    )
    response = engine.ask("Prompt", model="gemini-2.5-flash")

    assert response == '{"decision": "KEEP"}'
    assert captured == {
        "prompt": "Prompt",
        "model": "gemini-2.5-flash",
        "api_key": "test-key",
    }
