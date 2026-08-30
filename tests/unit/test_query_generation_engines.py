from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from litsync_app import app as server
from litsync_app.query import engines as query_engines
from litsync_app.query import generator as query_module
from litsync_app.query.engines import GeminiWebQueryEngine
from litsync_app.integrations.gemini_browser import GeminiWebConfig
from litsync_app.query.generator import (
    GeminiDirectConceptProposal,
    GeneratedQueryBundle,
    StructuredQueryDraft,
    generate_query_bundle,
)
from litsync_app.screening.local.engine import (
    GenerationResult,
    LocalAIError,
    LocalAIOutputError,
)
from litsync_app.screening.local.hardware import HardwareSnapshot, RuntimeProfile


QUESTION = "Federated learning in hospitals"
VALID_DRAFT = {
    "groups": [
        {
            "label": "Method", "role": "technology",
            "terms": ["Federated learning"], "source_spans": ["Federated learning"],
        },
        {
            "label": "Setting", "role": "domain",
            "terms": ["hospitals"], "source_spans": ["hospitals"],
        },
    ],
    "needs_grounding": False,
    "uncertain_terms": [],
}
VALID_AI_PROPOSAL = {
    "concepts": [
        {
            "label": "Technology",
            "balanced_terms": ["federated learning"],
            "high_recall_terms": ["federated learning", "distributed learning"],
        },
        {
            "label": "Setting",
            "balanced_terms": ["hospitals"],
            "high_recall_terms": ["hospitals", "clinical settings"],
        },
    ],
}


def _profile() -> RuntimeProfile:
    hardware = HardwareSnapshot(
        total_ram_gb=16.0, available_ram_gb=8.0, cpu_cores=8, platform="Test",
        gpu_name="", gpu_vram_gb=0.0, installed_models={"qwen3.5:4b": 1},
    )
    return RuntimeProfile(
        requested_tier="auto", resolved_tier="balanced", resource_profile="balanced",
        fast_model="qwen3.5:4b", strong_model="qwen3.5:4b", num_ctx=4096,
        keep_alive="30m", concurrency=1, memory_reserve_ratio=0.2,
        downgrade_reasons=(), hardware=hardware, calibration={},
    )


class DraftEngine:
    def __init__(self, value=None, error=None):
        self.value = value or VALID_DRAFT
        self.error = error
        self.calls = []

    def generate(self, model, prompt, schema, *, timeout_seconds=None):
        self.calls.append((model, prompt, schema, timeout_seconds))
        if self.error is not None:
            raise self.error
        return GenerationResult(value=self.value, model=model, elapsed_seconds=0.01)


class BrowserFactory:
    def __init__(self, outcomes, *, fail_start=False):
        self.outcomes = list(outcomes)
        self.fail_start = fail_start
        self.instances = []

    def __call__(self):
        index = len(self.instances)
        outcome = self.outcomes[min(index, len(self.outcomes) - 1)]
        factory = self

        class Browser:
            def __init__(self):
                self.closed = False
                self.prompts = []

            def start(self):
                if factory.fail_start:
                    raise RuntimeError("startup failed")

            def submit_prompt_and_get_response(self, prompt):
                self.prompts.append(prompt)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

            def close(self):
                self.closed = True

        browser = Browser()
        self.instances.append(browser)
        return browser


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class BudgetBrowserFactory:
    def __init__(self, clock, outcomes, advances, *, close_error=False):
        self.clock = clock
        self.outcomes = list(outcomes)
        self.advances = list(advances)
        self.close_error = close_error
        self.instances = []

    def __call__(self):
        index = len(self.instances)
        outcome = self.outcomes[min(index, len(self.outcomes) - 1)]
        start_advance, submit_advance = self.advances[min(index, len(self.advances) - 1)]
        factory = self

        class Page:
            def __init__(self):
                self.timeouts = []

            def set_default_timeout(self, value):
                self.timeouts.append(("default", value))

            def set_default_navigation_timeout(self, value):
                self.timeouts.append(("navigation", value))

        class Browser:
            def __init__(self):
                self.config = GeminiWebConfig()
                self._page = None
                self.closed = False
                self.start_timeout = None
                self.submit_timeout = None

            def start(self):
                self.start_timeout = self.config.response_timeout_ms
                factory.clock.advance(start_advance)
                self._page = Page()

            def submit_prompt_and_get_response(self, prompt):
                del prompt
                self.submit_timeout = self.config.response_timeout_ms
                factory.clock.advance(submit_advance)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

            def close(self):
                self.closed = True
                if factory.close_error:
                    raise RuntimeError("cleanup failed")

        browser = Browser()
        self.instances.append(browser)
        return browser


@pytest.mark.parametrize(
    "raw",
    [json.dumps(VALID_DRAFT), f"```json\n{json.dumps(VALID_DRAFT)}\n```"],
)
def test_gemini_adapter_parses_plain_json_or_one_json_fence(raw):
    browsers = BrowserFactory([raw])
    result = GeminiWebQueryEngine(browsers).generate(
        "gemini_web", "draft prompt", StructuredQueryDraft,
    )
    assert result.value["groups"][0]["label"] == VALID_DRAFT["groups"][0]["label"]
    assert result.value["groups"][1]["label"] == VALID_DRAFT["groups"][1]["label"]
    assert result.value["needs_grounding"] is False
    assert result.value["uncertain_terms"] == []
    assert len(browsers.instances) == 1
    assert browsers.instances[0].closed
    assert browsers.instances[0].prompts[0].count("StructuredQueryDraft JSON schema:") == 1


def test_internal_retry_is_distinct_from_one_generator_engine_call():
    browsers = BrowserFactory([TimeoutError("slow"), json.dumps(VALID_AI_PROPOSAL)])
    engine = GeminiWebQueryEngine(browsers)
    bundle = generate_query_bundle(
        QUESTION, processing_engine="gemini_web", engine=engine,
    )
    assert len(browsers.instances) == 2
    assert all(browser.closed for browser in browsers.instances)
    assert bundle.concepts["generation_status"] == "ai_assisted_expansion"


def test_invalid_schema_and_malformed_transport_each_receive_one_retry():
    invalid = BrowserFactory([json.dumps({"required_groups": []})])
    with pytest.raises(LocalAIOutputError, match="after two attempts"):
        GeminiWebQueryEngine(invalid).generate(
            "gemini_web", "prompt", GeminiDirectConceptProposal,
        )
    assert len(invalid.instances) == 2
    assert "CORRECTION ATTEMPT" in invalid.instances[1].prompts[0]
    malformed = BrowserFactory(["not json", "still not json"])
    with pytest.raises(LocalAIOutputError, match="after two attempts"):
        GeminiWebQueryEngine(malformed).generate(
            "gemini_web", "prompt", GeminiDirectConceptProposal,
        )
    assert len(malformed.instances) == 2
    assert all(browser.closed for browser in malformed.instances)


def test_failed_startup_retries_with_guarded_cleanup():
    browsers = BrowserFactory(["unused"], fail_start=True)
    with pytest.raises(LocalAIError, match="startup failed"):
        GeminiWebQueryEngine(browsers).generate(
            "gemini_web", "prompt", GeminiDirectConceptProposal,
        )
    assert len(browsers.instances) == 2
    assert all(browser.closed for browser in browsers.instances)


def test_gemini_budget_is_shared_across_retries(monkeypatch):
    clock = FakeClock()
    browsers = BudgetBrowserFactory(
        clock,
        ["not json", json.dumps(VALID_AI_PROPOSAL)],
        [(1.0, 1.0), (1.0, 1.0)],
    )
    monkeypatch.setattr(query_engines.time, "monotonic", clock)
    result = GeminiWebQueryEngine(browsers).generate(
        "gemini_web", "prompt", GeminiDirectConceptProposal, timeout_seconds=10,
    )
    assert result.value["concepts"]
    assert len(browsers.instances) == 2
    first, second = browsers.instances
    assert first.start_timeout == 10000
    assert first.submit_timeout == 9000
    assert second.start_timeout == 8000
    assert second.submit_timeout == 7000
    assert all(browser.closed for browser in browsers.instances)


def test_supported_timeout_caps_existing_browser_config_without_lengthening():
    @dataclass(frozen=True)
    class Config:
        ready_timeout_ms: int = 4000
        response_timeout_ms: int = 7000
        no_container_timeout_ms: int = 3000

    class Target:
        def __init__(self):
            self.config = Config()
            self._page = None

    target = Target()
    query_engines._apply_supported_timeout(target, 10.0)
    assert target.config.ready_timeout_ms == 4000
    assert target.config.response_timeout_ms == 7000
    assert target.config.no_container_timeout_ms == 3000

    query_engines._apply_supported_timeout(target, 2.0)
    assert target.config.ready_timeout_ms == 2000
    assert target.config.response_timeout_ms == 2000
    assert target.config.no_container_timeout_ms == 2000


def test_gemini_budget_exhaustion_prevents_second_browser_and_preserves_timeout(monkeypatch):
    clock = FakeClock()
    browsers = BudgetBrowserFactory(clock, ["not json"], [(1.0, 5.0)], close_error=True)
    monkeypatch.setattr(query_engines.time, "monotonic", clock)
    with pytest.raises(LocalAIError, match="budget expired"):
        GeminiWebQueryEngine(browsers).generate(
            "gemini_web", "prompt", GeminiDirectConceptProposal, timeout_seconds=5,
        )
    assert len(browsers.instances) == 1
    assert browsers.instances[0].closed


def test_schema_validation_retry_uses_remaining_budget(monkeypatch):
    clock = FakeClock()
    browsers = BudgetBrowserFactory(
        clock, [json.dumps({"required_groups": []})], [(0.5, 0.5)],
    )
    monkeypatch.setattr(query_engines.time, "monotonic", clock)
    with pytest.raises(LocalAIOutputError, match="after two attempts"):
        GeminiWebQueryEngine(browsers).generate(
            "gemini_web", "prompt", GeminiDirectConceptProposal, timeout_seconds=10,
        )
    assert len(browsers.instances) == 2
    assert all(browser.closed for browser in browsers.instances)


def test_local_remains_default_and_independently_selectable(monkeypatch):
    engine = DraftEngine()
    monkeypatch.setattr(query_module, "resolve_runtime_profile", _profile)
    monkeypatch.setattr(query_module, "OllamaStructuredEngine", lambda profile: engine)
    bundle = generate_query_bundle(QUESTION)
    assert len(engine.calls) == 1
    assert engine.calls[0][2] is StructuredQueryDraft
    assert bundle.concepts["processing_engine"] == "local"
    assert bundle.concepts["deadline_seconds"] == 15.0


def test_gemini_selection_builds_adapter_without_local_profile(monkeypatch):
    engine = DraftEngine(VALID_AI_PROPOSAL)
    monkeypatch.setattr(query_engines, "GeminiWebQueryEngine", lambda: engine)
    monkeypatch.setattr(
        query_module, "resolve_runtime_profile",
        lambda: (_ for _ in ()).throw(AssertionError("local profile must not resolve")),
    )
    bundle = generate_query_bundle(QUESTION, processing_engine="gemini_web")
    assert len(engine.calls) == 1
    assert issubclass(engine.calls[0][2], GeminiDirectConceptProposal)
    assert bundle.concepts["model"] == "gemini_web"
    assert bundle.concepts["deadline_seconds"] == 120.0


def test_unsupported_query_engine_is_rejected():
    with pytest.raises(ValueError, match="Unsupported query-generation engine"):
        generate_query_bundle("A valid question", processing_engine="gemini_api")


def test_local_engine_failure_returns_local_fallback_metadata():
    bundle = generate_query_bundle(
        QUESTION, processing_engine="local", profile=_profile(),
        engine=DraftEngine(error=LocalAIError("offline")), deadline_seconds=2,
    )
    assert bundle.concepts["fallback_reason"].startswith("local_draft_failed")
    assert bundle.google_scholar


def test_gemini_engine_failure_is_not_replaced_by_parser_fallback():
    with pytest.raises(LocalAIError, match="offline"):
        generate_query_bundle(
            QUESTION, processing_engine="gemini_web",
            engine=DraftEngine(error=LocalAIError("offline")), deadline_seconds=2,
        )


def _api_bundle() -> GeneratedQueryBundle:
    return GeneratedQueryBundle("g", "s", "w", "i", "p", {"groups": []})


@pytest.mark.parametrize(
    ("request_json", "expected_engine"),
    [
        ({"question": "  Test question  "}, "local"),
        ({"question": "Test question", "processing_engine": "gemini_web"},
         "gemini_web"),
    ],
)
def test_generate_api_defaults_and_forwards_engine(monkeypatch, request_json, expected_engine):
    calls = []

    def fake_generate(question, **kwargs):
        calls.append((question, kwargs))
        return _api_bundle()

    monkeypatch.setattr(server, "generate_query_bundle", fake_generate)
    response = TestClient(server.app).post("/generate", json=request_json)
    assert response.status_code == 200
    assert response.json()["concepts"]["question"] == "Test question"
    assert calls == [("Test question", {"processing_engine": expected_engine})]


def test_generate_api_rejects_blank_question_and_unsupported_engine():
    client = TestClient(server.app)
    assert client.post("/generate", json={"question": "  "}).json() == {
        "status": "error", "message": "Enter a research question.",
    }
    invalid = client.post(
        "/generate", json={"question": "Test", "processing_engine": "gemini_api"},
    ).json()
    assert invalid["status"] == "error"


def test_ui_keeps_engine_and_query_version_controls_with_honest_wording():
    html = Path("web/slr_query_generator.html").read_text(encoding="utf-8")
    assert '<option value="local" selected>Local Ollama</option>' in html
    assert '<option value="gemini_web">Gemini Web</option>' in html
    assert "processing_engine: selectedEngine" in html
    assert 'data-query-version="balanced"' in html
    assert 'data-query-version="high_recall"' in html
    assert "AI-assisted query expansion" in html
    assert "not independently checked against academic literature" in html
