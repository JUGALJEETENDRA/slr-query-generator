import json
import re

import pandas as pd
import pytest

import bulk_screen
from bulk_screen import ScreeningProgress, ScreeningSession
from gemini_web_prompt import ScreeningPaper
from gemini_web_screening import (
    GEMINI_WEB_VERSION, _execute_batch, _needs_critic, screen_csv_with_gemini_web,
)
from local_ai.contracts import ReviewProtocol


def _protocol_json():
    return json.dumps({
        "schema_version": "2.0", "protocol_id": "", "research_question": "Invented RQ?",
        "research_context": "Context only", "objective": "Assess invented fit.",
        "scope_interpretation": "Papers must directly study the invented relationship.",
        "criteria": [{
            "id": "inc1", "kind": "inclusion", "description": "Directly studies the relationship",
            "required": True, "expected_evidence": "Title or abstract evidence",
            "source": "research_question",
        }],
        "expected_relationships": [], "ambiguities": [], "semantic_boundaries": [],
        "prompt_version": GEMINI_WEB_VERSION, "model": "gemini-web",
    })


def _item(paper_id, *, decision="KEEP", certainty="HIGH"):
    return {
        "schema_version": "2.0", "paper_id": paper_id, "certainty": certainty,
        "summary": "The paper directly studies the relationship.",
        "criteria": [{
            "criterion_id": "inc1", "verdict": "MET", "rationale": "Direct title evidence.",
            "evidence": [{"source": "title", "evidence_id": "title_001"}],
        }],
        "contradictions": [], "missing_information": [], "decision": decision,
        "confidence": .9 if certainty == "HIGH" else .5,
        "reason": "Directly relevant." if decision == "KEEP" else "Needs a critic.",
        "uncertainty": [] if decision == "KEEP" else ["Borderline primary assessment."],
    }


class FakeGeminiBrowser:
    instances = []

    def __init__(self, config):
        self.prompts = []
        self.new_chat_calls = 0
        self.recovery_calls = 0
        FakeGeminiBrowser.instances.append(self)

    def __enter__(self): self.start_new_job_chat(); return self
    def __exit__(self, *args): return None
    def start_new_job_chat(self): self.new_chat_calls += 1
    def recover_job_chat(self): self.recovery_calls += 1

    def submit_prompt_and_get_response(self, prompt):
        self.prompts.append(prompt)
        if "Compile an immutable review protocol" in prompt:
            return "```json\n" + _protocol_json() + "\n```"
        ids = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
        critic = "adversarial systematic-review critic" in prompt
        items = []
        for offset, paper_id in enumerate(ids):
            if critic:
                items.append(_item(paper_id))
            elif offset == 0:
                items.append(_item(paper_id, decision="KEEP", certainty="LOW"))
            else:
                items.append(_item(paper_id))
        return json.dumps({"items": items})


def test_twelve_papers_use_one_chat_and_primary_batches_5_5_2(tmp_path):
    FakeGeminiBrowser.instances.clear()
    frame = pd.DataFrame({
        "Title": [f"Paper {i}" for i in range(12)],
        "Abstract": [f"Abstract evidence {i}." for i in range(12)],
    })
    progress = ScreeningProgress()
    session = ScreeningSession()
    assert progress.start_job("web-job")
    progress.begin_screening("web-job", 12, GEMINI_WEB_VERSION)
    output = tmp_path / "runs" / "screened-web.csv"

    summary = screen_csv_with_gemini_web(
        frame=frame, valid=frame, title_col="Title", abstract_col="Abstract",
        research_question="Invented RQ?", research_context="Context only",
        inclusion_criteria="", exclusion_criteria="", output_path=str(output),
        job_id="web-job", input_fingerprint="file-hash", resume=False, limit=0,
        progress=progress, screening_session=session, browser_factory=FakeGeminiBrowser,
    )

    assert len(FakeGeminiBrowser.instances) == 1
    browser = FakeGeminiBrowser.instances[0]
    assert browser.new_chat_calls == 1
    assert browser.recovery_calls == 0
    assert '"research_context_for_interpretation_only": "Context only"' in browser.prompts[0]
    primary = [p for p in browser.prompts if "FIVE-OR-FEWER PAPER BATCH" in p]
    critic = [p for p in browser.prompts if "RISKY PAPER BATCH" in p]
    assert [len(set(re.findall(r'"paper_id":\s*"([^"]+)"', p))) for p in primary] == [5, 5, 2]
    assert all('"research_context": "Context only"' in prompt for prompt in primary)
    assert len(critic) == 1
    assert summary["architecture_version"] == GEMINI_WEB_VERSION
    assert summary["keep"] == 12
    saved = pd.read_csv(output)
    assert set(saved["Prompt_Version"]) == {GEMINI_WEB_VERSION}
    assert saved["Validation_Status"].eq("validated").all()
    assert saved["Escalated"].sum() == 3

    resumed_progress = ScreeningProgress()
    resumed_session = ScreeningSession()
    assert resumed_progress.start_job("resumed-job")
    resumed_progress.begin_screening("resumed-job", 12, GEMINI_WEB_VERSION)
    resumed = screen_csv_with_gemini_web(
        frame=frame, valid=frame, title_col="Title", abstract_col="Abstract",
        research_question="Invented RQ?", research_context="Context only",
        inclusion_criteria="", exclusion_criteria="", output_path=str(output),
        job_id="resumed-job", input_fingerprint="file-hash", resume=True, limit=0,
        progress=resumed_progress, screening_session=resumed_session,
        browser_factory=FakeGeminiBrowser,
    )
    assert resumed["resumed_count"] == 12
    assert len(FakeGeminiBrowser.instances) == 1


def test_malformed_batch_retries_then_splits_only_that_batch():
    protocol = ReviewProtocol.model_validate_json(_protocol_json()).with_identity()
    papers = [ScreeningPaper(str(i), f"Paper {i}", f"Evidence {i}.") for i in range(5)]

    class SplitBrowser:
        def __init__(self): self.sizes = []; self.recoveries = 0
        def recover_job_chat(self): self.recoveries += 1
        def submit_prompt_and_get_response(self, prompt):
            ids = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            self.sizes.append(len(ids))
            if len(ids) > 2:
                return '{"items": []}'
            return json.dumps({"items": [_item(paper_id) for paper_id in ids]})

    browser = SplitBrowser()
    retries = []
    result = _execute_batch(
        browser, protocol, papers, critic=False, prior=None,
        record_retry=lambda: retries.append(1),
    )
    assert set(result) == {"0", "1", "2", "3", "4"}
    assert browser.sizes == [5, 5, 2, 3, 3, 1, 2]
    assert browser.recoveries == 0
    assert len(retries) == 4


def test_browser_timeouts_do_not_recursively_split_batch():
    protocol = ReviewProtocol.model_validate_json(_protocol_json()).with_identity()
    papers = [ScreeningPaper(str(index), f"Paper {index}", "Evidence.") for index in range(5)]

    class TimedOutBrowser:
        def __init__(self): self.calls = 0; self.recoveries = 0
        def recover_job_chat(self): self.recoveries += 1
        def submit_prompt_and_get_response(self, prompt):
            self.calls += 1
            raise TimeoutError("browser did not respond")

    browser = TimedOutBrowser()
    retries = []
    result = _execute_batch(
        browser, protocol, papers, critic=False, prior=None,
        record_retry=lambda: retries.append(1),
    )

    assert browser.calls == 2
    assert browser.recoveries == 2
    assert len(retries) == 2
    assert set(result) == {"0", "1", "2", "3", "4"}
    assert all(item.decision == "MAYBE" for item in result.values())


def test_bulk_gemini_web_never_constructs_generic_external_orchestrator(monkeypatch, tmp_path):
    source = tmp_path / "papers.csv"
    pd.DataFrame({"Title": ["A"], "Abstract": ["Evidence."]}).to_csv(source, index=False)
    called = []

    def dedicated(**kwargs):
        called.append(kwargs)
        bulk_screen.PROGRESS.finish(kwargs["job_id"])
        return {"keep": 1, "maybe": 0, "reject": 0, "architecture_version": GEMINI_WEB_VERSION}

    monkeypatch.setattr("gemini_web_screening.screen_csv_with_gemini_web", dedicated)
    monkeypatch.setattr(
        "external_ai.orchestrator.ExternalAIScreeningOrchestrator",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("generic route used")),
    )
    result = bulk_screen.screen_csv(
        str(source), "Invented RQ?", output_path=str(tmp_path / "out.csv"),
        screening_engine="gemini_web", resume=False,
    )
    assert result["architecture_version"] == GEMINI_WEB_VERSION
    assert len(called) == 1


def test_critic_targets_risky_definitive_results_not_safe_maybes():
    base = {
        "validation_status": "validated", "contradictions": [], "decision_risk": "HIGH",
    }
    assert not _needs_critic({**base, "decision": "MAYBE"})
    assert _needs_critic({**base, "decision": "KEEP"})
    assert _needs_critic({**base, "decision": "REJECT"})
    assert _needs_critic({**base, "decision": "MAYBE", "contradictions": ["conflict"]})
    assert _needs_critic({**base, "decision": "MAYBE", "validation_status": "unresolved"})


def test_critic_prompt_is_prediction_blind():
    from gemini_web_prompt import build_structured_critic_prompt

    protocol = ReviewProtocol.model_validate_json(_protocol_json()).with_identity()
    prompt = build_structured_critic_prompt(
        protocol=protocol.model_dump(mode="json"),
        papers=[ScreeningPaper("p1", "Title", "Abstract evidence.")],
        prior={"p1": {
            "decision": "REJECT", "reason": "DISTINCTIVE_HIDDEN_REASON",
            "criteria": [{"rationale": "DISTINCTIVE_HIDDEN_CRITERION"}],
            "validation_errors": ["missing evidence"], "contradictions": [],
        }},
        schema={"type": "object"},
    )

    assert "DISTINCTIVE_HIDDEN_REASON" not in prompt
    assert "DISTINCTIVE_HIDDEN_CRITERION" not in prompt
    assert '"decision": "REJECT"' not in prompt
    assert "missing evidence" in prompt
    assert "missing information is UNCLEAR" in prompt


def test_invalid_critic_does_not_overwrite_valid_primary(tmp_path):
    class InvalidCriticBrowser(FakeGeminiBrowser):
        def submit_prompt_and_get_response(self, prompt):
            self.prompts.append(prompt)
            if "Compile an immutable review protocol" in prompt:
                return _protocol_json()
            ids = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            items = [_item(paper_id, decision="KEEP", certainty="LOW") for paper_id in ids]
            if "adversarial systematic-review critic" in prompt:
                for item in items:
                    item["criteria"][0]["evidence"][0]["evidence_id"] = "invented_999"
            return json.dumps({"items": items})

    frame = pd.DataFrame({"Title": ["Direct relationship"], "Abstract": ["Direct evidence."]})
    progress = ScreeningProgress()
    session = ScreeningSession()
    assert progress.start_job("invalid-critic")
    progress.begin_screening("invalid-critic", 1, GEMINI_WEB_VERSION)
    output = tmp_path / "runs" / "screened.csv"

    summary = screen_csv_with_gemini_web(
        frame=frame, valid=frame, title_col="Title", abstract_col="Abstract",
        research_question="Invented RQ?", research_context="Context only",
        inclusion_criteria="", exclusion_criteria="", output_path=str(output),
        job_id="invalid-critic", input_fingerprint="invalid-critic-file", resume=False, limit=0,
        progress=progress, screening_session=session, browser_factory=InvalidCriticBrowser,
    )

    saved = pd.read_csv(output).iloc[0]
    trace = json.loads(saved["Layer_Trace_JSON"])
    assert summary["keep"] == 1
    assert saved["Decision"] == "KEEP"
    assert saved["Validation_Status"] == "validated"
    assert bool(saved["Escalated"])
    assert trace[-1]["validation_status"] == "ignored_invalid"


def test_protocol_failure_closes_browser(tmp_path):
    class BrokenProtocolBrowser:
        exited = False

        def __init__(self, config): pass
        def __enter__(self): return self
        def __exit__(self, *args): BrokenProtocolBrowser.exited = True
        def recover_job_chat(self): pass
        def submit_prompt_and_get_response(self, prompt): return "not structured JSON"

    frame = pd.DataFrame({"Title": ["Paper"], "Abstract": ["Evidence."]})
    progress = ScreeningProgress()
    session = ScreeningSession()
    assert progress.start_job("bad-protocol")
    progress.begin_screening("bad-protocol", 1, GEMINI_WEB_VERSION)

    with pytest.raises(RuntimeError, match="valid screening protocol"):
        screen_csv_with_gemini_web(
            frame=frame, valid=frame, title_col="Title", abstract_col="Abstract",
            research_question="Unique broken protocol RQ?", research_context="",
            inclusion_criteria="", exclusion_criteria="",
            output_path=str(tmp_path / "runs" / "screened.csv"),
            job_id="bad-protocol", input_fingerprint="bad-protocol-file", resume=False, limit=0,
            progress=progress, screening_session=session, browser_factory=BrokenProtocolBrowser,
        )

    assert BrokenProtocolBrowser.exited


def test_malformed_critic_fallback_does_not_overwrite_valid_primary(tmp_path):
    class MalformedCriticBrowser(FakeGeminiBrowser):
        def submit_prompt_and_get_response(self, prompt):
            self.prompts.append(prompt)
            if "Compile an immutable review protocol" in prompt:
                return _protocol_json()
            if "adversarial systematic-review critic" in prompt:
                return "malformed"
            ids = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            return json.dumps({
                "items": [_item(paper_id, decision="KEEP", certainty="LOW") for paper_id in ids]
            })

    frame = pd.DataFrame({"Title": ["Direct relationship"], "Abstract": ["Direct evidence."]})
    progress = ScreeningProgress()
    session = ScreeningSession()
    assert progress.start_job("malformed-critic")
    progress.begin_screening("malformed-critic", 1, GEMINI_WEB_VERSION)
    output = tmp_path / "runs" / "screened.csv"

    summary = screen_csv_with_gemini_web(
        frame=frame, valid=frame, title_col="Title", abstract_col="Abstract",
        research_question="Invented RQ?", research_context="Context only",
        inclusion_criteria="", exclusion_criteria="", output_path=str(output),
        job_id="malformed-critic", input_fingerprint="malformed-critic-file", resume=False, limit=0,
        progress=progress, screening_session=session, browser_factory=MalformedCriticBrowser,
    )

    saved = pd.read_csv(output).iloc[0]
    assert summary["keep"] == 1
    assert saved["Decision"] == "KEEP"
    assert json.loads(saved["Layer_Trace_JSON"])[-1]["validation_status"] == "ignored_invalid"


def test_response_completion_recognizes_only_complete_json():
    from gemini_web_automation import GeminiWebAutomation

    assert GeminiWebAutomation._is_complete_json('{"items": []}')
    assert GeminiWebAutomation._is_complete_json('```json\n{"items": []}\n```')
    assert not GeminiWebAutomation._is_complete_json('{"items": [')
    assert not GeminiWebAutomation._is_complete_json('Gemini is still thinking')


def test_browser_recovery_relaunches_after_new_chat_failure(monkeypatch):
    from gemini_web_automation import GeminiWebAutomation

    browser = GeminiWebAutomation()
    calls = []
    monkeypatch.setattr(browser, "start_new_job_chat", lambda: (_ for _ in ()).throw(RuntimeError("crash")))
    monkeypatch.setattr(browser, "close", lambda: calls.append("close"))
    monkeypatch.setattr(browser, "start", lambda: calls.append("start"))

    browser.recover_job_chat()

    assert calls == ["close", "start"]


def test_interrupted_run_resumes_without_duplicate_rows(tmp_path):
    class SimulatedInterruption(BaseException):
        pass

    class InterruptingBrowser(FakeGeminiBrowser):
        def __init__(self, config):
            super().__init__(config)
            self.primary_calls = 0

        def submit_prompt_and_get_response(self, prompt):
            if "FIVE-OR-FEWER PAPER BATCH" in prompt:
                self.primary_calls += 1
                if self.primary_calls == 2:
                    raise SimulatedInterruption("process stopped between batches")
            return super().submit_prompt_and_get_response(prompt)

    frame = pd.DataFrame({
        "Title": [f"Paper {index}" for index in range(7)],
        "Abstract": [f"Direct evidence {index}." for index in range(7)],
    })
    output = tmp_path / "runs" / "screened.csv"
    first_progress = ScreeningProgress()
    first_session = ScreeningSession()
    assert first_progress.start_job("interrupted")
    first_progress.begin_screening("interrupted", 7, GEMINI_WEB_VERSION)
    with pytest.raises(SimulatedInterruption):
        screen_csv_with_gemini_web(
            frame=frame, valid=frame, title_col="Title", abstract_col="Abstract",
            research_question="Resume RQ?", research_context="",
            inclusion_criteria="", exclusion_criteria="", output_path=str(output),
            job_id="interrupted", input_fingerprint="resume-file", resume=False, limit=0,
            progress=first_progress, screening_session=first_session,
            browser_factory=InterruptingBrowser,
        )
    partial = pd.read_csv(output)
    assert len(partial) == 5

    FakeGeminiBrowser.instances.clear()
    resumed_progress = ScreeningProgress()
    resumed_session = ScreeningSession()
    assert resumed_progress.start_job("resumed")
    resumed_progress.begin_screening("resumed", 7, GEMINI_WEB_VERSION)
    summary = screen_csv_with_gemini_web(
        frame=frame, valid=frame, title_col="Title", abstract_col="Abstract",
        research_question="Resume RQ?", research_context="",
        inclusion_criteria="", exclusion_criteria="", output_path=str(output),
        job_id="resumed", input_fingerprint="resume-file", resume=True, limit=0,
        progress=resumed_progress, screening_session=resumed_session,
        browser_factory=FakeGeminiBrowser,
    )

    saved = pd.read_csv(output)
    primary_prompts = [
        prompt for prompt in FakeGeminiBrowser.instances[0].prompts
        if "FIVE-OR-FEWER PAPER BATCH" in prompt
    ]
    assert summary["resumed_count"] == 5
    assert len(saved) == 7
    assert saved["Source_Row_Index"].nunique() == 7
    assert len(set(re.findall(r'"paper_id":\s*"([^"]+)"', primary_prompts[0]))) == 2
