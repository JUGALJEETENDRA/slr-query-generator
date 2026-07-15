import json
import re

import pandas as pd

import bulk_screen
from bulk_screen import ScreeningProgress, ScreeningSession
from gemini_web_prompt import ScreeningPaper
from gemini_web_screening import (
    GEMINI_WEB_VERSION, _execute_batch, screen_csv_with_gemini_web,
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
                items.append(_item(paper_id, decision="MAYBE", certainty="LOW"))
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
