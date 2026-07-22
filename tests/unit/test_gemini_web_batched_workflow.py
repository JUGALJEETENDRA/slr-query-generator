import json
import re

import pandas as pd
import pytest

import bulk_screen
from bulk_screen import ScreeningProgress, ScreeningSession
from gemini_web_prompt import ScreeningPaper
from gemini_web_screening import (
    GEMINI_WEB_VERSION, WebPaperAssessment, _acronym_grounding_errors, _critic_route,
    _execute_batch, _needs_critic, _public_result, _scope_support_errors,
    screen_csv_with_gemini_web,
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
            "scope_support": "SUBSTANTIVE",
        }],
        "contradictions": [], "missing_information": [], "decision": decision,
        "confidence": .9 if certainty == "HIGH" else .5,
        "reason": "Directly relevant." if decision == "KEEP" else "Needs a critic.",
        "uncertainty": [] if decision == "KEEP" else ["Borderline primary assessment."],
    }


def _reject_item(paper_id, *, certainty="HIGH"):
    item = _item(paper_id, decision="REJECT", certainty=certainty)
    item["summary"] = "The required relationship is affirmatively mismatched."
    item["criteria"][0].update({
        "verdict": "NOT_MET", "rationale": "The title establishes a different relationship.",
    })
    item["confidence"] = .9
    item["reason"] = "The supplied evidence affirmatively establishes a mismatch."
    item["uncertainty"] = []
    return item


class FakeGeminiBrowser:
    instances = []

    def __init__(self, config):
        self.prompts = []
        self.new_chat_calls = 0
        self.recovery_calls = 0
        self.recovery_actions = []
        FakeGeminiBrowser.instances.append(self)

    def __enter__(self): self.start_new_job_chat(); return self
    def __exit__(self, *args): return None
    def start_new_job_chat(self): self.new_chat_calls += 1
    def recover_job_chat(self): self.recovery_calls += 1
    def note_recovery(self, action): self.recovery_actions.append(action)

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
    assert browser.new_chat_calls == 3
    assert browser.recovery_calls == 0
    assert browser.recovery_actions == [
        "protocol_to_primary_clean_chat", "primary_to_critic_clean_chat",
    ]
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
    assert all(
        criterion["scope_support"] == "SUBSTANTIVE"
        for value in saved["Criteria_JSON"]
        for criterion in json.loads(value)
    )

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


def test_no_container_timeout_recycles_and_retries_only_affected_batch():
    protocol = ReviewProtocol.model_validate_json(_protocol_json()).with_identity()
    papers = [ScreeningPaper(str(index), f"Paper {index}", "Evidence.") for index in range(5)]

    class RecoveredBrowser:
        def __init__(self): self.calls = 0; self.recoveries = []
        def recover_transport_failure(self, *, exhausted=False):
            self.recoveries.append(exhausted)
        def submit_prompt_and_get_response(self, prompt):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("no new response container")
            ids = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            return json.dumps({"items": [_item(paper_id) for paper_id in ids]})

    browser = RecoveredBrowser()
    retries = []
    result = _execute_batch(
        browser, protocol, papers, critic=False, prior=None,
        record_retry=lambda: retries.append(1),
    )

    assert browser.calls == 2
    assert browser.recoveries == [False]
    assert len(retries) == 1
    assert set(result) == {"0", "1", "2", "3", "4"}
    assert all(item.decision == "KEEP" for item in result.values())


def _grounding_protocol(concept="blockchain technology"):
    return ReviewProtocol.model_validate({
        "schema_version": "2.0", "protocol_id": "", "research_question": f"How is {concept} used?",
        "objective": f"Assess use of {concept}.", "scope_interpretation": f"Require {concept}.",
        "criteria": [{
            "id": "required_concept", "kind": "inclusion",
            "description": f"The paper substantively studies {concept}", "required": True,
            "expected_evidence": f"The title or abstract explicitly supports {concept}",
            "source": "research_question",
        }],
        "expected_relationships": [], "ambiguities": [], "semantic_boundaries": [],
        "prompt_version": GEMINI_WEB_VERSION, "model": "gemini-web",
    }).with_identity()


def _grounding_keep(paper_id="p1", evidence_id="abstract_001"):
    return WebPaperAssessment.model_validate({
        "schema_version": "2.0", "paper_id": paper_id, "certainty": "HIGH",
        "summary": "The required concept is present.",
        "criteria": [{
            "criterion_id": "required_concept", "verdict": "MET",
            "rationale": "The abbreviation denotes the required concept.",
            "evidence": [{"source": "abstract", "evidence_id": evidence_id}],
            "scope_support": "SUBSTANTIVE",
        }],
        "contradictions": [], "missing_information": [], "decision": "KEEP",
        "confidence": .9, "reason": "The required concept is present.", "uncertainty": [],
    })


def test_reproduced_undefined_dlt_cannot_validate_blockchain_keep():
    title = "AI-Powered Digital Twins Revolutionizing Smart Manufacturing and Industrial IoT"
    abstract = (
        "DLT-based digital twins are virtual replicas of physical assets that use AI, machine learning, "
        "and analytics to optimize operational efficiency."
    )

    result = _public_result(
        _grounding_keep(), _grounding_protocol(), ScreeningPaper("p1", title, abstract),
        stage="gemini_web_primary", elapsed=0,
    )

    assert result["decision"] == "KEEP"
    assert result["validation_status"] == "unresolved"
    assert any("acronym" in error for error in result["validation_errors"])


def test_arbitrary_undefined_acronym_is_guarded_without_domain_rules():
    errors = _acronym_grounding_errors(
        _grounding_keep(), _grounding_protocol("quantum routing"),
        "Adaptive factory control", "QXZ coordinates production resources.",
    )

    assert errors and "QXZ" in errors[0]


def test_defined_acronym_and_independent_concept_support_remain_eligible():
    defined_errors = _acronym_grounding_errors(
        _grounding_keep(), _grounding_protocol("quantum routing"),
        "Adaptive factory control", "Quantum Routing (QXZ) coordinates production resources.",
    )
    independent_errors = _acronym_grounding_errors(
        _grounding_keep(), _grounding_protocol("quantum routing"),
        "Quantum routing for factories", "QXZ coordinates production resources.",
    )

    assert defined_errors == []
    assert independent_errors == []


def test_substantive_study_scope_supports_clear_keep():
    protocol = ReviewProtocol.model_validate_json(_protocol_json()).with_identity()
    paper = ScreeningPaper(
        "p1", "Direct relationship experiment",
        "The experiment evaluates the required relationship as its central system contribution.",
    )
    result = _public_result(
        WebPaperAssessment.model_validate(_item("p1")), protocol, paper,
        stage="gemini_web_primary", elapsed=0,
    )

    assert result["decision"] == "KEEP"
    assert result["validation_status"] == "validated"
    assert result["criteria"][0]["scope_support"] == "SUBSTANTIVE"


@pytest.mark.parametrize(
    ("description", "support"),
    [
        ("Background context mentions the relationship.", "INCIDENTAL"),
        ("A literature list defines the relationship.", "INCIDENTAL"),
        ("An example names the relationship without studying it.", "INCIDENTAL"),
        ("The supplied evidence cannot establish the relationship.", "INSUFFICIENT"),
    ],
)
def test_non_substantive_required_support_remains_unclear(description, support):
    protocol = ReviewProtocol.model_validate_json(_protocol_json()).with_identity()
    payload = _item("p1", decision="KEEP")
    payload.update({
        "decision": "MAYBE", "certainty": "BORDERLINE", "confidence": .5,
        "reason": "The required relationship is not a substantive study focus.",
        "uncertainty": ["Only incidental or insufficient support is supplied."],
    })
    payload["criteria"][0].update({
        "verdict": "UNCLEAR", "rationale": description, "scope_support": support,
    })
    result = _public_result(
        WebPaperAssessment.model_validate(payload), protocol,
        ScreeningPaper("p1", "Contextual paper", description),
        stage="gemini_web_primary", elapsed=0,
    )

    assert result["decision"] == "MAYBE"
    assert result["validation_status"] == "validated"
    assert result["criteria"][0]["verdict"] == "UNCLEAR"
    assert result["criteria"][0]["scope_support"] == support


def test_incidental_met_is_invalid_and_routes_to_critic():
    protocol = ReviewProtocol.model_validate_json(_protocol_json()).with_identity()
    payload = _item("p1")
    payload["criteria"][0]["scope_support"] = "INCIDENTAL"
    item = WebPaperAssessment.model_validate(payload)
    result = _public_result(
        item, protocol,
        ScreeningPaper("p1", "Contextual relationship", "The relationship appears as background context."),
        stage="gemini_web_primary", elapsed=0,
    )

    assert _scope_support_errors(item, protocol)
    assert result["validation_status"] == "unresolved"
    assert _critic_route(result, protocol) == "validation_failure"


def test_repeated_incidental_primary_and_critic_resolve_to_safe_maybe(tmp_path):
    protocol = ReviewProtocol.model_validate_json(_protocol_json()).with_identity()

    class IncidentalKeepBrowser(FakeGeminiBrowser):
        def submit_prompt_and_get_response(self, prompt):
            self.prompts.append(prompt)
            if "Compile an immutable review protocol" in prompt:
                return protocol.model_dump_json()
            ids = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            items = []
            for paper_id in ids:
                payload = _item(paper_id)
                payload["criteria"][0]["scope_support"] = "INCIDENTAL"
                items.append(payload)
            return json.dumps({"items": items})

    frame = pd.DataFrame({
        "Title": ["Contextual relationship"],
        "Abstract": ["The relationship is mentioned only as background motivation."],
    })
    progress = ScreeningProgress()
    session = ScreeningSession()
    assert progress.start_job("incidental-safe-maybe")
    progress.begin_screening("incidental-safe-maybe", 1, GEMINI_WEB_VERSION)
    output = tmp_path / "runs" / "screened.csv"

    summary = screen_csv_with_gemini_web(
        frame=frame, valid=frame, title_col="Title", abstract_col="Abstract",
        research_question=protocol.research_question, research_context="",
        inclusion_criteria="", exclusion_criteria="", output_path=str(output),
        job_id="incidental-safe-maybe", input_fingerprint="incidental-safe-maybe",
        resume=False, limit=0, progress=progress, screening_session=session,
        browser_factory=IncidentalKeepBrowser,
    )

    saved = pd.read_csv(output).iloc[0]
    assert summary["maybe"] == 1
    assert saved["Decision"] == "MAYBE"
    assert saved["Verification_Status"] == "failed"
    assert bool(saved["Escalated"])


def test_affirmative_substantive_mismatch_remains_clear_reject():
    protocol = ReviewProtocol.model_validate_json(_protocol_json()).with_identity()
    result = _public_result(
        WebPaperAssessment.model_validate(_reject_item("p1")), protocol,
        ScreeningPaper("p1", "Different relationship", "The study evaluates a different relationship."),
        stage="gemini_web_primary", elapsed=0,
    )

    assert result["decision"] == "REJECT"
    assert result["validation_status"] == "validated"
    assert result["criteria"][0]["scope_support"] == "SUBSTANTIVE"


def test_repeated_unsupported_primary_and_critic_resolve_to_safe_maybe(tmp_path):
    protocol = _grounding_protocol()

    class UnsupportedAcronymBrowser(FakeGeminiBrowser):
        def submit_prompt_and_get_response(self, prompt):
            self.prompts.append(prompt)
            if "Compile an immutable review protocol" in prompt:
                return protocol.model_dump_json()
            ids = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            return json.dumps({"items": [_grounding_keep(paper_id).model_dump() for paper_id in ids]})

    frame = pd.DataFrame({
        "Title": ["AI-Powered Digital Twins"],
        "Abstract": ["DLT-based digital twins optimize operational efficiency."],
    })
    progress = ScreeningProgress()
    session = ScreeningSession()
    assert progress.start_job("acronym-safe-maybe")
    progress.begin_screening("acronym-safe-maybe", 1, GEMINI_WEB_VERSION)
    output = tmp_path / "runs" / "screened.csv"

    summary = screen_csv_with_gemini_web(
        frame=frame, valid=frame, title_col="Title", abstract_col="Abstract",
        research_question=protocol.research_question, research_context="",
        inclusion_criteria="", exclusion_criteria="", output_path=str(output),
        job_id="acronym-safe-maybe", input_fingerprint="acronym-safe-maybe", resume=False,
        limit=0, progress=progress, screening_session=session,
        browser_factory=UnsupportedAcronymBrowser,
    )

    saved = pd.read_csv(output).iloc[0]
    assert summary["maybe"] == 1
    assert saved["Decision"] == "MAYBE"
    assert saved["Validation_Status"] == "validated"
    assert bool(saved["Escalated"])


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


def test_inclusion_only_reject_routes_but_explicit_exclusion_reject_does_not():
    protocol = ReviewProtocol.model_validate({
        **json.loads(_protocol_json()),
        "criteria": [
            {
                "id": "inc1", "kind": "inclusion", "description": "Required relationship",
                "required": True, "expected_evidence": "Direct evidence", "source": "research_question",
            },
            {
                "id": "ex1", "kind": "exclusion", "description": "Explicit disqualifier",
                "required": True, "expected_evidence": "Direct evidence", "source": "user",
            },
        ],
    }).with_identity()
    base = {
        "decision": "REJECT", "decision_risk": "LOW", "validation_status": "validated",
        "contradictions": [],
    }

    inclusion_only = {
        **base,
        "criteria": [
            {"criterion_id": "inc1", "verdict": "NOT_MET"},
            {"criterion_id": "ex1", "verdict": "NOT_MET"},
        ],
    }
    explicit_exclusion = {
        **base,
        "criteria": [
            {"criterion_id": "inc1", "verdict": "MET"},
            {"criterion_id": "ex1", "verdict": "MET"},
        ],
    }

    assert _critic_route(inclusion_only, protocol) == "inclusion_only_reject"
    assert _critic_route(explicit_exclusion, protocol) == ""


@pytest.mark.parametrize(
    ("critic_decision", "expected_decision", "expected_status"),
    [("REJECT", "REJECT", "agreed"), ("KEEP", "MAYBE", "disagreed")],
)
def test_inclusion_only_reject_requires_independent_agreement(
    tmp_path, critic_decision, expected_decision, expected_status,
):
    class InclusionRejectBrowser(FakeGeminiBrowser):
        def submit_prompt_and_get_response(self, prompt):
            self.prompts.append(prompt)
            if "Compile an immutable review protocol" in prompt:
                return _protocol_json()
            ids = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            critic = "adversarial systematic-review critic" in prompt
            if critic and critic_decision == "KEEP":
                items = [_item(paper_id) for paper_id in ids]
            else:
                items = [_reject_item(paper_id) for paper_id in ids]
            return json.dumps({"items": items})

    frame = pd.DataFrame({"Title": ["Different relationship"], "Abstract": ["Direct evidence."]})
    progress = ScreeningProgress()
    session = ScreeningSession()
    job_id = f"verified-reject-{critic_decision.lower()}"
    assert progress.start_job(job_id)
    progress.begin_screening(job_id, 1, GEMINI_WEB_VERSION)
    output = tmp_path / critic_decision.lower() / "runs" / "screened.csv"

    summary = screen_csv_with_gemini_web(
        frame=frame, valid=frame, title_col="Title", abstract_col="Abstract",
        research_question="Invented RQ?", research_context="Context only",
        inclusion_criteria="", exclusion_criteria="", output_path=str(output),
        job_id=job_id, input_fingerprint=job_id, resume=False, limit=0,
        progress=progress, screening_session=session, browser_factory=InclusionRejectBrowser,
    )

    saved = pd.read_csv(output).iloc[0]
    assert saved["Decision"] == expected_decision
    assert saved["Critic_Route"] == "inclusion_only_reject"
    assert saved["Verification_Status"] == expected_status
    assert bool(saved["Escalated"])
    assert summary["verified_reject_count"] == int(expected_decision == "REJECT")


def test_critic_transport_failure_is_safe_maybe_and_requeued_on_resume(tmp_path):
    class CriticTimeoutBrowser(FakeGeminiBrowser):
        def submit_prompt_and_get_response(self, prompt):
            self.prompts.append(prompt)
            if "Compile an immutable review protocol" in prompt:
                return _protocol_json()
            if "adversarial systematic-review critic" in prompt:
                raise TimeoutError("critic transport unavailable")
            ids = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            return json.dumps({"items": [_reject_item(paper_id) for paper_id in ids]})

    class HealthyRejectBrowser(FakeGeminiBrowser):
        def submit_prompt_and_get_response(self, prompt):
            self.prompts.append(prompt)
            if "Compile an immutable review protocol" in prompt:
                return _protocol_json()
            ids = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            return json.dumps({"items": [_reject_item(paper_id) for paper_id in ids]})

    frame = pd.DataFrame({"Title": ["Different relationship"], "Abstract": ["Direct evidence."]})
    output = tmp_path / "runs" / "screened.csv"
    first_progress = ScreeningProgress()
    first_session = ScreeningSession()
    assert first_progress.start_job("critic-timeout")
    first_progress.begin_screening("critic-timeout", 1, GEMINI_WEB_VERSION)
    first = screen_csv_with_gemini_web(
        frame=frame, valid=frame, title_col="Title", abstract_col="Abstract",
        research_question="Invented RQ?", research_context="Context only",
        inclusion_criteria="", exclusion_criteria="", output_path=str(output),
        job_id="critic-timeout", input_fingerprint="critic-timeout", resume=False, limit=0,
        progress=first_progress, screening_session=first_session,
        browser_factory=CriticTimeoutBrowser,
    )

    failed = pd.read_csv(output).iloc[0]
    assert first["maybe"] == 1
    assert failed["Decision"] == "MAYBE"
    assert failed["Failure_Class"] == "transport_timeout"
    assert failed["Verification_Status"] == "failed"

    resumed_progress = ScreeningProgress()
    resumed_session = ScreeningSession()
    assert resumed_progress.start_job("critic-recovered")
    resumed_progress.begin_screening("critic-recovered", 1, GEMINI_WEB_VERSION)
    recovered = screen_csv_with_gemini_web(
        frame=frame, valid=frame, title_col="Title", abstract_col="Abstract",
        research_question="Invented RQ?", research_context="Context only",
        inclusion_criteria="", exclusion_criteria="", output_path=str(output),
        job_id="critic-recovered", input_fingerprint="critic-timeout", resume=True, limit=0,
        progress=resumed_progress, screening_session=resumed_session,
        browser_factory=HealthyRejectBrowser,
    )

    assert recovered["resumed_count"] == 0
    assert pd.read_csv(output).iloc[0]["Decision"] == "REJECT"


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
    assert "missing information" in prompt
    assert "adjacent" in prompt


def test_invalid_critic_degrades_risky_primary_to_safe_maybe(tmp_path):
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
    assert summary["maybe"] == 1
    assert saved["Decision"] == "MAYBE"
    assert saved["Validation_Status"] == "validated"
    assert bool(saved["Escalated"])
    assert saved["Verification_Status"] == "failed"
    assert trace[-1]["decision"] == "MAYBE"


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


def test_malformed_critic_degrades_risky_primary_to_safe_maybe(tmp_path):
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
    assert summary["maybe"] == 1
    assert saved["Decision"] == "MAYBE"
    assert saved["Verification_Status"] == "failed"


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
