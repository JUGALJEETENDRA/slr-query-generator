import json
import re

import pandas as pd
import pytest

from litsync_app.screening.bulk import ScreeningProgress, ScreeningSession
from litsync_app.integrations.gemini_web_v24_automation import GeminiWebV24Config
from litsync_app.integrations.gemini_web_v24_prompt import V24Paper, build_primary_prompt, build_verification_prompt
from litsync_app.integrations.gemini_web_v24_screening import (
    GEMINI_WEB_V24_PROTOCOL_VERSION,
    GEMINI_WEB_V24_VERSION,
    V24_STRUCTURED_OUTPUT_FAILURE,
    V24_TRANSPORT_FAILURE,
    V24Assessment,
    V24Diagnostics,
    V24Protocol,
    _execute_batch_with_degraded_retry,
    _resume_rows,
    _validate_and_decide,
    _verification_route,
    screen_csv_with_gemini_web_v24,
)
from litsync_app.screening.engines import GEMINI_WEB_V24_ENGINE, normalize_processing_engine


def _protocol_payload():
    return {
        "protocol_version": GEMINI_WEB_V24_PROTOCOL_VERSION,
        "protocol_id": "",
        "research_question": "How does a requested approach address a requested task?",
        "objective": "Identify studies that substantively apply the requested approach to the requested task.",
        "population_or_subject": ["The subject specified by the researcher"],
        "methods_or_interventions": ["The approach specified by the researcher"],
        "target_tasks_or_outcomes": ["The task specified by the researcher"],
        "application_context": [],
        "required_inclusion_criteria": [{
            "id": "required_relationship",
            "kind": "inclusion",
            "description": "The paper substantively studies the requested relationship.",
            "required": True,
            "expected_evidence": "Title or abstract evidence of the objective, method, analysis, or findings.",
            "source": "research_question",
        }],
        "exclusion_boundaries": [],
        "ambiguities": ["An incidental mention does not establish the relationship."],
        "synonyms_and_equivalent_concepts": [],
        "near_neighbor_but_out_of_scope_concepts": [],
    }


def _protocol():
    return V24Protocol.model_validate(_protocol_payload()).with_identity()


def _assessment(
    paper_id="0",
    *,
    verdict="MET",
    support="SUBSTANTIVE",
    relationship="SUPPORTS",
    decision="KEEP",
    risk="LOW",
    evidence=True,
):
    return V24Assessment.model_validate({
        "paper_id": paper_id,
        "decision": decision,
        "confidence": 0.93,
        "decision_risk": risk,
        "reason": "The supplied title directly establishes the requested relationship.",
        "criterion_assessments": [{
            "criterion_id": "required_relationship",
            "verdict": verdict,
            "scope_support": support,
            "evidence_relationship": relationship,
            "rationale": "The title supplies the decisive relationship evidence.",
            "evidence": (
                [{"source": "title", "evidence_id": "title_001"}] if evidence else []
            ),
        }],
    })


def test_domain_neutral_deterministic_keep_reject_and_maybe():
    protocol = _protocol()
    paper = V24Paper("0", "A substantive requested relationship study", "The method is evaluated.")

    keep = _validate_and_decide(_assessment(), protocol, paper)
    assert keep["decision"] == "KEEP"
    assert keep["validation_status"] == "validated"

    reject = _validate_and_decide(_assessment(
        verdict="NOT_MET", relationship="CONFLICTS", decision="REJECT",
    ), protocol, paper)
    assert reject["decision"] == "REJECT"
    assert reject["validation_status"] == "validated"

    unclear = _validate_and_decide(_assessment(
        verdict="UNCLEAR", support="INSUFFICIENT", relationship="INSUFFICIENT",
        decision="MAYBE", risk="BORDERLINE", evidence=False,
    ), protocol, paper)
    assert unclear["decision"] == "MAYBE"
    assert unclear["validation_status"] == "validated"


def test_valid_insufficient_maybe_does_not_trigger_verification():
    protocol = _protocol()
    result = _validate_and_decide(
        _assessment(
            verdict="UNCLEAR",
            support="INSUFFICIENT",
            relationship="INSUFFICIENT",
            decision="MAYBE",
            risk="LOW",
            evidence=False,
        ),
        protocol,
        V24Paper(
            "0",
            "Broad related study",
            "The abstract does not establish the required relationship.",
        ),
    )

    assert result["decision"] == "MAYBE"
    assert result["decision_risk"] == "BORDERLINE"
    assert result["validation_status"] == "validated"
    assert _verification_route(result, protocol) == ""


def test_high_risk_substantive_unresolved_maybe_is_verified():
    protocol = _protocol()
    result = _validate_and_decide(
        _assessment(
            verdict="UNCLEAR",
            support="SUBSTANTIVE",
            relationship="INSUFFICIENT",
            decision="MAYBE",
            risk="HIGH",
            evidence=True,
        ),
        protocol,
        V24Paper(
            "0",
            "Substantive related study",
            "The paper evaluates part of the requested relationship.",
        ),
    )

    assert result["decision"] == "MAYBE"
    assert result["decision_risk"] == "HIGH"
    assert result["validation_status"] == "validated"
    assert _verification_route(result, protocol) == "borderline_primary"


def test_unmentioned_requirement_cannot_be_invented_as_not_met():
    result = _validate_and_decide(
        _assessment(
            verdict="NOT_MET", relationship="CONFLICTS", decision="REJECT", evidence=False,
        ),
        _protocol(),
        V24Paper("0", "A broad study", "The required relationship is not discussed."),
    )
    assert result["decision"] == "MAYBE"
    assert result["validation_status"] == "unresolved"
    assert any("affirmative conflicting evidence" in error for error in result["validation_errors"])


def test_incidental_or_hallucinated_evidence_cannot_support_keep():
    protocol = _protocol()
    paper = V24Paper("0", "A broad study", "The requested topic appears only as background.")
    incidental = _validate_and_decide(_assessment(
        support="INCIDENTAL", relationship="INCIDENTAL", decision="KEEP",
    ), protocol, paper)
    assert incidental["decision"] == "MAYBE"
    assert incidental["validation_status"] == "unresolved"

    payload = _assessment().model_dump(mode="json")
    payload["criterion_assessments"][0]["evidence"][0]["evidence_id"] = "abstract_999"
    hallucinated = _validate_and_decide(V24Assessment.model_validate(payload), protocol, paper)
    assert hallucinated["decision"] == "MAYBE"
    assert any("invalid evidence reference" in error for error in hallucinated["validation_errors"])


def test_inferred_exclusion_uncertainty_does_not_become_hidden_rejection():
    payload = _protocol_payload()
    payload["exclusion_boundaries"] = [{
        "id": "inferred_boundary",
        "kind": "exclusion",
        "description": "A logically incompatible focus.",
        "required": True,
        "expected_evidence": "Evidence of the incompatible focus.",
        "source": "research_question",
    }]
    protocol = V24Protocol.model_validate(payload).with_identity()
    assessment = _assessment().model_dump(mode="json")
    assessment["criterion_assessments"].append({
        "criterion_id": "inferred_boundary",
        "verdict": "UNCLEAR",
        "scope_support": "INSUFFICIENT",
        "evidence_relationship": "INSUFFICIENT",
        "rationale": "The inferred boundary is not indicated.",
        "evidence": [],
    })
    result = _validate_and_decide(
        V24Assessment.model_validate(assessment),
        protocol,
        V24Paper("0", "A substantive study", "The requested relationship is evaluated."),
    )
    assert result["decision"] == "KEEP"
    assert result["validation_status"] == "validated"
    assert _verification_route(result, protocol) == ""

    payload["exclusion_boundaries"][0]["source"] = "user"
    user_protocol = V24Protocol.model_validate(payload).with_identity()
    user_result = _validate_and_decide(
        V24Assessment.model_validate(assessment),
        user_protocol,
        V24Paper("0", "A substantive study", "The requested relationship is evaluated."),
    )
    assert user_result["decision"] == "MAYBE"


def test_unknown_or_missing_criterion_id_is_unresolved():
    payload = _assessment().model_dump(mode="json")
    payload["criterion_assessments"][0]["criterion_id"] = "invented_criterion"
    result = _validate_and_decide(
        V24Assessment.model_validate(payload),
        _protocol(),
        V24Paper("0", "A study", "An abstract."),
    )
    assert result["decision"] == "MAYBE"
    assert result["validation_status"] == "unresolved"


@pytest.mark.parametrize(
    ("title", "abstract"),
    [
        (
            "Clinical decision support evaluation",
            "The requested relationship is the objective and is evaluated on clinical records.",
        ),
        (
            "Automated defect analysis",
            "The requested relationship is implemented and evaluated in a software workflow.",
        ),
        (
            "Adaptive learning intervention",
            "The requested relationship is tested with learners in a controlled study.",
        ),
        (
            "Operational planning system",
            "The requested relationship is evaluated in an organizational process.",
        ),
    ],
)
def test_policy_is_identical_across_unrelated_domains(title, abstract):
    result = _validate_and_decide(
        _assessment(),
        _protocol(),
        V24Paper("0", title, abstract),
    )
    assert result["decision"] == "KEEP"
    assert result["validation_status"] == "validated"


def test_verifier_is_prediction_blind():
    prompt = build_verification_prompt(
        protocol=_protocol().model_dump(mode="json"),
        papers=[V24Paper("0", "Distinct title", "Distinct abstract")],
        flags={"0": {
            "validation_errors": ["criterion tension"],
            "unresolved_criterion_ids": ["required_relationship"],
        }},
        schema={"type": "object"},
    )
    assert '"decision": "KEEP"' not in prompt
    assert "DISTINCTIVE_PRIMARY_RATIONALE" not in prompt
    assert "criterion tension" in prompt


def test_prompt_defines_exclusion_polarity_without_domain_rules():
    prompt = build_primary_prompt(
        protocol=_protocol().model_dump(mode="json"),
        papers=[V24Paper("0", "Title", "Abstract")],
        schema={"type": "object"},
    )
    assert "exclusion MET means the" in prompt
    assert "disqualifying condition itself is present" in prompt
    assert "Never mark an exclusion MET because the paper avoids" in prompt


class FakeV24Browser:
    instances = []

    def __init__(self, config):
        self.prompts = []
        self.new_chats = 0
        self.recoveries = []
        FakeV24Browser.instances.append(self)

    def __enter__(self):
        self.start_new_job_chat()
        return self

    def __exit__(self, *args):
        return None

    def set_attempt_context(self, **kwargs):
        return None

    def start_new_job_chat(self):
        self.new_chats += 1

    def note_recovery(self, action):
        self.recoveries.append(action)

    def recover_transport_failure(self, **kwargs):
        self.recoveries.append("transport")

    def submit_prompt_and_get_response(self, prompt):
        self.prompts.append(prompt)
        if "Compile an immutable systematic-review screening protocol" in prompt:
            return json.dumps(_protocol_payload())
        identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
        return json.dumps({
            "items": [
                _assessment(paper_id=paper_id).model_dump(mode="json")
                for paper_id in identifiers
            ]
        })


def _run(
    tmp_path,
    *,
    job_id,
    output_name,
    browser_factory=FakeV24Browser,
    frame=None,
    resume=False,
    input_fingerprint="same-input",
):
    if frame is None:
        frame = pd.DataFrame({
            "Title": ["A substantive requested relationship study"],
            "Abstract": ["The requested approach is evaluated for the requested task."],
            "Existing": ["preserved"],
        })
    progress = ScreeningProgress()
    session = ScreeningSession()
    assert progress.start_job(job_id)
    progress.begin_screening(job_id, 1, GEMINI_WEB_V24_VERSION)
    output = tmp_path / "runs" / output_name
    summary = screen_csv_with_gemini_web_v24(
        frame=frame,
        valid=frame,
        title_col="Title",
        abstract_col="Abstract",
        research_question="How does a requested approach address a requested task?",
        research_context="",
        inclusion_criteria="",
        exclusion_criteria="",
        output_path=str(output),
        job_id=job_id,
        input_fingerprint=input_fingerprint,
        resume=resume,
        limit=0,
        progress=progress,
        screening_session=session,
        browser_factory=browser_factory,
    )
    return summary, pd.read_csv(output)


def test_end_to_end_one_paper_then_validated_cache(tmp_path):
    FakeV24Browser.instances.clear()
    first, saved = _run(tmp_path, job_id="v24-first", output_name="first.csv")
    assert first["architecture_version"] == GEMINI_WEB_V24_VERSION
    assert first["keep"] == 1
    assert first["verification_count"] == 0
    assert saved.loc[0, "Existing"] == "preserved"
    assert saved.loc[0, "Route_Used"] == "primary"
    assert saved.loc[0, "Validation_Status"] == "validated"
    assert saved.loc[0, "Prompt_Version"] == GEMINI_WEB_V24_VERSION
    assert len(FakeV24Browser.instances) == 1
    second, cached = _run(tmp_path, job_id="v24-cache", output_name="cached.csv")
    assert second["cache_hit_count"] == 1
    assert cached.loc[0, "Route_Used"] == "validated_cache"
    assert bool(cached.loc[0, "Cache_Hit"])
    assert len(FakeV24Browser.instances) == 1


def _five_paper_frame():
    return pd.DataFrame({
        "Title": [f"Substantive relationship study {index}" for index in range(5)],
        "Abstract": [
            "The requested approach is evaluated for the requested task."
            for _ in range(5)
        ],
    })


def _valid_batch_response(identifiers):
    return json.dumps({
        "items": [
            _assessment(paper_id=paper_id).model_dump(mode="json")
            for paper_id in identifiers
        ]
    })


def _diagnostic_events(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_timed_out_degraded_subgroup_replays_exactly_once_and_recovers(tmp_path):
    class ReplaySuccessBrowser:
        def __init__(self):
            self.batch_sizes = []
            self.paper_ids = []
            self.prompts = []
            self.events = []

        def set_attempt_context(self, **kwargs):
            return None

        def submit_prompt_and_get_response(self, prompt):
            identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            self.batch_sizes.append(len(identifiers))
            self.paper_ids.append(identifiers)
            self.prompts.append(prompt)
            self.events.append(("call", identifiers))
            call = len(self.batch_sizes)
            if call <= 2:
                return '{"items": []}'
            if call == 3:
                raise TimeoutError("response timed out")
            return _valid_batch_response(identifiers)

        def start_new_job_chat(self):
            self.events.append(("chat", []))

        def note_recovery(self, action):
            self.events.append(("note", action))

        def recover_transport_failure(self, **kwargs):
            self.events.append(("transport", kwargs))

    browser = ReplaySuccessBrowser()
    diagnostics_path = tmp_path / "replay-success-diagnostics.jsonl"
    diagnostics = V24Diagnostics(diagnostics_path)
    assessed, failures = _execute_batch_with_degraded_retry(
        browser,
        _protocol(),
        [
            V24Paper(str(index), f"Title {index}", f"Abstract {index}")
            for index in range(5)
        ],
        verification=False,
        flags=None,
        diagnostics=diagnostics,
    )

    assert browser.batch_sizes == [5, 5, 2, 2, 3]
    assert browser.paper_ids[2] == browser.paper_ids[3] == ["0", "1"]
    assert browser.prompts[2] == browser.prompts[3]
    first_two_calls = [
        index for index, event in enumerate(browser.events)
        if event == ("call", ["0", "1"])
    ]
    transport_recovery = next(
        index for index, event in enumerate(browser.events)
        if event[0] == "transport"
    )
    assert first_two_calls[0] < transport_recovery < first_two_calls[1]
    assert set(assessed) == {"0", "1", "2", "3", "4"}
    assert failures == {}
    assert diagnostics.degraded_subgroup_replay_count == 1
    assert diagnostics.degraded_subgroup_replay_success_count == 1
    assert diagnostics.degraded_subgroup_replay_exhaustion_count == 0
    assert diagnostics.papers_recovered_through_replay == 2

    subgroup_events = [
        event for event in _diagnostic_events(diagnostics_path)
        if event["event"] == "gemini_web_degraded_subgroup"
    ]
    assert [event["outcome"] for event in subgroup_events] == [
        "transport_failure",
        "transport_recovery",
        "transport_replay_succeeded",
    ]
    assert all(event["paper_ids"] == ["0", "1"] for event in subgroup_events)
    assert all(event["paper_count"] == 2 for event in subgroup_events)


def test_timed_out_degraded_subgroup_exhausts_without_recursive_split(tmp_path):
    class ReplayExhaustionBrowser:
        def __init__(self):
            self.batch_sizes = []

        def set_attempt_context(self, **kwargs):
            return None

        def submit_prompt_and_get_response(self, prompt):
            identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            self.batch_sizes.append(len(identifiers))
            if len(self.batch_sizes) <= 2:
                return '{"items": []}'
            if identifiers == ["0", "1"]:
                raise TimeoutError("response timed out")
            return _valid_batch_response(identifiers)

        def start_new_job_chat(self):
            return None

        def note_recovery(self, action):
            return None

        def recover_transport_failure(self, **kwargs):
            return None

    browser = ReplayExhaustionBrowser()
    diagnostics_path = tmp_path / "replay-exhaustion-diagnostics.jsonl"
    diagnostics = V24Diagnostics(diagnostics_path)
    assessed, failures = _execute_batch_with_degraded_retry(
        browser,
        _protocol(),
        [
            V24Paper(str(index), f"Title {index}", f"Abstract {index}")
            for index in range(5)
        ],
        verification=False,
        flags=None,
        diagnostics=diagnostics,
    )

    assert browser.batch_sizes == [5, 5, 2, 2, 3]
    assert set(assessed) == {"2", "3", "4"}
    assert set(failures) == {"0", "1"}
    assert {failure_class for _, failure_class in failures.values()} == {
        V24_TRANSPORT_FAILURE
    }
    assert diagnostics.fallback_count == 1
    assert diagnostics.degraded_subgroup_replay_count == 1
    assert diagnostics.degraded_subgroup_replay_success_count == 0
    assert diagnostics.degraded_subgroup_replay_exhaustion_count == 1
    assert diagnostics.papers_recovered_through_replay == 0
    outcomes = [
        event["outcome"] for event in _diagnostic_events(diagnostics_path)
        if event["event"] == "gemini_web_degraded_subgroup"
    ]
    assert outcomes == [
        "transport_failure",
        "transport_recovery",
        "transport_replay_exhausted",
    ]


def test_structured_degraded_subgroup_failure_is_terminal_without_replay(tmp_path):
    class StructuredTerminalBrowser:
        def __init__(self):
            self.batch_sizes = []

        def set_attempt_context(self, **kwargs):
            return None

        def submit_prompt_and_get_response(self, prompt):
            identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            self.batch_sizes.append(len(identifiers))
            if len(self.batch_sizes) <= 3:
                return '{"items": []}'
            return _valid_batch_response(identifiers)

        def start_new_job_chat(self):
            return None

        def note_recovery(self, action):
            return None

        def recover_transport_failure(self, **kwargs):
            return None

    browser = StructuredTerminalBrowser()
    diagnostics_path = tmp_path / "structured-terminal-diagnostics.jsonl"
    diagnostics = V24Diagnostics(diagnostics_path)
    assessed, failures = _execute_batch_with_degraded_retry(
        browser,
        _protocol(),
        [
            V24Paper(str(index), f"Title {index}", f"Abstract {index}")
            for index in range(5)
        ],
        verification=False,
        flags=None,
        diagnostics=diagnostics,
    )

    assert browser.batch_sizes == [5, 5, 2, 3]
    assert set(assessed) == {"2", "3", "4"}
    assert set(failures) == {"0", "1"}
    assert {failure_class for _, failure_class in failures.values()} == {
        V24_STRUCTURED_OUTPUT_FAILURE
    }
    assert diagnostics.degraded_subgroup_replay_count == 0
    subgroup_events = [
        event for event in _diagnostic_events(diagnostics_path)
        if event["event"] == "gemini_web_degraded_subgroup"
    ]
    assert len(subgroup_events) == 1
    assert subgroup_events[0]["outcome"] == "structured_output_terminal"
    assert subgroup_events[0]["paper_ids"] == ["0", "1"]


def test_successful_degraded_subgroup_replay_is_cached_end_to_end(tmp_path):
    class ReplayCachingBrowser(FakeV24Browser):
        def __init__(self, config):
            super().__init__(config)
            self.batch_sizes = []

        def submit_prompt_and_get_response(self, prompt):
            self.prompts.append(prompt)
            if "Compile an immutable systematic-review screening protocol" in prompt:
                return json.dumps(_protocol_payload())
            identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            self.batch_sizes.append(len(identifiers))
            call = len(self.batch_sizes)
            if call <= 2:
                return '{"items": []}'
            if call == 3:
                raise TimeoutError("response timed out")
            return _valid_batch_response(identifiers)

    FakeV24Browser.instances.clear()
    first, saved = _run(
        tmp_path,
        job_id="v24-subgroup-replay-cache",
        output_name="subgroup-replay-cache.csv",
        browser_factory=ReplayCachingBrowser,
        frame=_five_paper_frame(),
        input_fingerprint="subgroup-replay-cache-input",
    )

    first_browser = FakeV24Browser.instances[0]
    assert first_browser.batch_sizes == [5, 5, 2, 2, 3]
    assert first["keep"] == 5
    assert first["technical_fallback_count"] == 0
    assert first["timeout_fallback_count"] == 0
    assert first["degraded_subgroup_replay_count"] == 1
    assert first["degraded_subgroup_replay_success_count"] == 1
    assert first["degraded_subgroup_replay_exhaustion_count"] == 0
    assert first["papers_recovered_through_replay"] == 2
    assert set(saved["Route_Used"]) == {"primary"}

    second, cached = _run(
        tmp_path,
        job_id="v24-subgroup-replay-cache-second",
        output_name="subgroup-replay-cache-second.csv",
        browser_factory=ReplayCachingBrowser,
        frame=_five_paper_frame(),
        input_fingerprint="subgroup-replay-cache-input",
    )
    assert second["cache_hit_count"] == 5
    assert set(cached["Route_Used"]) == {"validated_cache"}
    assert len(FakeV24Browser.instances) == 1


def test_exhausted_subgroup_replay_is_resume_eligible_without_reprocessing_sibling(
    tmp_path,
):
    class ReplayResumeBrowser(FakeV24Browser):
        recover_failed = False

        def __init__(self, config):
            super().__init__(config)
            self.batch_sizes = []
            self.paper_ids = []

        def submit_prompt_and_get_response(self, prompt):
            self.prompts.append(prompt)
            if "Compile an immutable systematic-review screening protocol" in prompt:
                return json.dumps(_protocol_payload())
            identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            self.batch_sizes.append(len(identifiers))
            self.paper_ids.append(identifiers)
            if self.recover_failed:
                return _valid_batch_response(identifiers)
            if len(self.batch_sizes) <= 2:
                return '{"items": []}'
            if identifiers == ["0", "1"]:
                raise TimeoutError("response timed out")
            return _valid_batch_response(identifiers)

    FakeV24Browser.instances.clear()
    ReplayResumeBrowser.recover_failed = False
    first, saved = _run(
        tmp_path,
        job_id="v24-subgroup-replay-resume",
        output_name="subgroup-replay-resume.csv",
        browser_factory=ReplayResumeBrowser,
        frame=_five_paper_frame(),
        input_fingerprint="subgroup-replay-resume-input",
    )

    first_browser = FakeV24Browser.instances[0]
    assert first_browser.batch_sizes == [5, 5, 2, 2, 3]
    assert first["keep"] == 3
    assert first["maybe"] == 2
    assert first["timeout_fallback_count"] == 2
    assert first["degraded_subgroup_replay_exhaustion_count"] == 1
    failed = saved[saved["Decision"] == "MAYBE"]
    assert set(failed["Source_Row_Index"]) == {0, 1}
    assert set(failed["Failure_Class"]) == {V24_TRANSPORT_FAILURE}

    ReplayResumeBrowser.recover_failed = True
    second, resumed = _run(
        tmp_path,
        job_id="v24-subgroup-replay-resume-second",
        output_name="subgroup-replay-resume-second.csv",
        browser_factory=ReplayResumeBrowser,
        frame=_five_paper_frame(),
        resume=True,
        input_fingerprint="subgroup-replay-resume-input",
    )

    second_browser = FakeV24Browser.instances[1]
    assert second_browser.batch_sizes == [2]
    assert second_browser.paper_ids == [["0", "1"]]
    assert second["resumed_count"] == 3
    assert second["keep"] == 5
    assert second["technical_fallback_count"] == 0
    assert set(resumed["Decision"]) == {"KEEP"}


def test_invalid_five_paper_batch_recovers_as_two_plus_three_and_caches(tmp_path):
    class RecoveringStructuredBrowser(FakeV24Browser):
        def __init__(self, config):
            super().__init__(config)
            self.batch_sizes = []

        def submit_prompt_and_get_response(self, prompt):
            self.prompts.append(prompt)
            if "Compile an immutable systematic-review screening protocol" in prompt:
                return json.dumps(_protocol_payload())
            identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            self.batch_sizes.append(len(identifiers))
            if len(self.batch_sizes) <= 2:
                return '{"items": []}'
            return _valid_batch_response(identifiers)

    FakeV24Browser.instances.clear()
    first, saved = _run(
        tmp_path,
        job_id="v24-degraded-structured",
        output_name="degraded-structured.csv",
        browser_factory=RecoveringStructuredBrowser,
        frame=_five_paper_frame(),
        input_fingerprint="degraded-structured-input",
    )

    browser = FakeV24Browser.instances[0]
    assert browser.batch_sizes == [5, 5, 2, 3]
    assert first["keep"] == 5
    assert first["technical_fallback_count"] == 0
    assert first["structured_output_fallback_count"] == 0
    assert set(saved["Route_Used"]) == {"primary"}
    assert set(saved["Failure_Class"].fillna("")) == {""}
    assert "v24_primary_degraded_retry_structured_clean_chat" in browser.recoveries

    second, cached = _run(
        tmp_path,
        job_id="v24-degraded-structured-cache",
        output_name="degraded-structured-cache.csv",
        browser_factory=RecoveringStructuredBrowser,
        frame=_five_paper_frame(),
        input_fingerprint="degraded-structured-input",
    )
    assert second["cache_hit_count"] == 5
    assert set(cached["Route_Used"]) == {"validated_cache"}
    assert len(FakeV24Browser.instances) == 1


def test_timed_out_five_paper_batch_recovers_as_two_plus_three(tmp_path):
    class RecoveringTimeoutBrowser:
        def __init__(self):
            self.batch_sizes = []
            self.recoveries = []

        def set_attempt_context(self, **kwargs):
            return None

        def submit_prompt_and_get_response(self, prompt):
            identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            self.batch_sizes.append(len(identifiers))
            if len(self.batch_sizes) <= 2:
                raise TimeoutError("response timed out")
            return _valid_batch_response(identifiers)

        def start_new_job_chat(self):
            self.recoveries.append("chat")

        def note_recovery(self, action):
            self.recoveries.append(action)

        def recover_transport_failure(self, **kwargs):
            self.recoveries.append("transport")

    browser = RecoveringTimeoutBrowser()
    diagnostics = V24Diagnostics(tmp_path / "timeout-diagnostics.jsonl")
    assessed, failures = _execute_batch_with_degraded_retry(
        browser,
        _protocol(),
        [
            V24Paper(str(index), f"Title {index}", f"Abstract {index}")
            for index in range(5)
        ],
        verification=False,
        flags=None,
        diagnostics=diagnostics,
    )

    assert set(assessed) == {"0", "1", "2", "3", "4"}
    assert failures == {}
    assert browser.batch_sizes == [5, 5, 2, 3]
    assert "v24_primary_degraded_retry_transport_recovery" in browser.recoveries
    assert diagnostics.fallback_count == 0


def test_failed_two_paper_subgroup_is_isolated_and_reprocessed_on_resume(tmp_path):
    class PartialStructuredBrowser(FakeV24Browser):
        recover_two = False

        def __init__(self, config):
            super().__init__(config)
            self.batch_sizes = []

        def submit_prompt_and_get_response(self, prompt):
            self.prompts.append(prompt)
            if "Compile an immutable systematic-review screening protocol" in prompt:
                return json.dumps(_protocol_payload())
            identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            size = len(identifiers)
            self.batch_sizes.append(size)
            if size == 5 or (size == 2 and not self.recover_two):
                return '{"items": []}'
            return _valid_batch_response(identifiers)

    FakeV24Browser.instances.clear()
    PartialStructuredBrowser.recover_two = False
    first, saved = _run(
        tmp_path,
        job_id="v24-partial-degraded",
        output_name="partial-degraded.csv",
        browser_factory=PartialStructuredBrowser,
        frame=_five_paper_frame(),
        input_fingerprint="partial-degraded-input",
    )

    first_browser = FakeV24Browser.instances[0]
    assert first_browser.batch_sizes == [5, 5, 2, 3]
    assert first["keep"] == 3
    assert first["maybe"] == 2
    assert first["technical_fallback_count"] == 2
    assert first["structured_output_fallback_count"] == 2
    failed = saved[saved["Decision"] == "MAYBE"]
    assert set(failed["Route_Used"]) == {"technical_failure"}
    assert set(failed["Failure_Class"]) == {V24_STRUCTURED_OUTPUT_FAILURE}

    PartialStructuredBrowser.recover_two = True
    second, resumed = _run(
        tmp_path,
        job_id="v24-partial-degraded-resume",
        output_name="partial-degraded-resume.csv",
        browser_factory=PartialStructuredBrowser,
        frame=_five_paper_frame(),
        resume=True,
        input_fingerprint="partial-degraded-input",
    )

    second_browser = FakeV24Browser.instances[1]
    assert second_browser.batch_sizes == [2]
    assert second["resumed_count"] == 3
    assert second["keep"] == 5
    assert second["technical_fallback_count"] == 0
    assert set(resumed["Decision"]) == {"KEEP"}


def test_end_to_end_valid_insufficient_maybe_skips_verifier(tmp_path):
    class InsufficientMaybeBrowser(FakeV24Browser):
        def submit_prompt_and_get_response(self, prompt):
            self.prompts.append(prompt)
            if "Compile an immutable systematic-review screening protocol" in prompt:
                return json.dumps(_protocol_payload())
            identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            return json.dumps({
                "items": [
                    _assessment(
                        paper_id=paper_id,
                        verdict="UNCLEAR",
                        support="INSUFFICIENT",
                        relationship="INSUFFICIENT",
                        decision="MAYBE",
                        risk="BORDERLINE",
                        evidence=False,
                    ).model_dump(mode="json")
                    for paper_id in identifiers
                ]
            })

    InsufficientMaybeBrowser.instances.clear()
    summary, saved = _run(
        tmp_path,
        job_id="v24-insufficient-maybe",
        output_name="insufficient-maybe.csv",
        browser_factory=InsufficientMaybeBrowser,
    )

    browser = InsufficientMaybeBrowser.instances[0]
    assert summary["maybe"] == 1
    assert summary["verification_count"] == 0
    assert len(browser.prompts) == 2
    assert not any("independent prediction-blind verifier" in prompt for prompt in browser.prompts)
    assert "v24_primary_to_verification_clean_chat" not in browser.recoveries
    assert saved.loc[0, "Decision"] == "MAYBE"
    assert saved.loc[0, "Decision_Risk"] == "BORDERLINE"
    assert saved.loc[0, "Route_Used"] == "primary"
    assert saved.loc[0, "Verification_Status"] == "not_required"


def test_missing_abstract_is_safe_maybe_without_paper_assessment_call(tmp_path):
    FakeV24Browser.instances.clear()
    frame = pd.DataFrame({"Title": ["Title-only record"], "Abstract": [""]})
    progress = ScreeningProgress()
    session = ScreeningSession()
    assert progress.start_job("v24-missing-abstract")
    progress.begin_screening("v24-missing-abstract", 1, GEMINI_WEB_V24_VERSION)
    output = tmp_path / "runs" / "missing.csv"
    summary = screen_csv_with_gemini_web_v24(
        frame=frame,
        valid=frame,
        title_col="Title",
        abstract_col="Abstract",
        research_question="How does a requested approach address a requested task?",
        research_context="",
        inclusion_criteria="",
        exclusion_criteria="",
        output_path=str(output),
        job_id="v24-missing-abstract",
        input_fingerprint="missing-abstract",
        resume=False,
        limit=0,
        progress=progress,
        screening_session=session,
        browser_factory=FakeV24Browser,
    )
    saved = pd.read_csv(output)
    assert summary["maybe"] == 1
    assert saved.loc[0, "Route_Used"] == "missing_abstract"
    assert saved.loc[0, "Verification_Status"] == "not_required"
    assert saved.loc[0, "Validation_Status"] == "validated"
    assert len(FakeV24Browser.instances[0].prompts) == 1


def test_keep_reject_verification_conflict_resolves_to_maybe(tmp_path):
    class ConflictBrowser(FakeV24Browser):
        def submit_prompt_and_get_response(self, prompt):
            self.prompts.append(prompt)
            if "Compile an immutable systematic-review screening protocol" in prompt:
                return json.dumps(_protocol_payload())
            identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            verification = "independent prediction-blind verifier" in prompt
            items = []
            for paper_id in identifiers:
                if verification:
                    item = _assessment(
                        paper_id=paper_id,
                        verdict="NOT_MET",
                        relationship="CONFLICTS",
                        decision="REJECT",
                    )
                else:
                    item = _assessment(paper_id=paper_id, risk="BORDERLINE")
                items.append(item.model_dump(mode="json"))
            return json.dumps({"items": items})

    summary, saved = _run(
        tmp_path,
        job_id="v24-conflict",
        output_name="conflict.csv",
        browser_factory=ConflictBrowser,
    )
    assert summary["maybe"] == 1
    assert saved.loc[0, "Decision"] == "MAYBE"
    assert saved.loc[0, "Verification_Status"] == "disagreed"
    assert saved.loc[0, "Route_Used"] == "risky_definitive"
    assert not bool(saved.loc[0, "Cache_Hit"])


@pytest.mark.parametrize(
    "failure_class",
    [V24_TRANSPORT_FAILURE, V24_STRUCTURED_OUTPUT_FAILURE],
)
def test_technical_fallback_checkpoint_is_requeued(tmp_path, failure_class):
    checkpoint = tmp_path / "checkpoint.csv"
    pd.DataFrame([{
        "Source_Row_Index": 0,
        "Protocol_ID": "protocol",
        "Prompt_Version": GEMINI_WEB_V24_VERSION,
        "Decision": "MAYBE",
        "Validation_Status": "unresolved",
        "Verification_Status": "failed",
        "Failure_Class": failure_class,
        "Criteria_JSON": "[]",
        "Evidence_JSON": "[]",
    }]).to_csv(checkpoint, index=False)
    assert _resume_rows(checkpoint, "protocol", {"0"}) == {}


def test_malformed_single_paper_batch_gets_one_repair_then_safe_failure(tmp_path):
    class MalformedBrowser:
        def __init__(self):
            self.calls = 0
            self.new_chats = 0

        def set_attempt_context(self, **kwargs):
            return None

        def submit_prompt_and_get_response(self, prompt):
            self.calls += 1
            return '{"items": []}'

        def start_new_job_chat(self):
            self.new_chats += 1

        def recover_transport_failure(self, **kwargs):
            return None

    browser = MalformedBrowser()
    diagnostics = V24Diagnostics(tmp_path / "diagnostics.jsonl")
    assessed, failures = _execute_batch_with_degraded_retry(
        browser,
        _protocol(),
        [V24Paper("0", "Title", "Abstract")],
        verification=False,
        flags=None,
        diagnostics=diagnostics,
    )
    assert assessed == {}
    reason, failure_class = failures["0"]
    assert "failed after one retry" in reason
    assert failure_class == V24_STRUCTURED_OUTPUT_FAILURE
    assert browser.calls == 2
    assert browser.new_chats == 1
    assert diagnostics.retry_count == 2
    assert diagnostics.fallback_count == 1


def test_exhausted_timeout_is_classified_as_transport_failure(tmp_path):
    class TimeoutBrowser:
        def __init__(self):
            self.calls = 0
            self.recoveries = 0

        def set_attempt_context(self, **kwargs):
            return None

        def submit_prompt_and_get_response(self, prompt):
            self.calls += 1
            raise TimeoutError("response timed out")

        def start_new_job_chat(self):
            return None

        def recover_transport_failure(self, **kwargs):
            self.recoveries += 1

    browser = TimeoutBrowser()
    diagnostics = V24Diagnostics(tmp_path / "timeout-failure-diagnostics.jsonl")
    assessed, failures = _execute_batch_with_degraded_retry(
        browser,
        _protocol(),
        [V24Paper("0", "Title", "Abstract")],
        verification=False,
        flags=None,
        diagnostics=diagnostics,
    )

    assert assessed == {}
    assert failures["0"][1] == V24_TRANSPORT_FAILURE
    assert browser.calls == 2
    assert browser.recoveries == 2
    assert diagnostics.retry_count == 2
    assert diagnostics.fallback_count == 1


def test_v24_lifecycle_defaults_are_bounded_and_independent(monkeypatch):
    monkeypatch.delenv("GEMINI_WEB_V24_MAX_CHAT_SUBMISSIONS", raising=False)
    monkeypatch.delenv("GEMINI_WEB_V24_MAX_BROWSER_SUBMISSIONS", raising=False)
    config = GeminiWebV24Config()
    assert config.max_chat_submissions == 5
    assert config.max_browser_submissions == 10
    transport = config.transport_config()
    assert transport.max_chat_submissions == 5
    assert transport.max_browser_submissions == 10


def test_v24_engine_is_explicit_and_obsolete_engine_falls_back_to_local():
    assert normalize_processing_engine("gemini_web_v24") == GEMINI_WEB_V24_ENGINE
    assert normalize_processing_engine("gemini-web-v2.4") == GEMINI_WEB_V24_ENGINE
    assert normalize_processing_engine("gemini_web") == "local"
