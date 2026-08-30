from __future__ import annotations

import asyncio
import json

import pandas as pd

from litsync_app.integrations import gemini_web_screening as fast
from litsync_app.integrations.gemini_web_screening_prompt import ARCHITECTURE_VERSION
from litsync_app.screening.bulk import ScreeningProgress, ScreeningSession


def _papers_from_prompt(prompt: str) -> list[dict[str, str]]:
    return json.loads(
        prompt.split("Papers:\n", 1)[1].split(
            "\n\nRequired JSON schema:",
            1,
        )[0]
    )


def _reject_assessment(paper: dict[str, str]) -> dict[str, object]:
    return {
        "paper_id": paper["paper_id"],
        "decision": "REJECT",
        "confidence": 0.95,
        "reason": "The required inclusion criterion is not met.",
        "evidence_quote": paper["title"],
        "inclusion_assessments": [
            {
                "criterion_id": "I1",
                "status": "NOT_MET",
            }
        ],
        "exclusion_assessments": [
            {
                "criterion_id": "E1",
                "status": "NOT_MET",
            }
        ],
        "risk_flags": [],
    }


class RecordingSchedulerBrowser:
    """Fake browser that makes one primary batch slow and records stage overlap."""

    def __init__(self) -> None:
        self.started = 0
        self.closed = 0
        self.active_pages = 0
        self.peak_active_pages = 0
        self.pages_opened = 0
        self.pages_closed = 0
        self.activity_callback = None
        self.events: list[tuple[str, tuple[str, ...]]] = []

    async def start(self) -> None:
        self.started += 1

    async def close(self) -> None:
        self.closed += 1

    async def submit_fresh(
        self,
        prompt: str,
        *,
        timeout_seconds: float | None = None,
    ) -> str:
        del timeout_seconds

        self.pages_opened += 1
        self.active_pages += 1
        self.peak_active_pages = max(
            self.peak_active_pages,
            self.active_pages,
        )
        if self.activity_callback:
            self.activity_callback(self.active_pages)

        stage = "protocol"
        paper_ids: tuple[str, ...] = ()

        try:
            if "Papers:" not in prompt:
                self.events.append(("protocol_start", ()))
                await asyncio.sleep(0.001)
                return json.dumps(
                    {
                        "review_summary": "Include relevant primary studies.",
                        "inclusion_criteria": [
                            {
                                "criterion_id": "I1",
                                "text": "Relevant",
                            }
                        ],
                        "exclusion_criteria": [
                            {
                                "criterion_id": "E1",
                                "text": "Exclude reviews",
                            }
                        ],
                        "interpretation_notes": [],
                        "ambiguity_rules": [],
                        "evidence_requirements": [],
                    }
                )

            papers = _papers_from_prompt(prompt)
            paper_ids = tuple(
                str(paper["paper_id"])
                for paper in papers
            )
            stage = (
                "verification"
                if "prediction-blind" in prompt
                else "primary"
            )
            self.events.append((f"{stage}_start", paper_ids))

            # The first primary batch finishes quickly. The second remains
            # active long enough for a pipelined scheduler to start blind
            # verification in the newly freed concurrency slot.
            if stage == "primary" and paper_ids[0] == "0":
                await asyncio.sleep(0.01)
            elif stage == "primary":
                await asyncio.sleep(0.20)
            else:
                await asyncio.sleep(0.01)

            return json.dumps(
                {
                    "items": [
                        _reject_assessment(paper)
                        for paper in papers
                    ]
                }
            )
        finally:
            self.events.append((f"{stage}_end", paper_ids))
            self.active_pages = max(
                0,
                self.active_pages - 1,
            )
            self.pages_closed += 1
            if self.activity_callback:
                self.activity_callback(self.active_pages)


def _run_scheduler_case(
    tmp_path,
    monkeypatch,
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    RecordingSchedulerBrowser,
]:
    monkeypatch.setenv(
        "GEMINI_WEB_CONCURRENCY",
        "2",
    )
    monkeypatch.setenv(
        "GEMINI_WEB_BATCH_SIZE",
        "5",
    )
    monkeypatch.setenv(
        "GEMINI_WEB_VERIFICATION_BATCH_SIZE",
        "5",
    )
    monkeypatch.setenv(
        "GEMINI_WEB_JOB_TIMEOUT_SECONDS",
        "60",
    )

    frame = pd.DataFrame(
        {
            "Title": [
                f"Paper {index}"
                for index in range(10)
            ],
            "Abstract": [
                f"Abstract evidence {index}"
                for index in range(10)
            ],
            "Original": [
                f"value-{index}"
                for index in range(10)
            ],
        }
    )

    browser = RecordingSchedulerBrowser()
    progress = ScreeningProgress()
    session = ScreeningSession()
    job_id = "scheduler-test"
    output_path = tmp_path / "screened.csv"

    progress.start_job(job_id)
    progress.begin_screening(
        job_id,
        len(frame),
        ARCHITECTURE_VERSION,
    )
    session.begin(
        job_id,
        str(output_path),
        ARCHITECTURE_VERSION,
    )

    result = fast.screen_csv_with_gemini_web(
        frame=frame,
        valid=frame,
        title_col="Title",
        abstract_col="Abstract",
        research_question="Question",
        research_context="Context",
        inclusion_criteria="Relevant",
        exclusion_criteria="Exclude reviews",
        output_path=str(output_path),
        job_id=job_id,
        input_fingerprint="scheduler-input",
        source_dataset_fingerprint="scheduler-dataset",
        resume=False,
        limit=0,
        progress=progress,
        screening_session=session,
        browser_factory=lambda: browser,
    )

    output = pd.read_csv(
        output_path,
        keep_default_na=False,
    )
    return result, output, browser


def test_verification_starts_before_all_primary_batches_finish(
    tmp_path,
    monkeypatch,
):
    _, _, browser = _run_scheduler_case(
        tmp_path,
        monkeypatch,
    )

    first_verification_start = next(
        index
        for index, event in enumerate(browser.events)
        if event[0] == "verification_start"
    )
    final_primary_end = max(
        index
        for index, event in enumerate(browser.events)
        if event[0] == "primary_end"
    )

    assert first_verification_start < final_primary_end, (
        "Blind verification started only after every primary batch "
        "finished. The scheduler is still phase-separated instead of "
        "pipelined."
    )


def test_mixed_primary_and_verification_never_exceed_concurrency(
    tmp_path,
    monkeypatch,
):
    _, _, browser = _run_scheduler_case(
        tmp_path,
        monkeypatch,
    )

    assert browser.peak_active_pages == 2
    assert browser.pages_opened == browser.pages_closed
    assert browser.started == browser.closed == 1


def test_pipelined_scheduler_preserves_complete_validated_output(
    tmp_path,
    monkeypatch,
):
    result, output, _ = _run_scheduler_case(
        tmp_path,
        monkeypatch,
    )

    assert result["primary_batches_submitted"] == 2
    assert result["verification_batches_submitted"] == 2
    assert result["primary_papers_requested"] == 10
    assert result["verification_papers_requested"] == 10

    assert len(output) == 10
    assert output["Source_Row_Index"].nunique() == 10
    assert set(output["Decision"]) == {"REJECT"}
    assert set(output["Primary_Decision"]) == {"REJECT"}
    assert set(output["Verifier_Decision"]) == {"REJECT"}
    assert set(output["Agreement_Status"]) == {"agreement"}
    assert set(output["Validation_Status"]) == {"validated"}
    assert set(output["Failure_Class"]) == {""}
