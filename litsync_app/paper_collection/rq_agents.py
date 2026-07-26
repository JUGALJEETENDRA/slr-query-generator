from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from pydantic import Field

from litsync_app.query.generator import SemanticScholarGrounder
from litsync_app.screening.local.engine import OllamaStructuredEngine
from litsync_app.screening.local.hardware import resolve_runtime_profile
from litsync_app.screening.local.three_layer import DEEP_MODEL, TRIAGE_MODEL

from .models import RQCandidate, StrictModel


class ProposedRQ(StrictModel):
    question: str = Field(min_length=12, max_length=600)
    rationale: str = Field(min_length=1, max_length=1200)


class RQProposalBatch(StrictModel):
    candidates: list[ProposedRQ] = Field(min_length=5, max_length=5)


class CritiquedRQ(StrictModel):
    candidate_index: int = Field(ge=1, le=5)
    specificity: int = Field(ge=0, le=5)
    answerability: int = Field(ge=0, le=5)
    searchability: int = Field(ge=0, le=5)
    cross_database_suitability: int = Field(ge=0, le=5)
    valid: bool
    criticism: str = Field(max_length=1600)


class RQCritiqueBatch(StrictModel):
    evaluations: list[CritiquedRQ] = Field(min_length=5, max_length=5)


PROPOSER_PROMPT = """
You are the research-question proposer in an autonomous systematic-review workflow.
Given a broad topic, produce exactly five distinct, precise, answerable research questions.
Every question must retain the user's topic, express a researchable relationship or evidence
objective, and be suitable for searching Google Scholar, Scopus, Web of Science, IEEE Xplore,
and PubMed. Do not invent a population, comparison, date range, language restriction, or outcome
unless the topic states it. Return only schema-valid JSON.
""".strip()

CRITIC_PROMPT = """
You are an independent systematic-review question critic. Score each supplied question from 0 to
5 for specificity, answerability from published evidence, Boolean searchability, and suitability
across five broad scholarly databases. Mark invalid questions that are incoherent, unsupported by
the topic, compound multiple unrelated reviews, or cannot be answered from literature. Preserve
the candidate_index and return exactly one evaluation for each index 1 through 5. Return only
schema-valid JSON.
""".strip()


class RQAgentService:
    def __init__(self, engine=None, model: str | None = None, grounder=None):
        profile = resolve_runtime_profile()
        self.engine = engine or OllamaStructuredEngine(profile)
        installed = profile.hardware.installed_models
        default_model = next(
            (candidate for candidate in (DEEP_MODEL, TRIAGE_MODEL) if candidate in installed),
            DEEP_MODEL,
        )
        self.model = model or os.getenv("AGENTIC_RQ_MODEL", "").strip() or default_model
        self.grounder = grounder or SemanticScholarGrounder()

    def generate_and_select(self, topic: str) -> tuple[list[RQCandidate], RQCandidate]:
        proposal = self.engine.generate(
            self.model,
            f"{PROPOSER_PROMPT}\n\nTOPIC:\n{topic.strip()}",
            RQProposalBatch,
        )
        proposed = RQProposalBatch.model_validate(proposal.value)
        numbered = "\n".join(
            f"{index + 1}. {item.question}" for index, item in enumerate(proposed.candidates)
        )
        critique = self.engine.generate(
            self.model,
            f"{CRITIC_PROMPT}\n\nORIGINAL TOPIC:\n{topic.strip()}\n\nCANDIDATES:\n{numbered}",
            RQCritiqueBatch,
        )
        evaluated = RQCritiqueBatch.model_validate(critique.value)
        by_index = {item.candidate_index: item for item in evaluated.evaluations}
        if set(by_index) != {1, 2, 3, 4, 5}:
            raise ValueError("RQ critic did not evaluate each candidate exactly once")
        scored_in_order = [by_index[index] for index in range(1, 6)]
        with ThreadPoolExecutor(max_workers=5) as pool:
            evidence = list(pool.map(self._evidence_count, [
                item.question for item in proposed.candidates
            ]))
        candidates = []
        for proposed_item, scored, evidence_count in zip(
            proposed.candidates, scored_in_order, evidence
        ):
            candidates.append(RQCandidate(
                question=proposed_item.question,
                rationale=proposed_item.rationale,
                specificity=scored.specificity,
                answerability=scored.answerability,
                searchability=scored.searchability,
                cross_database_suitability=scored.cross_database_suitability,
                evidence_availability=min(5, evidence_count),
                evidence_record_count=evidence_count,
                valid=scored.valid,
                criticism=scored.criticism,
            ))
        valid = [(index, item) for index, item in enumerate(candidates) if item.valid]
        if not valid:
            raise ValueError("RQ agents did not produce a valid research question")
        _, selected = max(valid, key=lambda pair: (pair[1].total_score, -pair[0]))
        return candidates, selected

    def _evidence_count(self, question: str) -> int:
        try:
            return len(self.grounder.search(question))
        except Exception:
            return 0
