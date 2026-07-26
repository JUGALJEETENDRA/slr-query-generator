from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


DATABASES = (
    "google_scholar",
    "scopus",
    "web_of_science",
    "ieee_xplore",
    "pubmed",
)
RunStatus = Literal[
    "queued", "running", "needs_attention", "completed",
    "completed_partial", "failed", "cancelled",
]
SourceStatus = Literal[
    "pending", "running", "completed", "partial",
    "needs_attention", "failed", "skipped",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RQCandidate(StrictModel):
    question: str = Field(min_length=12, max_length=600)
    rationale: str = Field(default="", max_length=1200)
    specificity: int = Field(default=0, ge=0, le=5)
    answerability: int = Field(default=0, ge=0, le=5)
    searchability: int = Field(default=0, ge=0, le=5)
    cross_database_suitability: int = Field(default=0, ge=0, le=5)
    evidence_availability: int = Field(default=0, ge=0, le=5)
    evidence_record_count: int = Field(default=0, ge=0)
    valid: bool = True
    criticism: str = Field(default="", max_length=1600)

    @property
    def total_score(self) -> int:
        return (
            self.specificity + self.answerability + self.searchability
            + self.cross_database_suitability + self.evidence_availability
        )


class CollectedPaper(StrictModel):
    title: str = Field(min_length=1, max_length=1500)
    authors: list[str] = Field(default_factory=list, max_length=200)
    year: int | None = Field(default=None, ge=1000, le=2200)
    venue: str = Field(default="", max_length=1000)
    abstract: str = Field(default="", max_length=30000)
    doi: str = Field(default="", max_length=1000)
    url: str = Field(default="", max_length=4000)
    cited_by: int | None = Field(default=None, ge=0)
    database: str
    source_rank: int = Field(ge=1, le=30)
    query: str
    retrieved_at: str = Field(default_factory=utc_now)
    raw_artifact_ref: str = ""
    provenance: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("database")
    @classmethod
    def known_database(cls, value: str) -> str:
        if value not in DATABASES:
            raise ValueError(f"unsupported database: {value}")
        return value


class SourceCollection(StrictModel):
    database: str
    status: SourceStatus = "pending"
    records: list[CollectedPaper] = Field(default_factory=list, max_length=10)
    attempted_candidates: int = Field(default=0, ge=0, le=30)
    attempts: int = Field(default=0, ge=0, le=2)
    skyvern_run_id: str = ""
    live_url: str = ""
    blocker: str = ""
    error: str = ""
    started_at: str | None = None
    completed_at: str | None = None

    @field_validator("database")
    @classmethod
    def known_database(cls, value: str) -> str:
        if value not in DATABASES:
            raise ValueError(f"unsupported database: {value}")
        return value


class AgenticRun(StrictModel):
    run_id: str
    topic: str = Field(min_length=3, max_length=1000)
    status: RunStatus = "queued"
    stage: str = "topic_intake"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    error: str = ""
    cancelled: bool = False
    rq_candidates: list[RQCandidate] = Field(default_factory=list, max_length=5)
    selected_rq: str = ""
    selection_rationale: str = ""
    queries: dict[str, str] = Field(default_factory=dict)
    sources: dict[str, SourceCollection] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    screening: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(cls, run_id: str, topic: str) -> "AgenticRun":
        return cls(
            run_id=run_id,
            topic=topic.strip(),
            sources={
                database: SourceCollection(database=database)
                for database in DATABASES
            },
        )

