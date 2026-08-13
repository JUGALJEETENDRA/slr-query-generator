from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SOURCES = ("google_scholar", "scopus", "web_of_science", "ieee_xplore", "pubmed")
AUTOMATED_SOURCES = ("pubmed",)
ASSISTED_SOURCES = ("google_scholar", "scopus", "web_of_science", "ieee_xplore")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceState(StrictModel):
    source: str
    query: str = Field(min_length=1, max_length=20000)
    mode: Literal["automated", "assisted"]
    status: Literal[
        "ready", "running", "needs_attention", "imported", "failed", "skipped"
    ] = "ready"
    records: int = Field(default=0, ge=0)
    attempts: int = Field(default=0, ge=0)
    message: str = ""
    search_url: str = ""
    raw_filename: str = ""
    raw_sha256: str = ""
    started_at: str | None = None
    completed_at: str | None = None


class CollectionRun(StrictModel):
    run_id: str
    research_question: str = Field(min_length=3, max_length=2000)
    limit: int = Field(default=100, ge=1, le=100)
    status: Literal["ready", "collecting", "needs_attention", "completed", "failed"] = "ready"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    sources: dict[str, SourceState]
    counts: dict[str, int] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str = ""

    @field_validator("sources")
    @classmethod
    def exact_sources(cls, value: dict[str, SourceState]) -> dict[str, SourceState]:
        unknown = set(value) - set(SOURCES)
        if unknown:
            raise ValueError("unsupported sources: " + ", ".join(sorted(unknown)))
        return value
