from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import Field

from .models import CollectedPaper, DATABASES, SourceCollection, StrictModel, utc_now


SOURCE_CONFIG = {
    "google_scholar": {
        "name": "Google Scholar", "url": "https://scholar.google.com/",
        "credential_env": "SKYVERN_CREDENTIAL_GOOGLE_SCHOLAR",
        "native_export": False,
    },
    "scopus": {
        "name": "Scopus", "url": "https://www.scopus.com/search/form.uri",
        "credential_env": "SKYVERN_CREDENTIAL_SCOPUS",
        "native_export": True,
    },
    "web_of_science": {
        "name": "Web of Science", "url": "https://www.webofscience.com/wos/woscc/basic-search",
        "credential_env": "SKYVERN_CREDENTIAL_WEB_OF_SCIENCE",
        "native_export": True,
    },
    "ieee_xplore": {
        "name": "IEEE Xplore", "url": "https://ieeexplore.ieee.org/search/advanced",
        "credential_env": "SKYVERN_CREDENTIAL_IEEE_XPLORE",
        "native_export": True,
    },
    "pubmed": {
        "name": "PubMed", "url": "https://pubmed.ncbi.nlm.nih.gov/",
        "credential_env": "SKYVERN_CREDENTIAL_PUBMED",
        "native_export": True,
    },
}
BLOCKER_TERMS = (
    "captcha", "mfa", "2fa", "two-factor", "login", "sign in", "subscription",
    "access denied", "institutional access", "human verification",
)


class ExtractedPaper(StrictModel):
    title: str = Field(min_length=1, max_length=1500)
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str = ""
    abstract: str = ""
    doi: str = ""
    url: str = ""
    cited_by: int | None = None
    source_rank: int = Field(ge=1, le=30)


class ExtractedPaperBatch(StrictModel):
    papers: list[ExtractedPaper] = Field(default_factory=list, max_length=30)
    attempted_candidates: int = Field(default=0, ge=0, le=30)


class CollectionNeedsAttention(RuntimeError):
    def __init__(self, message: str, *, live_url: str = "", run_id: str = ""):
        super().__init__(message)
        self.live_url = live_url
        self.run_id = run_id


def _value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return _value(json.loads(value))
        except json.JSONDecodeError:
            return value
    if isinstance(value, dict):
        if "papers" not in value:
            for name in ("data", "output", "extracted_information", "result"):
                if value.get(name) is not None:
                    return _value(value[name])
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    for name in ("data", "output", "extracted_information", "result"):
        candidate = getattr(value, name, None)
        if candidate:
            return _value(candidate)
    return value


class SkyvernCloudClient:
    """Small lazy SDK adapter so tests and non-agentic mode do not require Skyvern."""

    def __init__(self, api_key: str | None = None):
        self.api_key = (api_key or os.getenv("SKYVERN_API_KEY", "")).strip()
        self._blocked_sessions: dict[str, tuple[Any, Any, dict[str, str]]] = {}

    async def collect(
        self, database: str, query: str, artifact_dir: Path
    ) -> tuple[ExtractedPaperBatch, dict[str, str]]:
        if not self.api_key:
            raise CollectionNeedsAttention("SKYVERN_API_KEY is not configured")
        try:
            from skyvern import Skyvern
        except ImportError as exc:
            raise RuntimeError(
                "Skyvern SDK is not installed. Install the pinned project requirements."
            ) from exc
        config = SOURCE_CONFIG[database]
        browser = None
        page = None
        metadata = {"skyvern_run_id": "", "live_url": ""}
        session_key = f"{artifact_dir.resolve()}|{database}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        download_tasks: list[asyncio.Task] = []
        try:
            prior = self._blocked_sessions.pop(session_key, None)
            if prior:
                browser, page, metadata = prior
            else:
                client = Skyvern(api_key=self.api_key)
                browser = await client.launch_cloud_browser()
                metadata["skyvern_run_id"] = str(
                    getattr(browser, "browser_session_id", "")
                    or getattr(browser, "id", "")
                )
                metadata["live_url"] = str(
                    getattr(browser, "live_url", "")
                    or getattr(browser, "livestream_url", "")
                )
                page = await browser.get_working_page()
                await page.goto(config["url"])
                credential_id = os.getenv(config["credential_env"], "").strip()
                if credential_id:
                    await page.agent.login(
                        credential_type="skyvern", credential_id=credential_id
                    )
            if hasattr(page, "on"):
                def capture_download(download):
                    async def save_download():
                        suggested = getattr(download, "suggested_filename", "database-export")
                        if callable(suggested):
                            suggested = suggested()
                        filename = re.sub(
                            r"[^A-Za-z0-9._-]+", "-", str(suggested or "database-export")
                        )[:180]
                        await download.save_as(str(artifact_dir / filename))

                    download_tasks.append(asyncio.create_task(save_download()))

                page.on("download", capture_download)
            await page.agent.run_task(self._navigation_prompt(config, query))
            extracted = await page.extract(
                self._extraction_prompt(config["name"]),
                ExtractedPaperBatch.model_json_schema(),
            )
            value = _value(extracted)
            if inspect.isawaitable(value):
                value = await value
            batch = ExtractedPaperBatch.model_validate(value)
            if download_tasks:
                await asyncio.gather(*download_tasks, return_exceptions=True)
            (artifact_dir / "extraction.json").write_text(
                json.dumps(batch.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            try:
                screenshot = await page.screenshot(full_page=True)
                if isinstance(screenshot, bytes):
                    (artifact_dir / "final-page.png").write_bytes(screenshot)
            except Exception:
                pass
            return batch, metadata
        except Exception as exc:
            message = str(exc)
            if any(term in message.lower() for term in BLOCKER_TERMS):
                if browser is not None and page is not None:
                    self._blocked_sessions[session_key] = (browser, page, metadata)
                    browser = None
                raise CollectionNeedsAttention(
                    message, live_url=metadata["live_url"],
                    run_id=metadata["skyvern_run_id"],
                ) from exc
            raise
        finally:
            if browser is not None:
                try:
                    await browser.close()
                except Exception:
                    pass

    @staticmethod
    def _navigation_prompt(config: dict[str, Any], query: str) -> str:
        export_instruction = (
            "If the site has a native export for selected records, select these 10 records and "
            "download a CSV, Excel, or RIS export for the private audit trail. "
            if config.get("native_export") else ""
        )
        return (
            f"On {config['name']}, search using this exact database query: {query}\n"
            "Keep the database's default relevance ordering. Do not apply date, language, "
            "citation, or document-type filters. Inspect no more than the first 30 ranked "
            "results and stop after 10 genuine scholarly paper records are available. "
            f"{export_instruction}"
            "Do not bypass access controls. COMPLETE when the ranked results are ready; "
            "TERMINATE if login, MFA, CAPTCHA, subscription, or access restrictions require "
            "human intervention."
        )

    @staticmethod
    def _extraction_prompt(name: str) -> str:
        return (
            f"Extract up to the first 10 usable scholarly paper records from the current {name} "
            "results in displayed relevance order. Open record details when needed for abstracts "
            "or DOI, but inspect no more than 30 candidates. Return title, authors, year, venue, "
            "abstract, DOI, canonical URL, citation count, source_rank, and attempted_candidates. "
            "Do not invent missing metadata."
        )


class DatabaseCollector:
    def __init__(self, client=None):
        self.client = client or SkyvernCloudClient()

    async def collect(self, database: str, query: str, private_root: Path) -> SourceCollection:
        if database not in DATABASES:
            raise ValueError(f"unsupported database: {database}")
        source = SourceCollection(
            database=database, status="running", attempts=0, started_at=utc_now()
        )
        artifact_dir = private_root / database
        for attempt in range(1, 3):
            source.attempts = attempt
            try:
                batch, metadata = await self.client.collect(database, query, artifact_dir)
                seen_ranks: set[int] = set()
                records: list[CollectedPaper] = []
                for item in sorted(batch.papers, key=lambda paper: paper.source_rank):
                    if item.source_rank in seen_ranks or len(records) >= 10:
                        continue
                    seen_ranks.add(item.source_rank)
                    records.append(CollectedPaper(
                        **item.model_dump(),
                        database=database,
                        query=query,
                        raw_artifact_ref=f"{database}/extraction.json",
                        retrieved_at=utc_now(),
                        provenance=[{
                            "database": database,
                            "source_rank": item.source_rank,
                            "query": query,
                            "retrieved_at": utc_now(),
                        }],
                    ))
                source.records = records
                source.attempted_candidates = min(30, batch.attempted_candidates)
                source.skyvern_run_id = metadata.get("skyvern_run_id", "")
                source.live_url = metadata.get("live_url", "")
                source.status = "completed" if len(records) == 10 else "partial"
                source.completed_at = utc_now()
                return source
            except CollectionNeedsAttention as exc:
                source.status = "needs_attention"
                source.blocker = str(exc)
                source.live_url = exc.live_url
                source.skyvern_run_id = exc.run_id
                return source
            except Exception as exc:
                source.error = str(exc)
                if attempt == 2:
                    source.status = "failed"
                    source.completed_at = utc_now()
                    return source
                await asyncio.sleep(0)
        return source
