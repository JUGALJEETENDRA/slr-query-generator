from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from threading import RLock

from .models import AgenticRun, utc_now


class AgenticRunStore:
    """SQLite-backed whole-run checkpoints with atomic replacement."""

    def __init__(self, path: str | Path = "private/agentic_runs.sqlite3"):
        self.path = Path(path)
        self._lock = RLock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        if not self._initialized:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agentic_runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._initialized = True
        return connection

    def save(self, run: AgenticRun) -> AgenticRun:
        run.updated_at = utc_now()
        payload = run.model_dump_json()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agentic_runs(run_id, status, payload, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (run.run_id, run.status, payload, run.updated_at),
            )
        return run

    def get(self, run_id: str) -> AgenticRun | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM agentic_runs WHERE run_id = ?", (str(run_id),)
            ).fetchone()
        return AgenticRun.model_validate_json(row[0]) if row else None

    def recoverable(self) -> list[AgenticRun]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM agentic_runs
                WHERE status IN ('queued', 'running')
                ORDER BY updated_at
                """
            ).fetchall()
        return [AgenticRun.model_validate_json(row[0]) for row in rows]

    def has_active(self) -> bool:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) FROM agentic_runs
                WHERE status IN ('queued', 'running', 'needs_attention')
                """
            ).fetchone()
        return bool(row and row[0])

    def public_payload(self, run_id: str) -> dict:
        run = self.get(run_id)
        if run is None:
            raise KeyError(run_id)
        payload = json.loads(run.model_dump_json())
        for source in payload.get("sources", {}).values():
            source["record_count"] = len(source.get("records") or [])
            source.pop("records", None)
        secrets = [
            os.getenv(name, "").strip()
            for name in (
                "SKYVERN_API_KEY",
                "SKYVERN_CREDENTIAL_GOOGLE_SCHOLAR",
                "SKYVERN_CREDENTIAL_SCOPUS",
                "SKYVERN_CREDENTIAL_WEB_OF_SCIENCE",
                "SKYVERN_CREDENTIAL_IEEE_XPLORE",
                "SKYVERN_CREDENTIAL_PUBMED",
            )
            if os.getenv(name, "").strip()
        ]

        def redact(value):
            if isinstance(value, dict):
                return {key: redact(item) for key, item in value.items()}
            if isinstance(value, list):
                return [redact(item) for item in value]
            if isinstance(value, str):
                for secret in secrets:
                    value = value.replace(secret, "[REDACTED]")
            return value

        return redact(payload)
