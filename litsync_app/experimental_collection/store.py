from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import RLock

from .models import CollectionRun, utc_now


class CollectionStore:
    def __init__(self, path: str | Path = "private/experimental_collection/runs.sqlite3"):
        self.path = Path(path)
        self._lock = RLock()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS collection_runs (
            run_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL)"""
        )
        return connection

    def save(self, run: CollectionRun) -> CollectionRun:
        run.updated_at = utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO collection_runs(run_id,payload,updated_at) VALUES(?,?,?)
                ON CONFLICT(run_id) DO UPDATE SET payload=excluded.payload,
                updated_at=excluded.updated_at""",
                (run.run_id, run.model_dump_json(), run.updated_at),
            )
        return run

    def get(self, run_id: str) -> CollectionRun | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM collection_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return CollectionRun.model_validate_json(row[0]) if row else None
