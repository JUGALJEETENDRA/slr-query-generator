from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any


def cache_key(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JsonDiskCache:
    def __init__(self, directory: str = "outputs/cache/local_ai"):
        self.directory = Path(directory)
        self._lock = Lock()

    def get(self, namespace: str, key: str) -> dict | None:
        path = self.directory / namespace / f"{key}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def set(self, namespace: str, key: str, value: dict) -> None:
        path = self.directory / namespace / f"{key}.json"
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, path)
