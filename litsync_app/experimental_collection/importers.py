from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from litsync_app.deduplication import MAPPED_KEYS, detect_source_and_map


SUPPORTED_SUFFIXES = {".csv", ".xls", ".xlsx", ".ris", ".nbib"}


def _ris_records(text: str) -> list[dict[str, str]]:
    """Parse the conservative RIS/NBIB subset emitted by scholarly databases."""
    aliases = {
        "TI": "Title", "T1": "Title", "AB": "Abstract", "N2": "Abstract",
        "AU": "Authors", "A1": "Authors", "FAU": "Authors",
        "PY": "Year", "Y1": "Year", "DP": "Year",
        "JO": "Source title", "JF": "Source title", "JT": "Source title", "T2": "Source title",
        "DO": "DOI", "UR": "Link",
    }
    rows: list[dict[str, str]] = []
    current: dict[str, list[str]] = {}
    for raw in text.splitlines():
        match = re.match(r"^([A-Z0-9]{2,4})\s*-\s*(.*)$", raw.rstrip())
        if not match:
            continue
        tag, value = match.groups()
        if tag == "ER":
            if current:
                rows.append({
                    key: "; ".join(values) if key == "Authors" else " ".join(values)
                    for key, values in current.items()
                })
            current = {}
            continue
        mapped = aliases.get(tag)
        if mapped and value.strip():
            current.setdefault(mapped, []).append(value.strip())
    if current:
        rows.append({
            key: "; ".join(values) if key == "Authors" else " ".join(values)
            for key, values in current.items()
        })
    return rows


def read_export(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("CSV encoding is not supported")
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path, sheet_name=0)
    if suffix in {".ris", ".nbib"}:
        rows = _ris_records(path.read_text(encoding="utf-8-sig", errors="replace"))
        return pd.DataFrame(rows, columns=MAPPED_KEYS)
    raise ValueError("Upload CSV, Excel, RIS, or NBIB")


def normalize_export(path: Path, source: str) -> pd.DataFrame:
    frame = read_export(path)
    if frame.empty:
        raise ValueError("The export contains no records")
    mapped = detect_source_and_map(frame, path.name)
    mapped = mapped[mapped["Title"].fillna("").astype(str).str.strip().ne("")].copy()
    if mapped.empty:
        raise ValueError("No paper titles could be read from the export")
    mapped["Collection Source"] = source
    return mapped
