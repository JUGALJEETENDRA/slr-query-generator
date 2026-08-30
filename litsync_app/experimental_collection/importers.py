from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from litsync_app.deduplication import MAPPED_KEYS, detect_source_and_map


SUPPORTED_SUFFIXES = {".csv", ".xls", ".xlsx", ".ris", ".nbib", ".txt"}


def _ris_records(text: str) -> list[dict[str, str]]:
    """Parse the conservative RIS/NBIB subset emitted by scholarly databases."""
    aliases = {
        "TI": "Title", "T1": "Title", "AB": "Abstract", "N2": "Abstract",
        "AU": "Authors", "A1": "Authors", "FAU": "Authors",
        "PY": "Year", "Y1": "Year", "DP": "Year",
        "JO": "Source title", "JF": "Source title", "JT": "Source title", "T2": "Source title",
        "DO": "DOI", "LID": "DOI", "AID": "DOI", "UR": "Link",
    }
    rows: list[dict[str, str]] = []
    current: dict[str, list[str]] = {}
    last_key = ""
    for raw in text.splitlines():
        match = re.match(r"^([A-Z0-9]{2,4})\s*-\s*(.*)$", raw.rstrip())
        if not match:
            continuation = raw.strip()
            if continuation and last_key and current.get(last_key):
                current[last_key][-1] += " " + continuation
            continue
        tag, value = match.groups()
        if tag == "PMID" and current:
            rows.append({
                key: "; ".join(values) if key == "Authors" else " ".join(values)
                for key, values in current.items()
            })
            current = {}
            last_key = ""
        if tag == "ER":
            if current:
                rows.append({
                    key: "; ".join(values) if key == "Authors" else " ".join(values)
                    for key, values in current.items()
                })
            current = {}
            last_key = ""
            continue
        mapped = aliases.get(tag)
        if mapped and value.strip():
            cleaned = value.strip()
            if mapped == "DOI":
                cleaned = re.sub(r"\s*\[(?:doi|pii)\]\s*$", "", cleaned, flags=re.I)
                if "10." not in cleaned and tag in {"LID", "AID"}:
                    continue
            current.setdefault(mapped, []).append(cleaned)
            last_key = mapped
        else:
            last_key = ""
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
    if suffix in {".ris", ".nbib", ".txt"}:
        rows = _ris_records(path.read_text(encoding="utf-8-sig", errors="replace"))
        if not rows:
            raise ValueError("No tagged RIS/PubMed records were found in the text export")
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


def detect_export_format(path: Path, frame: pd.DataFrame) -> str:
    suffix = path.suffix.lower()
    if suffix in {".ris", ".nbib", ".txt"}:
        return "pubmed_text" if suffix in {".nbib", ".txt"} else "ris"
    headers = {str(column).strip().lower() for column in frame.columns}
    if "document title" in headers or "article citation count" in headers:
        return "ieee_csv"
    if "eid" in headers or {"title", "source title"}.issubset(headers):
        return "scopus_csv"
    if "ut (unique wos id)" in headers or "article title" in headers:
        return "web_of_science"
    if "pmid" in headers or "journal/book" in headers:
        return "pubmed_csv"
    if {"title", "authors"}.issubset(headers):
        return "generic_bibliographic"
    return "unknown"


def source_warnings(source: str, detected: str, frame: pd.DataFrame) -> list[str]:
    expected = {
        "pubmed": {"pubmed_text", "pubmed_csv"},
        "scopus": {"scopus_csv", "ris", "generic_bibliographic"},
        "web_of_science": {"web_of_science", "ris", "generic_bibliographic"},
        "ieee_xplore": {"ieee_csv", "ris", "generic_bibliographic"},
        "google_scholar": {"ris", "generic_bibliographic"},
    }
    warnings = []
    if detected not in expected[source]:
        warnings.append(f"File structure looks like {detected}, not a typical {source} export")
    columns = {str(column).strip().lower() for column in frame.columns}
    if "abstract" not in columns or frame.get("Abstract", pd.Series(dtype=str)).fillna("").astype(str).str.strip().eq("").all():
        warnings.append("Abstracts are missing; screening may produce more MAYBE decisions")
    return warnings
