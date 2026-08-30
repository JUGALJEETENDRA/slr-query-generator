from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


CANONICAL_VERSION = "litsync-canonical-json-v1"


def text_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def source_row_id(value: Any) -> str:
    return text_value(value)


def canonical_json_bytes(value: Any) -> bytes:
    value = canonical_content(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_content(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): canonical_content(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_content(item) for item in value]
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n")
    return value


def canonical_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataframe_payload(frame: pd.DataFrame) -> dict[str, Any]:
    columns = [str(column) for column in frame.columns]
    return {
        "canonical_version": CANONICAL_VERSION,
        "columns": columns,
        "rows": [
            [text_value(value) for value in row]
            for row in frame.itertuples(index=False, name=None)
        ],
    }


def source_dataset_fingerprint(path: str | Path) -> str:
    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    return source_dataframe_fingerprint(frame)


def source_dataframe_fingerprint(frame: pd.DataFrame) -> str:
    return canonical_fingerprint({
        "fingerprint_kind": "source_dataset",
        **dataframe_payload(frame),
    })


def screening_output_fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    materialized = [dict(row) for row in rows]
    columns = sorted({
        str(column)
        for row in materialized
        for column in row
    })
    return canonical_fingerprint({
        "fingerprint_kind": "screening_output",
        "canonical_version": CANONICAL_VERSION,
        "columns": columns,
        "rows": [
            [text_value(row.get(column)) for column in columns]
            for row in materialized
        ],
    })


def finite_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default
