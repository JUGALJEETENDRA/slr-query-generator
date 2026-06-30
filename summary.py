"""Generate machine-readable screening run metrics."""

from __future__ import annotations

import os
from typing import Dict

import pandas as pd


def write_summary(metrics: Dict, output_dir: str) -> str:
    path = os.path.join(output_dir or ".", "summary.csv")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(
        [{"Metric": key, "Value": value} for key, value in metrics.items()]
    ).to_csv(path, index=False)
    return path
