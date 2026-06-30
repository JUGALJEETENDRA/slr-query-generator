"""Write complete and category-specific screening CSV files."""

from __future__ import annotations

import os
from typing import Dict, Iterable

import pandas as pd


RESULT_COLUMNS = [
    "Title", "Abstract", "Decision", "Reason", "Required_Evidence",
    "Paper_Contribution", "Similarity", "Screening_Stage",
]
CATEGORY_COLUMNS = ["Title", "Abstract", "Reason", "Similarity", "Screening_Stage"]


def write_screening_outputs(records: Iterable[Dict], output_path: str) -> Dict[str, str]:
    records = list(records)
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "screened": output_path,
        "included": os.path.join(output_dir, "included_studies.csv"),
        "excluded": os.path.join(output_dir, "excluded_studies.csv"),
        "maybe": os.path.join(output_dir, "maybe_studies.csv"),
    }
    pd.DataFrame(records, columns=RESULT_COLUMNS).to_csv(paths["screened"], index=False)
    for decision, key in (("KEEP", "included"), ("REJECT", "excluded"), ("MAYBE", "maybe")):
        category = [
            {column: record.get(column, "") for column in CATEGORY_COLUMNS}
            for record in records
            if record.get("Decision") == decision
        ]
        pd.DataFrame(category, columns=CATEGORY_COLUMNS).to_csv(paths[key], index=False)
    return paths
