from __future__ import annotations

import ast
from pathlib import Path

import pytest

from litsync_app.query.generator import ALLOWED_TERM_SOURCES, _validate_term_details


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_FILES = tuple(sorted((ROOT / "litsync_app").rglob("*.py")))
BANNED_IMPORT_ROOTS = {
    "archive", "benchmark", "experiments", "model_lab", "query_framework",
    "gemini_web_screening", "gemini_web_prompt", "gemini_web_parser",
}


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_production_import_graph_excludes_historical_and_fixture_code():
    violations = {
        str(path.relative_to(ROOT)): sorted(_import_roots(path) & BANNED_IMPORT_ROOTS)
        for path in PRODUCTION_FILES
        if _import_roots(path) & BANNED_IMPORT_ROOTS
    }
    assert violations == {}


def test_term_provenance_contract_rejects_unknown_or_removed_corpus_sources():
    assert ALLOWED_TERM_SOURCES == {
        "literal", "morphology", "source_acronym", "typo_correction",
        "validated_model", "ai_assisted_query_expansion",
    }
    with pytest.raises(ValueError, match="unsupported query-term provenance"):
        _validate_term_details([{"term": "x", "source": "forced_synonym"}])
    with pytest.raises(ValueError, match="unsupported query-term provenance"):
        _validate_term_details([{"term": "x", "source": "corpus", "supporting_paper_ids": []}])
