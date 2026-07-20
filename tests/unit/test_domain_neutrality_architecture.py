from __future__ import annotations

import ast
from pathlib import Path

import pytest

from direct_ai_generator import ALLOWED_TERM_SOURCES, _validate_term_details
from query_framework.registry import create_default_strategy_registry


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_FILES = (
    ROOT / "server.py",
    ROOT / "direct_ai_generator.py",
    ROOT / "screening_strategies.py",
    ROOT / "bulk_screen.py",
    *sorted((ROOT / "local_ai").glob("*.py")),
    *sorted((ROOT / "external_ai").glob("*.py")),
    *sorted((ROOT / "query_framework").glob("*.py")),
)
BANNED_IMPORT_ROOTS = {
    "archive", "benchmark", "classifier", "registries", "ontology_expander",
    "acronym_expander", "comparator_registry", "generator", "extractor", "validator",
    "experiments",
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


def test_default_strategy_registry_contains_only_nonexperimental_strategies():
    registry = create_default_strategy_registry(client=object(), model="unused")
    metadata = registry.list_metadata()
    assert [item.id for item in metadata] == ["direct_ai"]
    assert all(not item.experimental for item in metadata)
    with pytest.raises(ValueError, match="unavailable"):
        registry.get("litsync_workflow")


def test_term_provenance_contract_rejects_unknown_or_unsupported_corpus_sources():
    assert ALLOWED_TERM_SOURCES == {
        "literal", "morphology", "source_acronym", "typo_correction",
        "validated_model", "corpus",
    }
    with pytest.raises(ValueError, match="unsupported query-term provenance"):
        _validate_term_details([{"term": "x", "source": "forced_synonym"}])
    with pytest.raises(ValueError, match="lacks supporting papers"):
        _validate_term_details([{"term": "x", "source": "corpus", "supporting_paper_ids": []}])
