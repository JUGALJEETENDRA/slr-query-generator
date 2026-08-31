from pathlib import Path

from litsync_app.query.generator import ConceptGroup, compile_boolean_query


def group(label: str, terms: list[str], order: int) -> ConceptGroup:
    return ConceptGroup(
        label=label,
        role="other",
        terms=terms,
        compiled=True,
        source_order=order,
    )


def test_ieee_compiler_uses_command_search_all_metadata_per_term():
    query = compile_boolean_query(
        [
            group("Technology", ["large language models", "LLM"], 0),
            group("Evidence", ["systematic review", "systematic reviews", "literature review"], 1),
        ],
        "ieee_xplore",
    )

    assert query == (
        '("All Metadata":"large language models" OR "All Metadata":"LLM") AND '
        '("All Metadata":"systematic review*" OR "All Metadata":"literature review")'
    )


def test_pubmed_compiler_remains_fielded_and_unchanged():
    groups = [group("Technology", ["large language models", "LLM"], 0)]
    assert compile_boolean_query(groups, "pubmed") == (
        '("large language models"[tiab] OR "LLM"[tiab])'
    )


def test_site_contains_versioned_origin_local_query_bridge_without_fixed_limit():
    html = (Path(__file__).parents[2] / "web" / "slr_query_generator.html").read_text(
        encoding="utf-8"
    )

    assert "LITSYNC_QUERY_CONTEXT" in html
    assert "LITSYNC_QUERY_CONTEXT_CLEAR" in html
    assert "LITSYNC_EXTENSION_READY" in html
    assert "schema_version: LITSYNC_QUERY_SCHEMA_VERSION" in html
    assert "active_query_version: activeQueryVersion" in html
    assert "query_fingerprint:" in html
    assert "event.origin !== window.location.origin" in html
    assert "Query synced to extension" in html
    assert "Open PubMed" in html
    assert "Open IEEE Command Search" in html
    assert "limit: 100" not in html
