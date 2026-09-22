from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from litsync_app import app as server
from litsync_app.deduplication import MAPPED_KEYS, deduplicate, parse_upload_files


@pytest.fixture(params=['IEEE', 'GoogleScholar'])
def exports(request):
    pubmed = pd.DataFrame([{
        'PMID': '12345', 'Title': 'Shared paper', 'Authors': 'A. Smith',
        'Journal': 'Example Journal', 'Year': '2024', 'DOI': '10.1234/shared',
        'URL': '', 'Abstract': None,
    }])
    if request.param == 'IEEE':
        other = pd.DataFrame([{
            'Document Title': 'Shared paper', 'Authors': '',
            'Publication Title': '', 'Publication Year': '',
            'DOI': '10.1234/shared', 'PDF Link': '',
            'Link': 'https://example.org/paper',
            'Abstract': 'A usable abstract from the other database.',
            'Author Keywords': 'review; evidence',
        }])
    else:
        other = pd.DataFrame([{
            'Title': 'Shared paper', 'Authors': '', 'Publication': '',
            'Year': '', 'Citations': '7', 'DOI': '10.1234/shared',
            'URL': 'https://example.org/paper',
            'Abstract': 'A usable abstract from the other database.',
            'Keywords': 'review; evidence',
        }])
    return pubmed, other


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('extension', ['csv', 'xlsx'])
def test_export_duplicates_fill_metadata_and_preserve_abstract(tmp_path, exports, reverse, extension):
    paths = []
    for index, frame in enumerate(exports):
        path = tmp_path / f'export-{index}.{extension}'
        if extension == 'csv':
            frame.to_csv(path, index=False)
        else:
            frame.to_excel(path, index=False)
        paths.append(str(path))
    combined, total = parse_upload_files(paths[::-1] if reverse else paths)
    before = combined.copy(deep=True)
    result, removed = deduplicate(combined)
    assert (total, len(result), removed) == (2, 1, 1)
    row = result.iloc[0]
    assert row['Abstract'] == 'A usable abstract from the other database.'
    assert row['Authors'] == 'A. Smith'
    assert row['Source title'] == 'Example Journal'
    assert row['Year'] == '2024'
    assert row['DOI'] == '10.1234/shared'
    assert row['Keywords'] == 'review; evidence'
    assert row['Link']
    assert combined.iloc[1 if not reverse else 0]['Link'] == 'https://example.org/paper'
    pd.testing.assert_frame_equal(combined, before)


def records(*rows):
    return pd.DataFrame([{**dict.fromkeys(MAPPED_KEYS, ''), **row} for row in rows])


@pytest.mark.parametrize('blank', ['', '  \t', None, float('nan'), pd.NA])
@pytest.mark.parametrize('reverse', [False, True])
def test_blanks_never_replace_populated_fields(blank, reverse):
    full = dict(zip(MAPPED_KEYS, ['A. Smith', 'Paper', '2024', 'Journal', '0',
                                 '10.1234/paper', 'https://example.org', 'Abstract text']))
    empty = dict.fromkeys(MAPPED_KEYS, blank)
    empty['DOI'] = full['DOI']
    rows = [full, empty]
    result, removed = deduplicate(records(*(rows[::-1] if reverse else rows)))
    assert result.iloc[0].to_dict() == full
    assert removed == 1


def test_richer_record_is_base_and_safe_extensions_are_preserved():
    result, removed = deduplicate(records(
        {'DOI': 'doi', 'Title': 'First title', 'Abstract': 'Study findings...', 'Year': '2023'},
        {'DOI': 'doi', 'Title': 'Rich title', 'Abstract': 'Study findings demonstrate improvement.',
         'Authors': 'A. Smith', 'Year': '2024', 'Source title': 'Journal', 'Keywords': 'evidence'},
        {'DOI': 'doi', 'Authors': 'A. Smith; B. Jones', 'Link': 'https://example.org'},
    ))
    assert removed == 2
    row = result.iloc[0]
    assert row['Title'] == 'Rich title'
    assert row['Year'] == '2024'
    assert row['Abstract'] == 'Study findings demonstrate improvement.'
    assert row['Authors'] == 'A. Smith; B. Jones'
    assert row['Link'] == 'https://example.org'


def test_ambiguous_conflicts_keep_first_equally_rich_value():
    result, _ = deduplicate(records(
        {'DOI': 'doi', 'Abstract': 'First findings.', 'Year': '2024'},
        {'DOI': 'doi', 'Abstract': 'An unrelated and much longer abstract.', 'Year': '2025'},
    ))
    assert result.iloc[0]['Abstract'] == 'First findings.'
    assert result.iloc[0]['Year'] == '2024'


@pytest.mark.parametrize(('field', 'short', 'complete'), [
    ('Abstract', 'Study findings...', 'Study findings demonstrate improvement.'),
    ('Authors', 'A. Smith', 'A. Smith; B. Jones'),
    ('Keywords', 'evidence', 'review; evidence'),
    ('Source title', 'Example Journal', 'Example Journal of Research'),
])
def test_safe_field_extension_from_less_populated_duplicate(field, short, complete):
    result, removed = deduplicate(records(
        {'DOI': 'doi', 'Title': 'Paper', 'Year': '2024', field: short},
        {'DOI': 'doi', field: complete},
    ))
    assert result.iloc[0][field] == complete
    assert removed == 1


def test_unique_records_and_existing_doi_title_matching_remain_unchanged():
    original = records(
        {'DOI': 'doi-1', 'Title': 'Shared title'},
        {'DOI': 'doi-2', 'Title': 'Shared title'},
        {'Title': 'Shared title'},
        {'Title': ''},
        {'Title': ''},
        {'Title': 'Unique title', 'Abstract': None},
    )
    result, removed = deduplicate(original)
    pd.testing.assert_frame_equal(result, original)
    assert removed == 0
    duplicates = records(
        {'DOI': ' DOI-1 ', 'Title': 'Different title', 'Abstract': 'DOI abstract'},
        {'Title': ' SHARED   TITLE ', 'Abstract': 'Title abstract'},
    )
    result, removed = deduplicate(pd.concat([original, duplicates], ignore_index=True))
    assert removed == 2
    assert len(result) == len(original)
    assert result.iloc[0]['Abstract'] == 'DOI abstract'
    assert result.iloc[2]['Abstract'] == 'Title abstract'


def test_empty_input():
    result, removed = deduplicate(None)
    assert result.empty and list(result.columns) == MAPPED_KEYS and removed == 0
    original = records()
    result, removed = deduplicate(original)
    pd.testing.assert_frame_equal(result, original)
    assert removed == 0


def test_step2_upload_consolidates_output_and_retains_source_counts(exports):
    with TestClient(server.app) as client:
        response = client.post('/litsync', files=[
            ('files', (f'export-{index}.csv', frame.to_csv(index=False).encode(), 'text/csv'))
            for index, frame in enumerate(exports)
        ])
        body = response.json()
        assert body['status'] == 'success'
        assert body['counts'] == {'initial': 2, 'deduped': 1, 'duplicates_removed': 1}
        output = pd.read_csv(Path(server.OUTPUT_DIR) / 'imports' / body['import_id'] / body['output_filename'])
        assert len(output) == 1
        assert output.iloc[0]['Abstract'] == 'A usable abstract from the other database.'
        assert output.iloc[0]['Keywords'] == 'review; evidence'
        assert body['prisma']['identification']['records_identified'] == 2
        assert body['prisma']['identification']['duplicate_records_removed'] == 1
        assert body['prisma']['identification']['source_files'] == [
            {'name': 'export-0.csv', 'records': 1},
            {'name': 'export-1.csv', 'records': 1},
        ]
