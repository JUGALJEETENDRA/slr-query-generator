import os
import re
from typing import List, Dict, Tuple

import pandas as pd


MAPPED_KEYS = [
    'Authors',
    'Title',
    'Year',
    'Source title',
    'Cited by',
    'DOI',
    'Link',
    'Abstract',
]


def _normalize_str(s: str) -> str:
    if s is None:
        return ''
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s


def _try_get(row: Dict, keys: List[str]) -> str:
    for k in keys:
        if k in row and pd.notna(row[k]):
            return str(row[k])
    return ''


def detect_source_and_map(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=MAPPED_KEYS)

    # Normalize headers for detection
    cols = [str(c).strip() for c in df.columns]
    headers = {c.lower(): c for c in cols}

    def has_col(name: str) -> bool:
        return name.lower() in headers

    format_name = 'Unknown'

    # IEEE: "Document Title", "Article Citation Count"
    if has_col('Document Title') or has_col('Article Citation Count'):
        format_name = 'IEEE'
    # Web of Science
    elif has_col('UT (Unique WOS ID)') or has_col('Times Cited, All Databases') or has_col('Article Title'):
        format_name = 'WOS'
    # Scopus
    elif has_col('EID') or (has_col('Title') and has_col('Source title')):
        format_name = 'Scopus'
    # PubMed
    elif has_col('PMID') or has_col('PubMed ID') or has_col('Journal'):
        format_name = 'PubMed'
    # Google Scholar (best-effort)
    elif has_col('Citations') or (has_col('Authors') and has_col('Title') and has_col('Year')):
        format_name = 'GoogleScholar'
    else:
        format_name = 'Fallback'

    mapped_rows = []

    # Make row dict with original columns
    records = df.to_dict(orient='records')
    for row in records:
        mapped = {k: '' for k in MAPPED_KEYS}

        if format_name == 'IEEE':
            mapped['Authors'] = _try_get(row, ['Authors'])
            mapped['Title'] = _try_get(row, ['Document Title'])
            mapped['Year'] = _try_get(row, ['Publication Year'])
            mapped['Source title'] = _try_get(row, ['Publication Title'])
            mapped['Cited by'] = _try_get(row, ['Article Citation Count'])
            mapped['DOI'] = _try_get(row, ['DOI'])
            mapped['Link'] = _try_get(row, ['PDF Link', 'Link'])
            mapped['Abstract'] = _try_get(row, ['Abstract'])
        elif format_name == 'Scopus':
            mapped['Authors'] = _try_get(row, ['Authors'])
            mapped['Title'] = _try_get(row, ['Title'])
            mapped['Year'] = _try_get(row, ['Year', 'Publication Year'])
            mapped['Source title'] = _try_get(row, ['Source title', 'Source Title'])
            mapped['Cited by'] = _try_get(row, ['Cited by', 'Article Citation Count'])
            mapped['DOI'] = _try_get(row, ['DOI'])
            mapped['Link'] = _try_get(row, ['Link', 'URL'])
            mapped['Abstract'] = _try_get(row, ['Abstract'])
        elif format_name == 'WOS':
            mapped['Authors'] = _try_get(row, ['Authors'])
            mapped['Title'] = _try_get(row, ['Article Title'])
            mapped['Year'] = _try_get(row, ['Publication Year'])
            mapped['Source title'] = _try_get(row, ['Source Title'])
            mapped['Cited by'] = _try_get(row, ['Times Cited, All Databases'])
            mapped['DOI'] = _try_get(row, ['DOI'])
            mapped['Link'] = _try_get(row, ['DOI Link', 'Link', 'URL'])
            mapped['Abstract'] = _try_get(row, ['Abstract'])
        elif format_name == 'PubMed':
            mapped['Authors'] = _try_get(row, ['Authors', 'Author'])
            mapped['Title'] = _try_get(row, ['Title', 'Article Title'])
            mapped['Year'] = _try_get(row, ['Year', 'Publication Year'])
            mapped['Source title'] = _try_get(row, ['Journal', 'Source Title'])
            mapped['Cited by'] = _try_get(row, ['Cited by', 'Citations'])
            mapped['DOI'] = _try_get(row, ['DOI'])
            mapped['Link'] = _try_get(row, ['URL', 'Link'])
            mapped['Abstract'] = _try_get(row, ['Abstract'])
        elif format_name == 'GoogleScholar':
            mapped['Authors'] = _try_get(row, ['Authors', 'Author'])
            mapped['Title'] = _try_get(row, ['Title', 'Article Title'])
            mapped['Year'] = _try_get(row, ['Year', 'Publication Year'])
            mapped['Source title'] = _try_get(row, ['Publication', 'Journal', 'Source title'])
            mapped['Cited by'] = _try_get(row, ['Citations', 'Cited by'])
            mapped['DOI'] = _try_get(row, ['DOI'])
            mapped['Link'] = _try_get(row, ['URL', 'Link'])
            mapped['Abstract'] = _try_get(row, ['Abstract'])
        else:
            # Fallback best-effort mapping
            mapped['Authors'] = _try_get(row, ['Authors', 'Author'])
            mapped['Title'] = _try_get(row, ['Title', 'Article Title', 'Document Title'])
            mapped['Year'] = _try_get(row, ['Year', 'Publication Year'])
            mapped['Source title'] = _try_get(row, ['Source title', 'Source Title', 'Publication Title'])
            mapped['Cited by'] = _try_get(row, ['Cited by', 'Times Cited', 'Times Cited, All Databases'])
            mapped['DOI'] = _try_get(row, ['DOI'])
            mapped['Link'] = _try_get(row, ['Link', 'URL'])
            mapped['Abstract'] = _try_get(row, ['Abstract'])

        # DOI normalization
        if mapped['DOI']:
            mapped['DOI'] = str(mapped['DOI']).strip()

        # Ensure link via DOI if missing
        if not mapped['Link'] and mapped['DOI']:
            doi = str(mapped['DOI']).strip()
            if doi.lower().startswith('http'):
                mapped['Link'] = doi
            else:
                mapped['Link'] = 'https://doi.org/' + doi

        mapped_rows.append(mapped)

    return pd.DataFrame(mapped_rows, columns=MAPPED_KEYS)


def deduplicate(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame(columns=MAPPED_KEYS), 0

    unique_rows = []
    seen_dois = set()
    seen_titles = set()
    removed = 0

    for _, row in df.iterrows():
        doi = _normalize_str(row.get('DOI', ''))
        title = _normalize_str(row.get('Title', '')).replace('  ', ' ')

        if doi:
            if doi in seen_dois:
                removed += 1
                continue
            seen_dois.add(doi)
            unique_rows.append(row.to_dict())
        else:
            if title in seen_titles and title:
                removed += 1
                continue
            if title:
                seen_titles.add(title)
            unique_rows.append(row.to_dict())

    return pd.DataFrame(unique_rows, columns=MAPPED_KEYS), removed


def parse_upload_files(file_paths: List[str]) -> Tuple[pd.DataFrame, int]:
    all_mapped = []
    total_rows = 0

    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(path)
        elif ext in ['.xls', '.xlsx']:
            xls = pd.ExcelFile(path)
            first_sheet = xls.sheet_names[0]
            df = pd.read_excel(path, sheet_name=first_sheet)
        else:
            continue

        if df is None or df.empty:
            continue

        total_rows += len(df)
        mapped = detect_source_and_map(df, os.path.basename(path))
        all_mapped.append(mapped)

    if all_mapped:
        combined = pd.concat(all_mapped, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=MAPPED_KEYS)

    return combined, total_rows

