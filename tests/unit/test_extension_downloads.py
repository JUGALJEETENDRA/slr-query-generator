import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest
from fastapi.testclient import TestClient
from litsync_app import app as module


@pytest.mark.parametrize('collector', ['google-scholar', 'pubmed', 'ieee'])
def test_download_is_current_validated_package(collector):
    response = TestClient(module.app).get(f'/extensions/{collector}/download')
    assert response.status_code == 200
    assert response.headers['content-type'] == 'application/zip'
    assert f'litsync-{collector}-extension.zip' in response.headers['content-disposition']
    assert response.headers['cache-control'] == 'no-cache'
    original = module.PROJECT_ROOT / 'browser_extensions' / module.EXTENSION_PACKAGES[collector]
    assert hashlib.sha256(response.content).digest() == hashlib.sha256(original.read_bytes()).digest()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['manifest_version'] == 3
        assert 'LitSync' in manifest['name']
        assert ('Scholar' if collector == 'google-scholar' else collector).lower() in manifest['name'].lower()
        assert not any('..' in Path(name).parts for name in archive.namelist())


def test_unknown_download_cannot_access_arbitrary_files():
    client = TestClient(module.app)
    assert client.get('/extensions/unknown/download').status_code == 404
    assert client.get('/extensions/README.md/download').status_code == 404


def test_missing_package_reports_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(module, 'PROJECT_ROOT', tmp_path)
    assert TestClient(module.app).get('/extensions/pubmed/download').status_code == 503


def test_download_links_are_direct_and_limited_to_supported_collectors():
    html = module.HTML_FILE.read_text(encoding='utf-8')
    section = html.split('<section aria-label="Browser Extensions"', 1)[1].split('</section>', 1)[0]
    assert section.count('<a ') == 3
    for collector in module.EXTENSION_PACKAGES:
        assert f'href="/extensions/{collector}/download"' in section
        assert f'download="litsync-{collector}-extension.zip"' in section
    assert 'onclick' not in section
    assert 'Load unpacked' not in section
