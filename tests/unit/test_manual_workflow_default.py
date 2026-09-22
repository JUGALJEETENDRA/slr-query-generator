from pathlib import Path
from fastapi.testclient import TestClient
from litsync_app.app import app


def test_manual_pipeline_is_default_without_mode_switches():
    html = TestClient(app).get('/').text
    assert 'id="manualWorkspace"' in html
    assert '#manualWorkspace { display: none' not in html
    assert 'id="tab-gen" role="tab" aria-selected="true"' in html
    for panel in ('panel-gen', 'panel-ls', 'panel-scr'):
        assert f'id="{panel}"' in html
    for retired in ('agenticWorkspace', 'startAgenticRun', 'showManualWorkspace', 'showAgenticWorkspace', '/agentic-runs', 'Advanced / Manual Mode', 'Back to Agentic Mode'):
        assert retired not in html
    assert 'Browser Extensions' in html


def test_agentic_endpoints_and_implementation_are_removed():
    assert not any('/agentic-runs' in getattr(route, 'path', '') for route in app.routes)
    client = TestClient(app)
    assert client.post('/agentic-runs', json={'topic': 'unused'}).status_code == 404
    assert client.get('/agentic-runs/unused').status_code == 404
    root = Path(__file__).resolve().parents[2]
    assert not list((root / 'litsync_app/paper_collection').glob('*.py'))
