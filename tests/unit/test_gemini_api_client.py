from pydantic import BaseModel

from litsync_app.integrations import gemini_api as gemini_client


class TinyResult(BaseModel):
    decision: str


class FakeResponse:
    status_code = 200
    ok = True
    headers = {}

    def json(self):
        return {"candidates": [{"content": {"parts": [{"text": '{"decision":"KEEP"}'}]}}]}


def test_structured_gemini_request_uses_header_key_and_json_schema(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return FakeResponse()

    monkeypatch.setattr(gemini_client.requests, "post", fake_post)
    client = gemini_client.GeminiAPIClient("secret-key")
    raw = client.generate("screen this", schema=TinyResult)

    assert raw == '{"decision":"KEEP"}'
    assert captured["headers"]["x-goog-api-key"] == "secret-key"
    assert "secret-key" not in captured["url"]
    config = captured["json"]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"]["properties"]["decision"]["type"] == "string"
    assert "default" not in str(config["responseJsonSchema"])
    assert "gemini-3.5-flash" in captured["url"]


def test_gemini_error_does_not_expose_key(monkeypatch):
    class Denied(FakeResponse):
        status_code = 403
        ok = False

    monkeypatch.setattr(gemini_client.requests, "post", lambda *args, **kwargs: Denied())
    client = gemini_client.GeminiAPIClient("never-print-this")
    try:
        client.generate("screen this")
    except gemini_client.GeminiAPIError as exc:
        assert "never-print-this" not in str(exc)
        assert "rejected the API key" in str(exc)
    else:
        raise AssertionError("Expected a Gemini API error")
