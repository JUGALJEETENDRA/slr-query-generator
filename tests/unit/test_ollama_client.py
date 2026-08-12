import ollama_client


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_ask_ollama_uses_configured_gptoss_profile(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(url=url, json=json, timeout=timeout)
        return FakeResponse({"response": '{"decision":"KEEP"}'})

    monkeypatch.setenv("LOCAL_MODEL", "gpt-oss:20b")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "30m")
    monkeypatch.setattr(ollama_client.requests, "post", fake_post)

    assert ollama_client.ask_ollama("test", model=None) == '{"decision":"KEEP"}'
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["json"]["model"] == "gpt-oss:20b"
    assert captured["json"]["options"]["num_ctx"] == 8192
    assert captured["json"]["keep_alive"] == "30m"


def test_ollama_status_marks_selected_model_installed(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests,
        "get",
        lambda *args, **kwargs: FakeResponse({"models": [{"name": "gpt-oss:20b"}]}),
    )

    status = ollama_client.ollama_status("gpt-oss:20b")
    assert status["reachable"] is True
    assert status["model_installed"] is True
