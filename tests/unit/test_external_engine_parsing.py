import pytest
from pydantic import BaseModel

from litsync_app.screening.external.engine import InjectedStructuredEngine, parse_structured_model_output
from litsync_app.screening.local.engine import LocalAIOutputError


class Decision(BaseModel):
    decision: str
    reason: str


def test_gemini_web_fenced_json_is_accepted():
    parsed = parse_structured_model_output(
        'Here is the result:\n```json\n{"decision":"KEEP","reason":"Relevant"}\n```',
        Decision,
    )
    assert parsed == {"decision": "KEEP", "reason": "Relevant"}


def test_parser_skips_unrelated_object_before_schema_matching_result():
    parsed = parse_structured_model_output(
        'Example {"wrong":true}. Final {"decision":"REJECT","reason":"Outside scope"}',
        Decision,
    )
    assert parsed["decision"] == "REJECT"


@pytest.mark.parametrize("raw", ["", "Gemini is still thinking", "```json\nnot-json\n```"])
def test_empty_or_unstructured_web_output_has_friendly_retry_error(raw):
    with pytest.raises(LocalAIOutputError) as error:
        parse_structured_model_output(raw, Decision)
    assert "Expecting value" not in str(error.value)


def test_injected_engine_accepts_normal_gemini_markdown_json():
    class WebEngine:
        def ask_structured(self, prompt, schema, model):
            return '```json\n{"decision":"MAYBE","reason":"Insufficient abstract"}\n```'

    generated = InjectedStructuredEngine(WebEngine()).generate("gemini-web", "prompt", Decision)
    assert generated.value["decision"] == "MAYBE"
