import pytest
from pydantic import BaseModel, ConfigDict
from typing import Literal

from litsync_app.screening.external.engine import InjectedStructuredEngine, parse_structured_model_output
from litsync_app.screening.local.engine import LocalAIOutputError


class Decision(BaseModel):
    decision: str
    reason: str


class StrictDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["KEEP", "MAYBE", "REJECT"]
    reason: str


def _failed_diagnostic(raw, schema=StrictDecision):
    captured = {}
    with pytest.raises(LocalAIOutputError):
        parse_structured_model_output(raw, schema, diagnostic_sink=captured.update)
    return captured


def _successful_diagnostic(raw, schema=StrictDecision):
    captured = {}
    parsed = parse_structured_model_output(
        raw,
        schema,
        diagnostic_sink=captured.update,
    )
    return parsed, captured


def test_plain_json_is_accepted_identically():
    raw = '{"decision":"KEEP","reason":"Relevant"}'
    assert parse_structured_model_output(raw, Decision) == {
        "decision": "KEEP",
        "reason": "Relevant",
    }


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


def test_malformed_response_is_classified_as_no_json_decodable_candidate():
    diagnostic = _failed_diagnostic('{"decision":"KEEP"')
    assert diagnostic["failure_code"] == "no_json_decodable_candidate"
    assert diagnostic["json_decodable_candidate_count"] == 0


def test_json_list_is_classified_as_decoded_non_dictionary():
    diagnostic = _failed_diagnostic('["KEEP", "private paper title"]')
    assert diagnostic["failure_code"] == "json_decoded_but_not_object"
    assert diagnostic["non_dictionary_candidate_count"] > 0
    assert diagnostic["decoded_top_level_type"] == "list"


def test_missing_field_is_classified_as_schema_validation_failure():
    diagnostic = _failed_diagnostic('{"decision":"KEEP"}')
    assert diagnostic["failure_code"] == "schema_validation_failed"
    assert diagnostic["schema_validation_failure_count"] > 0
    assert diagnostic["validation_error_types"] == ["missing"]
    assert diagnostic["validation_error_locations"] == [["reason"]]


def test_wrong_enum_and_forbidden_extra_expose_only_safe_error_metadata():
    secret = "PRIVATE_TITLE_DO_NOT_LOG"
    diagnostic = _failed_diagnostic(
        '{"decision":"INVALID","reason":"%s","unexpected":"SECRET_ABSTRACT"}'
        % secret
    )
    assert set(diagnostic["validation_error_types"]) == {
        "literal_error",
        "extra_forbidden",
    }
    assert ["decision"] in diagnostic["validation_error_locations"]
    assert ["unexpected"] in diagnostic["validation_error_locations"]
    serialized = str(diagnostic)
    assert secret not in serialized
    assert "SECRET_ABSTRACT" not in serialized
    assert '"decision":"INVALID"' not in serialized


def test_diagnostic_counts_and_lists_are_bounded():
    fields = {
        f"field_{index}": (str, ...)
        for index in range(30)
    }
    from pydantic import create_model

    ManyFields = create_model("ManyFields", __config__=ConfigDict(extra="forbid"), **fields)
    diagnostic = _failed_diagnostic("{}", ManyFields)
    assert diagnostic["validation_error_count"] == 30
    assert len(diagnostic["validation_error_types"]) == 10
    assert len(diagnostic["validation_error_locations"]) == 10
    assert len(diagnostic["validation_error_messages"]) == 10
    assert len(str(diagnostic).encode("utf-8")) <= 8192


def test_complete_valid_json_reports_decodable_and_schema_valid():
    _, diagnostic = _successful_diagnostic(
        '{"decision":"KEEP","reason":"Relevant"}'
    )
    assert diagnostic["full_response_json_decodable"] is True
    assert diagnostic["full_response_top_level_type"] == "dict"
    assert diagnostic["full_response_schema_valid"] is True
    assert diagnostic["full_response_raw_decode_succeeded"] is True
    assert diagnostic["full_response_raw_decode_consumed_ratio"] == 1.0


def test_prose_distinguishes_complete_response_from_embedded_candidate():
    _, diagnostic = _successful_diagnostic(
        'Result: {"decision":"KEEP","reason":"Relevant"}'
    )
    assert diagnostic["full_response_json_decodable"] is False
    assert diagnostic["full_response_raw_decode_succeeded"] is False
    assert diagnostic["candidate_source"] == "embedded_json"
    assert diagnostic["full_response_json_error_message"] == "Expecting value"


def test_truncated_outer_json_reports_decode_error_near_end():
    diagnostic = _failed_diagnostic(
        '{"decision":"KEEP","reason":"Relevant"'
    )
    assert diagnostic["full_response_json_decodable"] is False
    assert diagnostic["full_response_json_error_type"] == "JSONDecodeError"
    assert diagnostic["full_response_json_error_position_ratio"] > 0.9
    assert diagnostic["full_response_brace_balance"] == 1


def test_unterminated_string_is_identified_without_recording_content():
    secret = "PRIVATE_UNTERMINATED_CONTENT"
    diagnostic = _failed_diagnostic(
        '{"decision":"KEEP","reason":"' + secret
    )
    assert diagnostic["full_response_inside_string_at_end"] is True
    assert diagnostic["full_response_json_error_message"].startswith(
        "Unterminated string"
    )
    assert secret not in str(diagnostic)


def test_missing_braces_and_brackets_have_quote_aware_balances():
    diagnostic = _failed_diagnostic(
        '{"decision":"KEEP","reason":"quoted {[}]","extra":['
    )
    assert diagnostic["full_response_brace_balance"] == 1
    assert diagnostic["full_response_bracket_balance"] == 1


def test_braces_inside_valid_quoted_string_do_not_change_balance():
    _, diagnostic = _successful_diagnostic(
        '{"decision":"KEEP","reason":"quoted {[}]}"}'
    )
    assert diagnostic["full_response_brace_balance"] == 0
    assert diagnostic["full_response_bracket_balance"] == 0


def test_multiple_top_level_values_are_distinct_from_truncation():
    parsed, diagnostic = _successful_diagnostic(
        '{"decision":"KEEP","reason":"first"} '
        '{"decision":"REJECT","reason":"second"}'
    )
    assert parsed["decision"] == "KEEP"
    assert diagnostic["candidate_source"] == "embedded_json"
    assert diagnostic["full_response_json_decodable"] is False
    assert diagnostic["full_response_json_error_message"] == "Extra data"
    assert diagnostic["full_response_raw_decode_succeeded"] is True
    assert diagnostic["full_response_raw_decode_consumed_ratio"] < 1
    assert diagnostic["full_response_trailing_nonwhitespace_characters"] > 0
    assert diagnostic["full_response_brace_balance"] == 0


def test_complete_schema_failure_is_distinct_from_json_decode_failure():
    diagnostic = _failed_diagnostic('{"decision":"KEEP"}')
    assert diagnostic["full_response_json_decodable"] is True
    assert diagnostic["full_response_schema_valid"] is False
    assert diagnostic["full_response_json_error_type"] == ""
    assert diagnostic["candidate_source"] == "complete_response"
    assert diagnostic["validation_error_locations"] == [["reason"]]


def test_diagnostic_sink_failure_does_not_alter_successful_parsing():
    def broken_sink(_payload):
        raise RuntimeError("diagnostic storage failed")

    assert parse_structured_model_output(
        '{"decision":"KEEP","reason":"Relevant"}',
        Decision,
        diagnostic_sink=broken_sink,
    ) == {"decision": "KEEP", "reason": "Relevant"}


def test_diagnostic_sink_failure_does_not_replace_original_parser_error():
    def broken_sink(_payload):
        raise RuntimeError("diagnostic storage failed")

    with pytest.raises(LocalAIOutputError) as captured:
        parse_structured_model_output(
            "not json",
            Decision,
            diagnostic_sink=broken_sink,
        )
    assert "response was not valid structured JSON" in str(captured.value)


def test_injected_engine_accepts_normal_gemini_markdown_json():
    class WebEngine:
        def ask_structured(self, prompt, schema, model):
            return '```json\n{"decision":"MAYBE","reason":"Insufficient abstract"}\n```'

    generated = InjectedStructuredEngine(WebEngine()).generate("gemini-web", "prompt", Decision)
    assert generated.value["decision"] == "MAYBE"
