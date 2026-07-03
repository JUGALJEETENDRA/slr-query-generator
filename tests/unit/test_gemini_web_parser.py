import pytest

from gemini_web_parser import GeminiResponseParseError, parse_gemini_screening_response


def test_parse_json_response_maps_decisions():
    response = """
    {
      "decisions": [
        {"id": "1", "decision": "Include", "reason": "Directly relevant."},
        {"id": "2", "decision": "Exclude", "reason": "Outside scope."},
        {"id": "3", "decision": "Maybe", "reason": "Not enough detail."}
      ]
    }
    """

    decisions = parse_gemini_screening_response(response, {"1", "2", "3"})

    assert [decision.decision for decision in decisions] == ["KEEP", "REJECT", "MAYBE"]


def test_parse_fenced_json_response():
    response = """
    ```json
    {"decisions": [{"id": "paper-1", "decision": "keep", "reason": "Relevant."}]}
    ```
    """

    decisions = parse_gemini_screening_response(response, {"paper-1"})

    assert decisions[0].paper_id == "paper-1"
    assert decisions[0].decision == "KEEP"


def test_parse_line_response():
    response = "Paper ID: A1 - Exclude - The paper studies another task."

    decisions = parse_gemini_screening_response(response, {"A1"})

    assert decisions[0].decision == "REJECT"
    assert "another task" in decisions[0].reason


def test_rejects_missing_ids():
    response = '{"decisions": [{"id": "1", "decision": "Include", "reason": "Relevant."}]}'

    with pytest.raises(GeminiResponseParseError):
        parse_gemini_screening_response(response, {"1", "2"})
