import pytest

from model_lab.service import LocalModelLab, _comparison, _gold_metrics, _run_metrics, parse_csv_papers


def test_model_lab_csv_accepts_title_only_rows_and_optional_gold_labels():
    papers = parse_csv_papers(b"Title,Abstract,Year,Gold_Decision\nTitle only,,2024,KEEP\nWith abstract,Useful text,2025,UNSURE\n,,2026,REJECT\n")
    assert [(paper.title, paper.abstract, paper.gold_decision) for paper in papers] == [
        ("Title only", "", "KEEP"), ("With abstract", "Useful text", "UNSURE")
    ]


def test_model_lab_metrics_and_comparison_are_deterministic():
    def result(decision, valid=True):
        return {"trace": [], "final": {"assessment": {"decision": decision}, "validation": {"valid": valid, "exact_quote_count": 1}, "metrics": {"wall_seconds": 1.2, "tokens_per_second": 10}}}
    rows = [
        {"paper": {"id": "1", "title": "A"}, "candidates": {"one": result("KEEP"), "two": result("REJECT")}},
        {"paper": {"id": "2", "title": "B"}, "candidates": {"one": result("MAYBE", False), "two": result("MAYBE", False)}},
    ]
    metrics = _run_metrics([row["candidates"]["one"] for row in rows])
    assert metrics["decisions"] == {"KEEP": 1, "MAYBE": 1, "REJECT": 0}
    assert metrics["validated"] == 1
    comparison = _comparison(rows, ["one", "two"])
    assert comparison[0]["agreement_rate"] == 0.5
    assert comparison[0]["disagreements"][0]["paper_id"] == "1"


def test_model_lab_gold_metrics_use_only_human_keep_and_reject_labels():
    def row(identifier, gold, decision):
        return {
            "paper": {"id": identifier, "gold_decision": gold},
            "candidates": {"one": {"final": {"assessment": {"decision": decision}}}},
        }

    metrics = _gold_metrics([
        row("1", "KEEP", "KEEP"),
        row("2", "KEEP", "MAYBE"),
        row("3", "KEEP", "REJECT"),
        row("4", "REJECT", "KEEP"),
        row("5", "UNSURE", "KEEP"),
    ], "one")
    assert metrics == {
        "labeled_papers": 4,
        "unsure_or_unlabeled": 1,
        "keep_maybe_recall": 0.6667,
        "false_reject_rate": 0.3333,
        "definitive_keep_precision": 0.5,
    }


def test_model_lab_rejects_incomplete_experiments_before_starting_a_thread(tmp_path):
    lab = LocalModelLab(tmp_path)
    with pytest.raises(ValueError, match="research question"):
        lab.create_job({"models": ["qwen2.5:3b"], "papers": [{"title": "Paper"}]})
    with pytest.raises(ValueError, match="at least one installed local model"):
        lab.create_job({"research_question": "A valid question", "papers": [{"title": "Paper"}]})
    with pytest.raises(ValueError, match="at least one paper"):
        lab.create_job({"research_question": "A valid question", "models": ["qwen2.5:3b"]})
