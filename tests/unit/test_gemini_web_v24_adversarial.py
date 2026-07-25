import pandas as pd
import pytest

from evaluation.gemini_web_v24_adversarial import prepare_adversarial_benchmark


def _source(count=20):
    challenge_types = [
        "exact match", "explicit exclusion", "adjacent scope",
        "incidental relevance", "insufficient evidence",
    ]
    return pd.DataFrame([{
        "Title": f"Paper {index}",
        "Abstract": "" if index == 0 else f"Abstract {index}",
        "Year": 2020 + index % 5,
        "Gold_Decision": ("KEEP", "MAYBE", "REJECT")[index % 3],
        "Gold_Rationale": f"Independent rationale {index}",
        "Challenge_Type": challenge_types[index % len(challenge_types)],
    } for index in range(count)])


def test_adversarial_builder_blinds_labels_without_balancing(tmp_path):
    source = tmp_path / "source.csv"
    papers = tmp_path / "papers.csv"
    gold = tmp_path / "gold.csv"
    _source().to_csv(source, index=False)
    result = prepare_adversarial_benchmark(
        adjudicated_csv=source, papers_out=papers, gold_out=gold,
    )
    screening = pd.read_csv(papers, keep_default_na=False)
    sidecar = pd.read_csv(gold, keep_default_na=False)
    assert result["labels_blinded"]
    assert set(screening.columns) == {"Title", "Abstract", "Year"}
    assert set(sidecar["Gold_Decision"]) == {"KEEP", "MAYBE", "REJECT"}
    assert screening.loc[0, "Abstract"] == ""


def test_adversarial_builder_rejects_easy_or_unadjudicated_fixture(tmp_path):
    source = tmp_path / "source.csv"
    papers = tmp_path / "papers.csv"
    gold = tmp_path / "gold.csv"
    frame = _source()
    frame["Challenge_Type"] = "one easy group"
    frame.loc[0, "Gold_Rationale"] = ""
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError):
        prepare_adversarial_benchmark(
            adjudicated_csv=source, papers_out=papers, gold_out=gold,
        )
    assert not papers.exists()
    assert not gold.exists()
