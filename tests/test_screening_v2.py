import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from batch_builder import build_screening_prompt, make_batches
from embedding_screener import EmbeddingScreener
from gemini_query_generator import (
    build_query_prompt, generate_queries_with_gemini, parse_query_response,
)
from response_parser import ResponseParseError, parse_batch_response
from bulk_screen import screen_csv


class FakeEmbeddingScreener:
    def __init__(self, model):
        self.model = model

    def score_titles(self, research_question, titles):
        return [0.2, 0.8]


class FakeBrowser:
    prompts = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def submit(self, prompt):
        self.prompts.append(prompt)
        return json.dumps({"results": [{
            "id": "paper_2",
            "decision": "KEEP",
            "reason": "Directly relevant",
            "required_evidence": "",
            "paper_contribution": "Automates screening",
        }]})


class FakeQueryBrowser:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def submit(self, prompt):
        return json.dumps({
            "concepts": {
                "PRIMARY": "permissioned blockchain",
                "DOMAIN": "supply chain traceability",
                "COMPARATOR": "centralised database",
            },
            "base_query": '("permissioned blockchain" OR "distributed ledger") AND ("supply chain traceability")',
        })


class FakeRetryQueryBrowser(FakeQueryBrowser):
    calls = 0

    def submit(self, prompt):
        type(self).calls += 1
        if type(self).calls == 1:
            return '{"concepts":{"PRIMARY":"blockchain"}}'
        return super().submit(prompt)


class ScreeningComponentsTests(unittest.TestCase):
    def test_gemini_query_generation(self):
        prompt = build_query_prompt("Does blockchain improve traceability?")
        self.assertIn("blockchain improve traceability", prompt)
        result = generate_queries_with_gemini(
            "Does blockchain improve traceability?", browser_factory=FakeQueryBrowser
        )
        self.assertEqual(result["provider"], "gemini")
        self.assertTrue(result["scopus"].startswith("TITLE-ABS-KEY("))
        self.assertIn('"permissioned blockchain"[tiab]', result["pubmed"])

    def test_gemini_query_parser_accepts_nested_and_alternate_keys(self):
        result = parse_query_response(json.dumps({
            "search_strategy": {"boolean query": '("blockchain" AND "traceability")'},
            "concepts": {"primary": "blockchain", "domain": "traceability"},
        }))
        self.assertEqual(result["google_scholar"], '("blockchain" AND "traceability")')
        self.assertEqual(result["concepts"]["PRIMARY"], "blockchain")

    def test_gemini_query_generation_retries_invalid_shape(self):
        FakeRetryQueryBrowser.calls = 0
        result = generate_queries_with_gemini(
            "Does blockchain improve traceability?", browser_factory=FakeRetryQueryBrowser
        )
        self.assertEqual(FakeRetryQueryBrowser.calls, 2)
        self.assertEqual(result["provider"], "gemini")

    def test_cosine_similarity(self):
        self.assertAlmostEqual(EmbeddingScreener.cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(EmbeddingScreener.cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_batches_and_prompt(self):
        papers = [{"id": f"paper_{i}", "title": "T", "abstract": "A"} for i in range(3)]
        self.assertEqual([len(batch) for batch in make_batches(papers, 2)], [2, 1])
        prompt = build_screening_prompt("RQ", papers[:1])
        self.assertIn('"id": "paper_0"', prompt)
        self.assertIn("untrusted data", prompt)

    def test_parser_accepts_fenced_json_and_preserves_order(self):
        text = '```json\n{"results":[' \
               '{"id":"b","decision":"reject","reason":"x"},' \
               '{"id":"a","decision":"keep","reason":"y"}]}\n```'
        result = parse_batch_response(text, ["a", "b"])
        self.assertEqual([item["id"] for item in result], ["a", "b"])
        self.assertEqual(result[0]["decision"], "KEEP")

    def test_parser_rejects_missing_ids(self):
        with self.assertRaises(ResponseParseError):
            parse_batch_response('{"results":[]}', ["paper_1"])


class HybridWorkflowTests(unittest.TestCase):
    def test_hybrid_filters_and_batches(self):
        FakeBrowser.prompts = []
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "papers.csv")
            output = os.path.join(directory, "outputs", "screened.csv")
            pd.DataFrame([
                {"Title": "Unrelated", "Abstract": "No match"},
                {"Title": "Relevant", "Abstract": "Strong match"},
            ]).to_csv(source, index=False)

            with patch("bulk_screen.EmbeddingScreener", FakeEmbeddingScreener):
                summary = screen_csv(
                    source,
                    "Can AI automate reviews?",
                    output_path=output,
                    mode="hybrid",
                    embedding_threshold=0.35,
                    browser_factory=FakeBrowser,
                )

            self.assertEqual(summary["embedding_rejected"], 1)
            self.assertEqual(summary["sent_to_gemini"], 1)
            self.assertEqual(summary["keep"], 1)
            self.assertEqual(summary["reject"], 1)
            self.assertEqual(len(FakeBrowser.prompts), 1)
            screened = pd.read_csv(output)
            self.assertEqual(screened["Decision"].tolist(), ["REJECT", "KEEP"])
            for filename in (
                "included_studies.csv", "excluded_studies.csv", "maybe_studies.csv", "summary.csv"
            ):
                self.assertTrue(os.path.exists(os.path.join(directory, "outputs", filename)))

    def test_local_screens_every_titled_paper_without_embedding_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "papers.csv")
            output = os.path.join(directory, "screened.csv")
            pd.DataFrame([
                {"Title": "First", "Abstract": "A"},
                {"Title": "Second", "Abstract": "B"},
            ]).to_csv(source, index=False)
            fake_result = {
                "decision": "MAYBE", "reason": "Needs full text",
                "required_evidence": "Methods", "paper_contribution": "",
            }
            with patch("bulk_screen.screen_paper", return_value=fake_result) as local_screen, \
                    patch("bulk_screen.EmbeddingScreener") as embedding:
                summary = screen_csv(source, "RQ", output_path=output, mode="local")

            self.assertEqual(local_screen.call_count, 2)
            embedding.assert_not_called()
            self.assertEqual(summary["maybe"], 2)
            self.assertEqual(summary["sent_to_gemini"], 0)


if __name__ == "__main__":
    unittest.main()
