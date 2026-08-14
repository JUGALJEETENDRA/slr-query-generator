from __future__ import annotations

import re

from playwright.sync_api import sync_playwright


def test_experimental_collection_actual_site(tmp_path):
    export = tmp_path / "scopus.csv"
    export.write_text(
        "Authors,Title,Year,Source title,DOI,Abstract\nA,Visible workflow paper,2024,Journal,10.1/site,Useful evidence\n",
        encoding="utf-8",
    )
    queries = {
        "google_scholar": '"adaptive learning" students',
        "scopus": 'TITLE-ABS-KEY("adaptive learning" AND students)',
        "web_of_science": 'TS=("adaptive learning" AND students)',
        "ieee_xplore": '("All Metadata":"adaptive learning")',
        "pubmed": '("adaptive learning"[tiab]) AND students[tiab]',
    }
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://127.0.0.1:8000", wait_until="domcontentloaded")
        page.locator("#qi").fill("How does adaptive learning affect students?")
        page.evaluate(
            "queries => { generatedQueryVersions = {balanced: queries, high_recall: queries}; activeQueryVersion = 'balanced'; document.getElementById('results').classList.add('on'); }",
            queries,
        )
        page.locator("#tab-ls").click()
        page.get_by_role("button", name="Create collection run").click()
        page.wait_for_url(re.compile(r"collection_run="))
        run_url = page.url
        page.reload(wait_until="domcontentloaded")
        assert page.url == run_url
        source = page.locator("#collectionSources article").filter(has_text="Scopus")
        source.get_by_text("Upload export").locator("input").set_input_files(export)
        source.get_by_text("imported").wait_for()
        page.get_by_role("button", name="Finalize + deduplicate").click()
        page.get_by_role("link", name="clean dataset").wait_for()
        download_url = page.get_by_role("link", name="clean dataset").get_attribute("href")
        response = page.request.get("http://127.0.0.1:8000" + download_url)
        assert response.ok
        assert "Visible workflow paper" in response.text()
        browser.close()
