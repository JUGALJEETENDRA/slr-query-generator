from __future__ import annotations

from pathlib import Path
from urllib.parse import quote_plus


class CollectionNeedsAttention(RuntimeError):
    pass


SOURCE_INFO = {
    "google_scholar": {
        "name": "Google Scholar", "url": "https://scholar.google.com/",
        "mode": "assisted",
        "reason": "Google Scholar restricts automated bulk access; use a visible search and Zotero export.",
        "export_steps": [
            "Open the exact search below; keep relevance order and do not add filters.",
            "Use the open-source Zotero Connector to save the first results to a temporary collection.",
            "Export that Zotero collection as RIS, then upload it here.",
        ],
        "recommended_format": "Zotero RIS",
    },
    "ieee_xplore": {
        "name": "IEEE Xplore", "url": "https://ieeexplore.ieee.org/search/advanced",
        "mode": "assisted",
        "reason": "IEEE restricts automated agents; use the visible interface and upload the native CSV.",
        "export_steps": [
            "Open IEEE Xplore and paste the exact query; keep relevance order.",
            "Select up to the first 100 results and choose Export > CSV.",
            "Include citation information and abstracts, then upload the CSV here.",
        ],
        "recommended_format": "CSV with abstracts",
    },
    "scopus": {
        "name": "Scopus", "url": "https://www.scopus.com/search/form.uri",
        "mode": "assisted", "reason": "Scopus requires authenticated native export.",
        "export_steps": [
            "Open Scopus Advanced Search and paste the exact TITLE-ABS-KEY query.",
            "Keep relevance order, select the first 100 documents, then choose Export > CSV.",
            "Select Citation, Bibliographical, and Abstract & Keywords fields; upload the downloaded CSV.",
        ],
        "recommended_format": "Scopus CSV with abstracts",
    },
    "web_of_science": {
        "name": "Web of Science", "url": "https://www.webofscience.com/wos/woscc/advanced-search",
        "mode": "assisted", "reason": "Web of Science requires authenticated native export.",
        "export_steps": [
            "Open Web of Science Advanced Search and paste the exact TS query.",
            "Keep relevance order and export records 1-100.",
            "Choose Excel or RIS and Full Record; upload the downloaded file.",
        ],
        "recommended_format": "Excel/RIS, Full Record",
    },
    "pubmed": {
        "name": "PubMed", "url": "https://pubmed.ncbi.nlm.nih.gov/",
        "mode": "automated",
        "reason": "Public native CSV export; upload PubMed text/NBIB when abstracts are required.",
        "export_steps": [
            "Run browser for an automatic first-page CSV export.",
            "For abstract-complete records, use Save > PubMed or Citation Manager and upload the TXT/NBIB file.",
        ],
        "recommended_format": "Automatic CSV or abstract-rich PubMed text",
    },
}


def source_launch_url(source: str, query: str) -> str:
    if source == "google_scholar":
        return "https://scholar.google.com/scholar?q=" + quote_plus(query)
    if source == "ieee_xplore":
        return "https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=" + quote_plus(query)
    return str(SOURCE_INFO[source]["url"])


class NativeExportBrowser:
    """Deterministic Playwright adapters; never bypass CAPTCHA or access controls."""

    def __init__(self, profile_dir: str | Path = "private/experimental_collection/browser-profile"):
        self.profile_dir = Path(profile_dir)

    async def collect(self, source: str, query: str, limit: int, artifact_dir: Path) -> tuple[Path, str]:
        if source != "pubmed":
            raise CollectionNeedsAttention(SOURCE_INFO[source]["reason"])
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError("Playwright is not installed") from exc
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        playwright = await async_playwright().start()
        context = None
        try:
            context = await playwright.chromium.launch_persistent_context(
                str(self.profile_dir.resolve()), headless=False, accept_downloads=True,
                args=["--start-maximized"], viewport=None,
            )
            page = context.pages[0] if context.pages else await context.new_page()
            downloaded, search_url = await self._pubmed(page, query, limit, artifact_dir)
            try:
                await page.screenshot(path=str(artifact_dir / "completed.png"), full_page=True)
            except Exception:
                pass
            return downloaded, search_url
        except CollectionNeedsAttention:
            raise
        except Exception as exc:
            raise CollectionNeedsAttention(
                f"Browser stopped safely: {exc}. Finish/export manually, then upload the file."
            ) from exc
        finally:
            if context is not None:
                try:
                    await context.close()
                except Exception:
                    pass
            try:
                await playwright.stop()
            except Exception:
                pass

    @staticmethod
    async def _detect_blocker(page) -> None:
        body = (await page.locator("body").inner_text(timeout=10000)).lower()
        terms = ("captcha", "verify you are human", "access denied", "unusual traffic")
        if any(term in body for term in terms):
            raise CollectionNeedsAttention("Human verification or access control detected; no bypass was attempted.")

    async def _pubmed(self, page, query: str, limit: int, directory: Path):
        url = (
            "https://pubmed.ncbi.nlm.nih.gov/?term=" + quote_plus(query)
            + f"&size={limit}"
        )
        await page.goto(url, wait_until="domcontentloaded", timeout=90000)
        await self._detect_blocker(page)
        await page.locator("#save-results-panel-trigger").click(timeout=20000)
        selection = page.locator("#save-action-selection")
        if await selection.count():
            values = await selection.locator("option").evaluate_all("els => els.map(e => e.value)")
            choice = next(
                (value for value in values if value.lower() in {"page", "this-page"}),
                next((value for value in values if "selection" in value.lower()), values[-1]),
            )
            await selection.select_option(choice, force=True)
        fmt = page.locator("#save-action-format")
        await fmt.select_option("csv", force=True)
        submit = page.get_by_role("button", name="Create file")
        if await submit.is_disabled():
            options = await selection.locator("option").evaluate_all(
                "els => els.map(e => ({value:e.value, text:e.textContent.trim(), disabled:e.disabled}))"
            )
            raise CollectionNeedsAttention(
                "PubMed did not enable its native export after selection. "
                f"Available selection controls: {options}"
            )
        async with page.expect_download(timeout=60000) as info:
            await submit.click()
        download = await info.value
        target = directory / "pubmed.csv"
        await download.save_as(str(target))
        return target, page.url
