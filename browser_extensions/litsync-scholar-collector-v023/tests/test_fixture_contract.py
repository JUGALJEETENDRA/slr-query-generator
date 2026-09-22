from pathlib import Path
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent


def soup(name: str):
    return BeautifulSoup((ROOT / "fixtures" / name).read_text(encoding="utf-8"), "html.parser")


def test_search_contract_matches_known_scholar_selectors():
    doc = soup("search_page.html")
    rows = doc.select(".gs_r[data-cid]")
    assert [row.get("data-cid") for row in rows] == ["cid001", "cid002", "cid003"]
    assert len(doc.select(".gs_or_sav")) == 2
    assert len(doc.select(".gs_or_sav_add")) == 1
    assert len(doc.select(".gs_or_lbl")) == 1
    assert doc.select_one("#gs_ab_md").get_text(strip=True).startswith("About 9,61,000")
    assert doc.select_one("#gs_n a").get("href").find("start=10") >= 0


def test_label_dialog_contract_matches_known_scholar_ids():
    doc = soup("label_dialog.html")
    assert doc.select_one("#gs_md_albl-d")
    assert doc.select_one("#gs_md_albl-d-bdy")
    assert doc.select_one("#gs_lbd_new")
    assert doc.select_one("#gs_lbd_new-input")
    assert doc.select_one("#gs_lbd_new_in .gs_in_cb")
    assert doc.select_one("#gs_lbd_apl").get_text(strip=True) == "Done"


def test_library_export_contract():
    doc = soup("library_page.html")
    assert any(a.get_text(strip=True) == "litsync_20260905_abcdef12" for a in doc.select("a"))
    assert any(a.get_text(strip=True) == "Export all" for a in doc.select("a,button"))
    assert any(a.get_text(strip=True) == "CSV" for a in doc.select("a,button"))


def test_real_final_page_shape_supports_405_accessible_results_with_stale_next():
    doc = soup("final_page_405.html")
    rows = doc.select(".gs_r[data-cid]")
    assert len(rows) == 5
    assert rows[-1].get("data-cid") == "cid405"
    assert "405 results" in doc.select_one("#gs_ab_md").get_text(" ", strip=True)
    assert "start=400" in doc.select_one("#gs_n a").get("href")


def test_export_articles_dialog_fixture_matches_live_final_step():
    html = (ROOT / "fixtures" / "export_articles_dialog.html").read_text(encoding="utf-8")
    assert "Export articles" in html
    assert "Export articles on this page" in html
    assert "Export all articles with this label" in html
    assert html.count('type="radio"') == 2
    assert ">EXPORT<" in html


def test_export_format_menu_contract_uses_scholar_csv_data_a_4():
    doc = soup("export_format_menu.html")
    csv = doc.select_one('[data-a="4"]')
    assert csv is not None
    assert csv.get_text(strip=True) == "CSV"
    assert doc.select_one('[data-a="0"]').get_text(strip=True) == "BibTeX"


def test_live_duplicate_controls_are_one_scholar_label():
    doc = soup('library_labels_live_20260905.html')
    links = doc.select('a')
    assert len(links) == 2
    assert {a.get_text() for a in links} == {'litsync_20260905_d2ee15f2_1412378...'}
    assert len({a['href'] for a in links}) == 1
    assert 'scilib=1030' in links[0]['href']
    assert links[0]['role'] == 'menuitemradio'
    assert links[1].parent.name == 'li'
