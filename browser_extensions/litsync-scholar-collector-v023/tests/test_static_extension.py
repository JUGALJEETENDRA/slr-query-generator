import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_is_mv3_and_narrowly_scoped():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == 3
    assert set(manifest["permissions"]) == {"storage", "tabs", "downloads"}
    assert "https://scholar.google.com/*" in manifest["host_permissions"]
    assert "https://scholar.googleusercontent.com/*" in manifest["host_permissions"]
    assert "<all_urls>" not in manifest["host_permissions"]


def test_no_captcha_bypass_or_scholar_bib_path():
    source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src").glob("*.js"))
    assert "scholar.bib" not in source
    assert "stealth" not in source.lower()
    assert "captcha solver" not in source.lower()
    assert "unusual traffic" in source.lower()


def test_collection_cap_is_1000():
    source = (ROOT / "src" / "core.js").read_text(encoding="utf-8")
    assert "const RESULT_CAP = 1000" in source


def test_save_label_export_selectors_are_present():
    source = (ROOT / "src" / "scholar.js").read_text(encoding="utf-8")
    for selector in [".gs_r[data-cid]", ".gs_or_sav", ".gs_or_lbl", "#gs_md_albl-d", "#gs_lbd_new-input", "#gs_lbd_apl"]:
        assert selector in source
    assert '"Export all"' in source
    assert 'findCsvFormatControl' in source
    assert 'controlLabel' in source
    assert '"Export all articles with this label"' in source
    assert "completeExportArticlesDialog" in source
    assert "v0.1.3 marked EXPORTING" in source
    assert "No CSV download started" in source
    assert "findVisibleByText" in source
    assert ".gs_in_ra" in source
    assert "associatedRadio" in source
    assert "findCsvFormatControl" in source
    assert 'cit_fmt=4' not in source
    assert 'searchParams.set("cit_fmt"' not in source
    assert "recoverNonCsvExportPage" in source
    assert "looksLikeExportArticlesDialog" in source


def test_failure_isolation_end_reconciliation_and_throttle_circuit_breaker_are_present():
    source = (ROOT / "src" / "scholar.js").read_text(encoding="utf-8")
    core = (ROOT / "src" / "core.js").read_text(encoding="utf-8")
    assert "paper_skipped_after_retries" in source
    assert "RETRYING_FAILURES" in source
    assert "paper_recovered_on_final_retry" in source
    assert "paper_final_retry_failed" in source
    assert "actual_end_of_results" in source
    assert "applyThrottleCooldown" in source
    assert "confirmedEndOfResults" in source
    assert "finalAccessibleTarget" in source
    assert "throttleCooldownMs" in core
    assert 'INCOMPLETE: "incomplete"' in core


def test_final_hardening_version_is_026():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "0.2.6"


def test_label_integrity_audit_and_repair_are_mandatory_before_full_export():
    source = (ROOT / "src" / "scholar.js").read_text(encoding="utf-8")
    core = (ROOT / "src" / "core.js").read_text(encoding="utf-8")
    background = (ROOT / "src" / "background.js").read_text(encoding="utf-8")
    popup = (ROOT / "popup" / "popup.js").read_text(encoding="utf-8")
    for token in ["VERIFYING_LABEL", "REPAIRING_LABEL", "auditRunLabel", "repairLabelMembership", "label_integrity_mismatch", "label_integrity_pass"]:
        assert token in source or token in core
    assert "labelIntegritySatisfied" in core
    assert "verified-in-label" in background
    assert "fullyVerifiedComplete" in popup
    assert "Refusing to export" in source


def test_csv_content_and_row_count_validation_gate_completion():
    core = (ROOT / "src" / "core.js").read_text(encoding="utf-8")
    background = (ROOT / "src" / "background.js").read_text(encoding="utf-8")
    popup = (ROOT / "popup" / "popup.js").read_text(encoding="utf-8")
    assert "inspectScholarCsv" in core
    assert "fullExportIntegritySatisfied" in core
    assert "verifyDownloadedCsvFile" in background
    assert "exportCsvVerified" in background
    assert "exportCsvRowCount" in background
    assert "LSSC_RENDERED_CSV" in background
    assert "fullExportIntegritySatisfied" in popup
    assert "cit_fmt=4" not in background
    assert "LSSC_VERIFY_EXPORT_REQUEST" in background
    assert "file:///*" in json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))["host_permissions"]
    assert "verifyScholarCsvRequest" in background
    assert "exportRequestFromControl" in (ROOT / "src" / "scholar.js").read_text(encoding="utf-8")


def test_v020_uses_data_lid_to_bridge_search_and_library_id_namespaces():
    core = (ROOT / "src" / "core.js").read_text(encoding="utf-8")
    scholar = (ROOT / "src" / "scholar.js").read_text(encoding="utf-8")
    assert "libraryIdsBySearchId" in core
    assert "expectedLibraryIds" in core
    assert 'getAttribute("data-lid")' in scholar
    assert "INDEXING_LIBRARY_IDS" in scholar
