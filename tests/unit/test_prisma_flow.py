from __future__ import annotations

from litsync_app.prisma import Prisma2020Manifest, manifest_csv, manifest_svg


def _rows(keep: int, maybe: int, reject: int):
    rows = []
    for decision, count in (("KEEP", keep), ("MAYBE", maybe), ("REJECT", reject)):
        for index in range(count):
            rows.append({
                "Title": f"{decision} paper {index}",
                "Decision": decision,
                "Decision_Source": "tool_assisted_screening",
                "Exclusion_Reason": "",
            })
    return rows


def test_mocked_ten_record_job_has_live_conservative_counts_and_exports(tmp_path):
    store = Prisma2020Manifest()
    store.create_import(
        output_root=tmp_path,
        import_id="import-12",
        records_identified=12,
        duplicate_records_removed=2,
        source_files=[{"name": "database.csv", "records": 12}],
        clean_fingerprint="clean-fingerprint",
        clean_path=str(tmp_path / "clean.csv"),
    )
    store.begin_screening(
        output_root=tmp_path,
        job_id="job-10",
        input_fingerprint="clean-fingerprint",
        screening_engine="local",
        import_id="import-12",
    )
    store.configure_screening(
        "job-10", input_rows=10, missing_abstracts=0,
        records_available=10, records_selected=10,
    )

    rows = _rows(3, 2, 5)
    live = store.snapshot("job-10", progress={"status": "finished"}, rows=rows)
    assert live["identification"]["records_identified"] == 12
    assert live["identification"]["duplicate_records_removed"] == 2
    assert live["screening"]["records_screened"] == 10
    assert live["screening"]["records_excluded"] == 5
    assert live["screening"]["records_awaiting_manual_review"] == 2
    assert live["screening"]["records_included_after_title_abstract"] == 3
    assert live["integrity"]["equations_valid"] is True

    unchanged = store.snapshot("job-10", progress={"status": "finished"}, rows=rows)
    assert unchanged["revision"] == live["revision"]
    assert unchanged["updated_at"] == live["updated_at"]

    rows[3].update(Decision="KEEP", Decision_Source="manual_review")
    rows[4].update(
        Decision="REJECT", Decision_Source="manual_review",
        Exclusion_Reason="Does not meet the protocol",
    )
    reviewed = store.snapshot("job-10", progress={"status": "finished"}, rows=rows)
    assert reviewed["revision"] != live["revision"]
    assert reviewed["screening"]["records_included_after_title_abstract"] == 4
    assert reviewed["screening"]["records_excluded"] == 6
    assert reviewed["screening"]["records_awaiting_manual_review"] == 0
    assert reviewed["screening"]["excluded_by_manual_review"] == 1
    assert {item["reason"] for item in reviewed["screening"]["exclusion_reasons"]} == {
        "Does not meet the protocol", "reason not classified"
    }

    store.mark_finalized("job-10", csv_counts_match=True)
    final = store.snapshot("job-10", rows=rows)
    assert final["status"] == "title_abstract_complete"
    assert final["integrity"]["csv_counts_match"] is True
    assert not any("MAYBE records" in warning for warning in final["integrity"]["warnings"])

    csv_text = manifest_csv(final)
    svg = manifest_svg(final)
    assert "screening.records_excluded,6" in csv_text
    assert "screening.records_included_after_title_abstract,4" in csv_text
    assert "PRISMA 2020" in svg
    assert "Records identified from databases/registers" in svg
    assert "Records included after title/abstract screening" in svg
    assert "Provisional; not final study inclusion" in svg
    assert "Reports sought for retrieval" not in svg
    assert "Reports assessed for eligibility" not in svg


def test_checkpoint_reload_and_incomplete_input_warnings_are_truthful(tmp_path):
    store = Prisma2020Manifest()
    store.begin_screening(
        output_root=tmp_path,
        job_id="limited-job",
        input_fingerprint="raw-upload",
        screening_engine="gemini_web_v24",
    )
    store.configure_screening(
        "limited-job", input_rows=12, missing_abstracts=2,
        records_available=10, records_selected=4,
    )
    rows = _rows(1, 1, 1)
    partial = store.snapshot(
        "limited-job",
        progress={"status": "running", "keep": 1, "maybe": 1, "reject": 1},
        rows=rows,
    )
    assert partial["identification"]["deduplication_status"] == "not_performed"
    assert partial["identification"]["records_removed_other_reasons"] == 2
    assert partial["screening"]["records_screened"] == 3
    assert partial["screening"]["records_awaiting_screening"] == 7
    assert partial["screening"]["records_awaiting_current_run"] == 1
    assert partial["screening"]["records_deferred_by_limit"] == 6
    assert partial["screening"]["records_excluded"] == 1
    assert partial["screening"]["records_awaiting_manual_review"] == 1
    assert any("Deduplication was not performed" in warning for warning in partial["integrity"]["warnings"])
    assert any("row limit" in warning for warning in partial["integrity"]["warnings"])

    restored_store = Prisma2020Manifest()
    restored = restored_store.snapshot(
        "limited-job", output_root=tmp_path,
    )
    assert restored["identification"] == partial["identification"]
    assert restored["screening"] == partial["screening"]
    assert restored["revision"] == partial["revision"]
