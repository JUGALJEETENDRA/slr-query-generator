"""Canonical, persisted PRISMA 2020 title/abstract-screening records."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from threading import Lock
from typing import Any


SCHEMA_VERSION = "prisma-2020-litsync-title-abstract-v1"
STANDARD = "PRISMA 2020"
SCOPE = "title_abstract_screening"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _integer(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _decision_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        name.lower(): sum(str(row.get("Decision", "")).upper() == name for row in rows)
        for name in ("KEEP", "MAYBE", "REJECT")
    }


class Prisma2020Manifest:
    """Thread-safe store whose snapshots are the only PRISMA count authority."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._contexts: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _directory(output_root: str | Path) -> Path:
        return Path(output_root) / "prisma"

    def _path(self, output_root: str | Path, workflow_id: str) -> Path:
        return self._directory(output_root) / f"{workflow_id}.json"

    def _write(self, context: dict[str, Any]) -> None:
        path = self._path(context["output_root"], context["workflow_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _get(self, workflow_id: str, output_root: str | Path | None = None) -> dict[str, Any]:
        context = self._contexts.get(str(workflow_id))
        if context is not None:
            return context
        if output_root is None:
            raise KeyError(workflow_id)
        path = self._path(output_root, str(workflow_id))
        context = json.loads(path.read_text(encoding="utf-8"))
        self._contexts[str(workflow_id)] = context
        return context

    def create_import(
        self,
        *,
        output_root: str | Path,
        import_id: str,
        records_identified: int,
        duplicate_records_removed: int,
        source_files: list[dict[str, Any]],
        clean_fingerprint: str,
        clean_path: str,
    ) -> dict[str, Any]:
        context = {
            "workflow_id": str(import_id), "job_id": None, "import_id": str(import_id),
            "output_root": str(output_root), "kind": "import", "screening_engine": None,
            "created_at": _now(), "updated_at": _now(), "status": "identification_complete",
            "identification": {
                "records_identified": _integer(records_identified),
                "duplicate_records_removed": _integer(duplicate_records_removed),
                "records_removed_other_reasons": 0,
                "records_available_for_screening": _integer(records_identified) - _integer(duplicate_records_removed),
                "deduplication_status": "performed",
                "source_files": source_files,
            },
            "screening_plan": {"records_selected": 0, "records_deferred_by_limit": 0},
            "clean_fingerprint": clean_fingerprint, "clean_path": clean_path,
            "csv_counts_match": None, "finalized": False,
        }
        with self._lock:
            self._contexts[str(import_id)] = context
            self._write(context)
        return self.snapshot(str(import_id))

    def begin_screening(
        self,
        *,
        output_root: str | Path,
        job_id: str,
        input_fingerprint: str,
        screening_engine: str,
        import_id: str | None = None,
        protocol_inputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        lineage = None
        warning = None
        with self._lock:
            if import_id:
                try:
                    candidate = self._get(import_id, output_root)
                    if candidate.get("clean_fingerprint") == input_fingerprint:
                        lineage = candidate
                    else:
                        warning = "The selected file did not match the deduplicated import; import lineage was not inherited."
                except (OSError, KeyError, ValueError, json.JSONDecodeError):
                    warning = "The requested import lineage was unavailable; deduplication is reported as not performed."
            identification = dict((lineage or {}).get("identification") or {
                "records_identified": 0,
                "duplicate_records_removed": None,
                "records_removed_other_reasons": 0,
                "records_available_for_screening": 0,
                "deduplication_status": "not_performed",
                "source_files": [],
            })
            context = {
                "workflow_id": str(job_id), "job_id": str(job_id),
                "import_id": lineage.get("import_id") if lineage else None,
                "output_root": str(output_root), "kind": "screening",
                "screening_engine": screening_engine, "created_at": _now(), "updated_at": _now(),
                "status": "starting", "identification": identification,
                "screening_plan": {"records_selected": 0, "records_deferred_by_limit": 0},
                "input_fingerprint": input_fingerprint,
                "protocol_inputs": dict(protocol_inputs or {}),
                "lineage_warning": warning, "csv_counts_match": None, "finalized": False,
            }
            self._contexts[str(job_id)] = context
            self._write(context)
        return self.snapshot(str(job_id))

    def configure_screening(
        self,
        workflow_id: str,
        *,
        input_rows: int,
        missing_abstracts: int,
        records_available: int,
        records_selected: int,
    ) -> None:
        with self._lock:
            context = self._get(workflow_id)
            identification = context["identification"]
            if identification.get("deduplication_status") != "performed":
                identification["records_identified"] = _integer(input_rows)
            identification["records_available_for_screening"] = _integer(records_available)
            identification["records_removed_other_reasons"] = _integer(missing_abstracts)
            context["screening_plan"] = {
                "records_selected": _integer(records_selected),
                "records_deferred_by_limit": max(0, _integer(records_available) - _integer(records_selected)),
            }
            context["status"] = "screening"
            context["updated_at"] = _now()
            self._write(context)

    def mark_finalized(self, workflow_id: str, csv_counts_match: bool) -> None:
        with self._lock:
            context = self._get(workflow_id)
            context["finalized"] = True
            context["csv_counts_match"] = bool(csv_counts_match)
            available = _integer(context.get("identification", {}).get("records_available_for_screening"))
            selected = _integer(context.get("screening_plan", {}).get("records_selected"))
            complete = bool(csv_counts_match) and selected >= available
            context["status"] = "title_abstract_complete" if complete else "incomplete"
            context["updated_at"] = _now()
            self._write(context)

    def snapshot(
        self,
        workflow_id: str,
        *,
        output_root: str | Path | None = None,
        progress: dict[str, Any] | None = None,
        rows: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            context = dict(self._get(workflow_id, output_root))
        identification = dict(context["identification"])
        plan = dict(context.get("screening_plan") or {})
        rows = list(rows or [])
        live_progress = progress if progress and all(
            key in progress for key in ("keep", "maybe", "reject")
        ) else None
        stored_state = dict(context.get("screening_state") or {})
        has_live_state = bool(rows) or live_progress is not None
        if rows:
            counts = _decision_counts(rows)
        elif live_progress is not None:
            counts = {name: _integer(live_progress.get(name)) for name in ("keep", "maybe", "reject")}
        else:
            counts = {
                name: _integer(stored_state.get("counts", {}).get(name))
                for name in ("keep", "maybe", "reject")
            }
        screened = sum(counts.values())
        selected = _integer(plan.get("records_selected"))
        available_for_screening = _integer(identification.get("records_available_for_screening"))
        awaiting_screening = max(0, available_for_screening - screened)
        automated_rejects = 0
        manual_rejects = 0
        reasons: dict[str, int] = {}
        for row in rows:
            if str(row.get("Decision", "")).upper() != "REJECT":
                continue
            if str(row.get("Decision_Source", "")).lower() == "manual_review":
                manual_rejects += 1
            else:
                automated_rejects += 1
            reason = str(row.get("Exclusion_Reason") or "reason not classified").strip()
            reasons[reason] = reasons.get(reason, 0) + 1
        if not rows and live_progress is not None:
            automated_rejects = counts["reject"]
            if counts["reject"]:
                reasons["reason not classified"] = counts["reject"]
        elif not rows:
            automated_rejects = _integer(stored_state.get("automated_rejects"))
            manual_rejects = _integer(stored_state.get("manual_rejects"))
            reasons = {
                str(reason): _integer(count)
                for reason, count in (stored_state.get("reasons") or {}).items()
            }
        warnings = [
            "Title/abstract screening only; full-text eligibility and final study inclusion were not performed in LitSync."
        ]
        if identification.get("deduplication_status") != "performed":
            warnings.append("Deduplication was not performed in the linked LitSync import workflow.")
        if plan.get("records_deferred_by_limit"):
            warnings.append("A row limit left records unscreened in this run.")
        if counts["maybe"]:
            warnings.append("MAYBE records still require manual review.")
        if context.get("kind") == "screening" and awaiting_screening:
            warnings.append("Some available records have not been screened in this workflow.")
        if context.get("lineage_warning"):
            warnings.append(context["lineage_warning"])
        if context.get("csv_counts_match") is None:
            warnings.append("Final CSV consistency has not yet been verified.")
        identification_equation = (
            _integer(identification.get("records_identified"))
            == _integer(identification.get("duplicate_records_removed"))
            + _integer(identification.get("records_removed_other_reasons"))
            + _integer(identification.get("records_available_for_screening"))
        ) if identification.get("duplicate_records_removed") is not None else True
        screening_equation = (
            screened == counts["keep"] + counts["maybe"] + counts["reject"]
            and available_for_screening == screened + awaiting_screening
            and screened <= selected
            and selected == screened + max(0, selected - screened)
        )
        status = context.get("status", "incomplete")
        if context.get("kind") == "screening" and not context.get("finalized"):
            progress_status = str((progress or {}).get("status") or "")
            if progress_status == "error":
                status = "error"
            elif progress_status == "starting" or (
                context.get("status") == "starting" and selected == 0
            ):
                status = "starting"
            else:
                status = "manual_review" if screened and counts["maybe"] else (
                    "screening" if screened < selected else (
                        "incomplete" if awaiting_screening else "title_abstract_complete"
                    )
                )
        manifest = {
            "schema_version": SCHEMA_VERSION, "standard": STANDARD, "scope": SCOPE,
            "workflow_id": context["workflow_id"], "job_id": context.get("job_id"),
            "import_id": context.get("import_id"), "status": status,
            "screening_engine": context.get("screening_engine"),
            "protocol_inputs": dict(context.get("protocol_inputs") or {}),
            "updated_at": context.get("last_snapshot_updated_at", context.get("updated_at", _now())),
            "identification": identification,
            "screening": {
                "records_selected_for_run": selected,
                "records_screened": screened,
                "records_awaiting_screening": awaiting_screening,
                "records_awaiting_current_run": max(0, selected - screened),
                "records_deferred_by_limit": _integer(plan.get("records_deferred_by_limit")),
                "records_excluded": counts["reject"],
                "excluded_by_tool_assisted_screening": automated_rejects,
                "excluded_by_manual_review": manual_rejects,
                "records_awaiting_manual_review": counts["maybe"],
                "records_included_after_title_abstract": counts["keep"],
                "exclusion_reasons": [
                    {"reason": reason, "count": count} for reason, count in sorted(reasons.items())
                ],
            },
            "full_text_stage_status": "not_performed",
            "reports_sought_for_retrieval": None,
            "reports_assessed_for_eligibility": None,
            "studies_included_in_review": None,
            "reports_of_included_studies": None,
            "integrity": {
                "equations_valid": identification_equation and screening_equation,
                "csv_counts_match": context.get("csv_counts_match"),
                "warnings": warnings,
            },
        }
        state_payload = json.dumps(
            {key: value for key, value in manifest.items() if key != "updated_at"},
            sort_keys=True,
            ensure_ascii=False,
        )
        revision = hashlib.sha256(state_payload.encode("utf-8")).hexdigest()[:16]
        with self._lock:
            live_context = self._get(workflow_id, output_root)
            state_changed = False
            if has_live_state:
                screening_state = {
                    "counts": counts,
                    "automated_rejects": automated_rejects,
                    "manual_rejects": manual_rejects,
                    "reasons": reasons,
                }
                state_changed = live_context.get("screening_state") != screening_state
                live_context["screening_state"] = screening_state
            if live_context.get("last_snapshot_signature") != revision:
                live_context["last_snapshot_signature"] = revision
                live_context["last_snapshot_updated_at"] = _now()
                state_changed = True
            if state_changed:
                self._write(live_context)
            manifest["updated_at"] = live_context["last_snapshot_updated_at"]
        manifest["revision"] = revision
        return manifest


PRISMA_STORE = Prisma2020Manifest()


def manifest_csv(manifest: dict[str, Any]) -> str:
    rows: list[tuple[str, Any]] = [
        ("schema_version", manifest.get("schema_version")),
        ("standard", manifest.get("standard")),
        ("scope", manifest.get("scope")),
        ("workflow_id", manifest.get("workflow_id")),
        ("job_id", manifest.get("job_id")),
        ("updated_at", manifest.get("updated_at")),
    ]
    for section in ("identification", "screening"):
        for key, value in manifest.get(section, {}).items():
            rows.append((
                f"{section}.{key}",
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (list, dict)) else value,
            ))
    rows.extend([
        ("status", manifest.get("status")),
        ("full_text_stage_status", manifest.get("full_text_stage_status")),
        ("integrity.equations_valid", manifest.get("integrity", {}).get("equations_valid")),
        ("integrity.csv_counts_match", manifest.get("integrity", {}).get("csv_counts_match")),
    ])
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["field", "value"])
    writer.writerows(rows)
    return stream.getvalue()


def manifest_svg(manifest: dict[str, Any]) -> str:
    ident = manifest["identification"]
    screen = manifest["screening"]
    updated = escape(str(manifest.get("updated_at") or ""))
    engine = escape(str(manifest.get("screening_engine") or "not started"))
    duplicates = ident.get("duplicate_records_removed")
    duplicate_text = "Not performed" if duplicates is None else f"n = {_integer(duplicates)}"
    reasons = screen.get("exclusion_reasons") or []
    reason_lines = reasons[:3] or [{"reason": "reason not classified", "count": screen["records_excluded"]}]
    reason_svg = "".join(
        f'<text x="635" y="{548 + index * 18}" class="small">{escape(str(item["reason"]))}: n = {_integer(item["count"])}</text>'
        for index, item in enumerate(reason_lines)
    )
    warning = escape("Title/abstract screening only; full-text eligibility and final study inclusion were not performed in LitSync.")
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="980" height="900" viewBox="0 0 980 900" role="img" aria-labelledby="title desc">
<title id="title">PRISMA 2020 flow of records</title><desc id="desc">Live LitSync title and abstract screening flow diagram.</desc>
<style>.box{{fill:#fff;stroke:#1f2933;stroke-width:1.5}}.side{{fill:#fafafa;stroke:#52606d;stroke-width:1.2}}.stage{{font:700 15px Arial;letter-spacing:1px}}.label{{font:600 15px Arial;fill:#111827}}.count{{font:700 18px Arial;fill:#111827}}.small{{font:13px Arial;fill:#374151}}.note{{font:12px Arial;fill:#4b5563}}.arrow{{stroke:#374151;stroke-width:1.5;fill:none;marker-end:url(#a)}}</style>
<defs><marker id="a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#374151"/></marker></defs>
<text x="490" y="35" text-anchor="middle" class="stage">PRISMA 2020 FLOW DIAGRAM</text>
<text x="490" y="58" text-anchor="middle" class="small">LitSync title/abstract screening record · {engine}</text>
<text x="30" y="150" class="stage" transform="rotate(-90 30 150)">IDENTIFICATION</text>
<rect x="100" y="90" width="430" height="105" rx="3" class="box"/><text x="315" y="125" text-anchor="middle" class="label">Records identified from databases/registers</text><text x="315" y="160" text-anchor="middle" class="count">n = {_integer(ident.get("records_identified"))}</text>
<rect x="610" y="90" width="300" height="135" rx="3" class="side"/><text x="760" y="120" text-anchor="middle" class="label">Records removed before screening</text><text x="635" y="153" class="small">Duplicate records removed: {duplicate_text}</text><text x="635" y="177" class="small">Other recorded reasons: n = {_integer(ident.get("records_removed_other_reasons"))}</text><text x="635" y="201" class="small">Pre-screen automation removal: not used</text>
<path d="M530 142H610" class="arrow"/><path d="M315 195V285" class="arrow"/>
<text x="30" y="400" class="stage" transform="rotate(-90 30 400)">SCREENING</text>
<rect x="100" y="285" width="430" height="105" rx="3" class="box"/><text x="315" y="320" text-anchor="middle" class="label">Records screened by title and abstract</text><text x="315" y="355" text-anchor="middle" class="count">n = {_integer(screen.get("records_screened"))}</text>
<rect x="610" y="285" width="300" height="115" rx="3" class="side"/><text x="760" y="315" text-anchor="middle" class="label">Records awaiting screening</text><text x="760" y="345" text-anchor="middle" class="count">n = {_integer(screen.get("records_awaiting_screening"))}</text><text x="635" y="374" class="small">Deferred by row limit: n = {_integer(screen.get("records_deferred_by_limit"))}</text><path d="M530 337H610" class="arrow"/>
<path d="M315 390V460" class="arrow"/><rect x="100" y="460" width="430" height="110" rx="3" class="box"/><text x="315" y="493" text-anchor="middle" class="label">Records retained after screening</text><text x="315" y="525" text-anchor="middle" class="small">KEEP: n = {_integer(screen.get("records_included_after_title_abstract"))} · MAYBE awaiting manual review: n = {_integer(screen.get("records_awaiting_manual_review"))}</text>
<rect x="610" y="430" width="300" height="160" rx="3" class="side"/><text x="760" y="458" text-anchor="middle" class="label">Records excluded</text><text x="760" y="486" text-anchor="middle" class="count">n = {_integer(screen.get("records_excluded"))}</text><text x="635" y="512" class="small">Tool-assisted: n = {_integer(screen.get("excluded_by_tool_assisted_screening"))} Â· Manual: n = {_integer(screen.get("excluded_by_manual_review"))}</text><text x="635" y="532" class="small">Explicit reasons (otherwise unclassified):</text>{reason_svg}<path d="M530 505H610" class="arrow"/>
<path d="M315 570V675" class="arrow"/><text x="30" y="745" class="stage" transform="rotate(-90 30 745)">INCLUDED</text>
<rect x="100" y="675" width="430" height="120" rx="3" class="box"/><text x="315" y="712" text-anchor="middle" class="label">Records included after title/abstract screening</text><text x="315" y="750" text-anchor="middle" class="count">n = {_integer(screen.get("records_included_after_title_abstract"))}</text><text x="315" y="778" text-anchor="middle" class="small">Provisional; not final study inclusion</text>
<text x="100" y="835" class="note">{warning}</text><text x="100" y="858" class="note">Last updated: {updated}</text><text x="100" y="880" class="note">Adapted from PRISMA 2020 flow diagram, CC BY 4.0.</text></svg>'''
