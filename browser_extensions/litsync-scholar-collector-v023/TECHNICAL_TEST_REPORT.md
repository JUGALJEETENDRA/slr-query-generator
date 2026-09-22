# v0.2.4 Technical Test Report

## Scope

Narrow fix for Google Scholar sidebar truncation of v0.2.3 per-run labels. No collection/export algorithm changes.

## Regression reproduced

Live shape from the browser:
- Scholar total/target: 58
- Saved: 58
- Unresolved: 0
- label visibly present in My Library
- run label: `litsync_20260905_d2ee15f2_141237814`
- sidebar rendered the tail with `...`
- v0.2.3 exact-text lookup incorrectly reported the label as not visible.

## Fixes

- Added an unambiguous truncated-prefix fallback for legacy long labels.
- Exact text/title/ARIA matches remain preferred.
- Ambiguous truncated prefixes are rejected.
- New per-run labels are capped at 26 characters using `litsync_YYMMDD_FFFF_TTTTTT`.
- Strict label ID-set verification remains mandatory before any full export.

## Test coverage

- Legacy long-label truncation recovery.
- Ambiguous truncated-prefix refusal.
- Fresh-run labels remain unique and start from zero.
- Compact-label maximum length.
- All prior collection, pagination, retry, label-integrity, and CSV verification tests retained.
