# LitSync Scholar Collector v0.2.4

This is a narrow label-navigation hardening update over v0.2.3.

## Why this update exists

v0.2.3 correctly created a new label for every fresh run, but its label name became long enough for Google Scholar's **My Library** sidebar to render the tail as `...`. The collector then looked for the full label by exact visible text and could falsely report that the label was missing even though it was visibly present.

## What changed

1. **Existing v0.2.3 runs are recoverable.** The My Library label resolver first uses exact visible text/title/ARIA text. If Scholar renders a literal trailing ellipsis, it accepts the truncated link only when exactly one sufficiently-long prefix matches the current run label. The normal strict ID-set audit still runs before export, so a wrong label can never silently pass.
2. **Future run labels are shorter.** New labels use the form `litsync_YYMMDD_FFFF_TTTTTT`, where the final token is milliseconds since UTC midnight encoded in base36. They remain unique for normal human-started runs while staying compact enough for the Scholar sidebar.

Example: `litsync_260905_a6b3_0s7y5r`.

## Unchanged

Collection, pagination, save/label behavior, throttling, retry handling, result-count reconciliation, data-cid/data-lid mapping, strict label integrity audit/repair, native CSV export, local downloaded-file CSV verification, LitSync query sync, popup behavior, and completion rules are unchanged.

## In-place update for a live v0.2.3 checkpoint

To preserve an existing run, overwrite the files in the already-loaded v0.2.3 extension directory and click **Reload** in `edge://extensions`. Do not remove the extension and do not Reset the run.
