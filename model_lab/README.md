# LitSync Local Model Evaluation Lab

This is a separate, local-only workspace for testing LitSync screening models without changing the production website or screening pipeline.

## Start it

```powershell
python -m model_lab
```

Open `http://127.0.0.1:8010`.

## What it evaluates

- One shared local-AI review protocol for a research question and optional criteria.
- Up to four local Ollama models on identical title/abstract papers.
- Optional independently prompted local critic for unsafe primary outcomes.
- Structured-output validity, exact evidence-unit resolution, decision distribution, latency, token counts, tokens/second, critic calls, and model-to-model disagreements.
- Optional blinded human labels (`Gold_Decision`: `KEEP`, `REJECT`, or `UNSURE`) from a CSV or manual entry. The lab reports retrieval recall, false-REJECT rate, and definitive-KEEP precision only against those labels; `UNSURE` never inflates a quality score.
- Full paper-level traces and saved JSON run artifacts under `model_lab/runs/`.

It stores concise evidence and rationales only; it never requests or stores hidden chain-of-thought. It has no Gemini, cloud API, or production-screening route.

## Suggested first experiment

Use 5–15 papers that include obvious KEEP, obvious REJECT, borderline, and title-only cases. Select two installed local models, keep the critic off for the baseline, then rerun the same set with a critic to see what changes.

For a serious comparison, preserve a human decision in a `Gold_Decision` CSV column before running the models. The evaluator keeps those labels separate from the prompt so a candidate model cannot see or learn from them.
