# LitSync human gold set

Create a stratified, dual-reviewed gold set covering the heart-disease, blockchain,
SLR-automation, and digital-twin corpora. Copy `gold_set_template.csv`, retain the source
row identifier and title, and fill `Gold_Decision` with `KEEP` or `REJECT`. Human reviewers
must adjudicate disagreements; generated model decisions must never be used as gold labels.

The release evaluator enforces 95% relevant-paper retrieval recall, at most 5% false rejects,
85% definitive-KEEP precision, exact evidence quotes, and zero invalid definitive decisions.
