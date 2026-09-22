# v0.2.5 targeted legacy-label navigation fix

Scope: only My Library run-label discovery was changed from v0.2.4, plus tests and manifest version.

The finder now canonicalizes sidebar label text by removing display whitespace, zero-width characters, and trailing ellipsis, then permits only a unique sufficiently-long prefix match for legacy v0.2.3 labels. This covers Scholar DOM clipping/wrapping without relaxing the downstream strict ID-set audit.
