---
name: glyph-literature
description: Find source-backed literature markers, freeze their definitions and run matched studies comparing Glyph concepts with published baselines.
---

# Glyph literature studies

Search primary sources and freeze a cited marker manifest before evaluating
held-out outcomes. Read [the end-to-end pipeline](references/pipeline.md) for retrieval, optional
Paperclip credentials, source review, freezing, reconstruction and resumable execution.
Use [the comparison workflow](references/comparison-workflow.md) for numerical details. This is an agent harness, not a
method on the Glyph class. Frozen numerical studies need no live search or LLM.

Run `scripts/compare.py --cohort COHORT.csv --glyph CONCEPTS.csv --markers MARKERS.csv
--manifest MANIFEST.json --output NEW_DIR`. Optionally pass
`--glyph-associations SAVED_ROWS.csv` to reuse original Glyph effect estimates;
the matched study's testing family is still explicit. `--bootstrap 2000` controls
paired resampling. The ridge engine is adapted from the research implementation;
there are no private-repository or cohort dependencies at runtime.

Compare literature-only, Glyph-only and combined prediction arms on the exact same
eligible subjects, learning normalization on train and selecting ridge strength
on validation. Keep individual-marker association comparisons separate: the runner
uses equal marker counts and records its own correction family. Supply Glyph
selection and its training-only rule in the manifest; do not optimize them on test.

Inspect the actual outputs, including calibration, rather than presuming Glyph
wins. A review of a dataset's own historical markers is a worked benchmark,
not independent validation. Cite what a source supports; distinguish direct
measurements from proxies and report incomplete searches as such.

Start with `glyph harness literature-search --protocol search.json --output search_run`.
Paperclip is optional and reads `PAPERCLIP_API_KEY` or the protocol's named environment
variable. Review the retrieved packet as the agent; API retrieval does not establish
scientific eligibility. Fill the screening ledger, verify original sources, define
locally reconstructable markers, then use `literature-freeze` and `literature-study`.
No study/report stage publishes results. Keep credentials out of commands, packets
and model outputs. Do not send cohort rows to literature providers.

Optional comparative-study confounder adjustment is FMR orthogonal, depth 2, using
`--confounder-cols ...` and optionally `--adjustment-device cuda`; off by default.
It produces separate adjusted outputs and does not turn model performance or
unadjusted associations into causal evidence.
