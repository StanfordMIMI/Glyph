# Local release readiness

## Private review preparation — 26 October 2026

The user requested preparation of a private GitHub review push and will run the
final push themselves. Public release, model/data uploads, and Git-history
rewriting remain deferred. The single recommended demo is now
[DINOv3 on public digit images](../demos/dinov3_digits/README.md); older demo
artifacts remain local and are excluded from this review commit.

A fresh source regression check passed 135 tests with one opt-in test skipped.
A final report/skill check passed 17 tests after correcting the legend entry for
significant outcomes beyond the first five.
The demo uses actual official DINOv3 weights and separately executes extraction,
SAE fitting, association testing, interactive HTML, exact saved-run reload and
the report harness. See its README and provenance for the completed run results.
No private research cohort was used.


Updated 23 October 2026. This is a local preparation record. Nothing was published,
pushed, uploaded or rewritten as part of this work.

## Implemented

The wheel includes four discoverable agent skills and the executable retrieval,
review-freeze, comparison, interpretation and report harnesses. `glyph harness list`
and `glyph skills list` show these without importing the model stack. Literature
retrieval uses PubMed directly and optionally a user-provided GXL Paperclip API key
from an environment variable. Screening decisions and source verification precede
freezing markers. Search counts distinguish total hits, retrieved records,
deduplicated records, supplementary results and retrieval truncation.

The study harness reconstructs explicit formulas, compares literature-only,
Glyph-only and combined models, and saves train preprocessing, validation tuning,
held-out predictions, paired bootstrap comparisons and equal-count individual
associations. Explicit confounder columns optionally enable separate depth-2 FMR
orthogonal adjusted associations. Adjustment is off by default; CPU is the default,
with `--adjustment-device cuda` for a CUDA installation. These are association
estimates using the updated orthogonal score; they do not establish causal effects.
The estimator was updated on 23 October; see [numerical verification](confounder_adjustment.md).

Interpretation is optional, supports Gemini environment credentials or Vertex,
and scores complete predictions on balanced held-out examples. Incomplete model
responses remain unscored. Repeated concepts are interpreted once per modality.
Reports can read those outputs alongside the complete alive concept space,
corrected associations, search funnel, citations and numerical comparison results.
The report and HTML explorer remain separate optional outputs.

A local release preparer exports exact saved activations, numeric SAE parameters,
normalization, original concept indices and checksums. It requires an explicit
one-to-one mapping to public sample identifiers. Only synthetic export fixtures
were used. No research cohort was exported.

## Existing public models

Anonymous requests and loading succeeded for all three SAE bundles from
[`RobbieHolland/HypothesisExtractor`](https://huggingface.co/RobbieHolland/HypothesisExtractor)
at `23c5da2d6d415e88255912da398e83035d507069`. Each has 8,192 concepts and k=40.
Input dimensions are 2,048 for CT, 4,096 for findings and 768 for labs.
The pinned files include normalization and interpretations. They contain SAE
encoders only, so they support concept inference but cannot measure the published
dictionaries' reconstruction R². That is reported as unavailable rather than
fabricated or evaluated with an unrelated decoder.

The labs encoder and reviewed LOINC mappings are anonymously accessible from
[`RobbieHolland/Glyph-labs-encoder`](https://huggingface.co/RobbieHolland/Glyph-labs-encoder)
at `972acbaea6d7ea2249a84d562a5f6dbabb994827`. This pinned bundle has an MIT license
and a model card. The SAE repository's pinned revision lacks both a model card and
license metadata. A model-card draft is in `release/hf_saes/`; it was not uploaded.
Historical Merlin and Qwen encoder revisions still need to be established from
training records; current upstream revisions are not substitutes for provenance.

Anonymous GitHub requests returned 404 for both `RobbieHolland/Glyph` and
`RobbieHolland/Glyph-research`. They are not publicly readable at the time of
checking; this does not distinguish private repositories from absent ones.
Publication and public URL checks remain deferred.

## Paper version 26

The PDF's text and embedded hyperlinks were checked. The package repository is
linked, and Code and Data Availability on PDF page 26 points to
`github.com/RobbieHolland/Glyph-research`. Neither Hugging Face asset above is
linked, and no per-study activation download is given. Add the actual asset links
when the final release locations and scope are agreed.

The Open source section on PDF page 27 still describes selection by lowest
reconstruction error. The package now uses the agreed greedy sparsity rule:
start at validation R² >= 0.85; allow another 100% of the first qualifying k for
0.90 and another 100% for 0.95; warn if 0.85 is not reached. Align this wording
before release. Published encoder-only dictionaries also cannot supply the same
reconstruction check until their original decoders are provided.

The fitting API expects metadata splits named `train`, `val` and `test` (or
creates those splits when no split column is provided). Unsupported or missing
labels in an explicit split column now raise an error instead of silently dropping
observations. The separate literature comparison harness can evaluate multiple
named held-out splits, as described in its input contract.

## Verification and remaining inputs

Local source checks passed 176 tests; a later focused check passed 31 tests after
the latest interpretation, workflow and skill changes. Search retrieval was also
exercised against live PubMed and replayed offline. All three modified skills
passed the skill-format validator. A further 41 tests passed after the
metadata-split regression fix, covering saved runs, extraction and comparison.
The fresh wheel was built and installed into
an isolated test directory. A real installed-wheel synthetic workflow passed:
SAE fit, exact activation reload, PheWAS, explorer HTML, reviewed-fixture marker
freezing, three-arm prediction, eight successful depth-2 CPU-adjusted estimates,
a full-space two-page PDF and repeat execution without rewriting the report.
All 12 alive concepts appeared in the report. Installed PubMed retrieval also
replayed its live cache offline. This was a software integration check, not a
clinical demo or a literature rediscovery. Detailed logs and verification JSON
are under `test_outputs/release-work/`.

No Paperclip or Gemini/Vertex credentials were configured, so authenticated live
calls were not exercised. Controlled transport/scoring tests cover authentication
headers, secret handling, invalid responses and optional configuration. A user's
key is still needed to verify their live account, quota and model access.

Remaining release inputs are the intended SAE license, original SAE decoders if
reconstruction validation is to be supported, historical encoder provenance, and
public sample-ID mapping plus the activation release scope. Source/data
publication, public access checks after publication, final version identifiers and
any eventual Git-history reset are deferred until explicitly authorized.


## Reader API follow-up

`Glyph.from_pretrained`, `measure(code, metadata=..., sample_id=...)`, whole-space
`measure(metadata=...)`, and subsequent `visualize()` are now implemented. Full
results retain all 8,192 paper code columns; plots display concepts active in the
reader's cohort. New measurement runs do not require labels or train/test splits.
See [reader verification](reader_verification.md) for real-weight tests and the
completed real DINOv3 test. The README now follows the manuscript's
examples, uses `Glyph.svg`, and omits the obsolete MechSci paper/video links.


## Full-width public image demos

Both DINOv3 demos now learn 8,192-entry dictionaries. The digits run selected
k=64 (validation R² 0.868); the public TCIA chest X-ray mortality run selected
k=768 (validation R² 0.905, test R² 0.902). The medical example uses 948 distinct
patients and includes an age/sex proxy baseline, three frozen Glyph concepts and
paired patient-bootstrap prediction comparisons. It is internal validation.

A training-only scalar calibration fixes overamplified encoder initialization for
large dictionaries; `sae.calibrate_encoder_init=false` retains the legacy option.
Pretrained inference is unchanged. The report harness supports `--image-examples`,
with images ranked by activation independently of outcome labels, and displays
sample IDs, activations and exact association statistics. PubMed parsing now
selects the article's DOI rather than a cited reference's DOI. Large concept-map
edge construction is vectorized with unchanged edge rules.

The notebooks are executed with embedded PNGs; reports and HTML are included in
the demos. Raw caches, model weights and full prediction tables remain excluded.
No push, publication or history rewrite has been performed.


## Six prediction arms and grouped concept reports

Literature comparisons now include full and training-ranked top-ten versions of
Literature, Glyph, and their combination. The combined cap is ten predictors
in total, retaining up to ten literature predictors before adding ranked Glyph concepts. Wide dictionaries use the same ridge objective with a memory-safe
solver. Reports keep each concept's local map, association statistics and actual
activating training/held-out images together. Curated demo PDFs live directly in
the demo directories, with supporting material in `report_files/`.


## Prevalence sizing and bounded correlations

Static maps, HTML maps and the bundled report harness now use absolute activation
prevalence for node area, with a tiny visibility floor and a common density factor
for large dictionaries. Report zooms use the same prevalence rule. Correlation
links are visible by default in HTML and included in saved-run reports, with a
strict maximum degree of eight. The capped combined predictor now keeps the
training-ranked literature baseline (up to ten predictors) and adds ranked Glyph
concepts only to fill the remaining places. No SAE fits or k values were changed.


## Validation-stopped prediction and activation-scale reports

The prediction harness now restores validation-selected checkpoints and stops on
validation patience, with training-only preprocessing/ranking. Combined models
independently regularize literature and Glyph and can retain the literature-only
initial checkpoint. Candidate traces and both penalties are exported. Concept
pages include zero/33rd/66th/100th-positive-percentile examples, full histograms
and 95th–100th-positive-percentile tail detail. HTML reports provide expandable
tails. Comparison plots include paired-bootstrap p-value brackets and stars; the
all-variable combined row is omitted from the compact report. Correlation lines
are thinner and scale in width and opacity with absolute correlation.


## Third public-image application — 27 October 2026

[Smartphone skin-lesion diagnosis](../demos/pad_ufes/README.md) adds a raw-image DINOv3 demo with 1,373 independent patient observations and a source-backed eight-field clinical comparator. The scientific fit settings are defaults; selected k=80 misses the 0.85 reconstruction target, and both the warning and numerical results are retained. The local artifacts include an executed notebook, interactive HTML, four-page PDF, expandable image report and all six comparative arms. Clinical-only test AUROC is 0.939, Glyph 0.888 and full combination 0.963; these are internal diagnostic benchmark results.

The comparison harness now supports an explicit pooled top-ten strategy for richer clinical pools, handles constant held-out markers as non-estimable rather than crashing, and records fit statuses. Reports state the capped-selection policy, flag failed reconstruction targets and limit forest displays using training rankings while preserving all estimates/counts. The original two demos were not refitted. No release, push, model/data upload or history rewrite was performed.


## Sidebar image examples and Pages preparation — 4 November 2026

Interactive maps now include two activation-selected 2D thumbnails per concept when raw image paths are available. Images are shared across concept references, and `visualize(image_examples=0)` excludes them. The three demo viewers were refreshed from saved activations while preserving exact numerical payloads and layouts. Standalone sizes are 4.262 MB (digits), 13.395 MB (chest X-rays) and 11.101 MB (skin lesions). Pages-ready copies are 4.121, 4.432 and 2.069 MB of HTML, with thumbnails served separately on selection. A standard-library site builder copies only allowlisted public demo outputs; no models or research inputs are included. Viewer/reader/browser regressions passed 38 tests, the image API group passed three checks, and the site builder passed two checks. All six actual standalone/Pages viewer combinations displayed two images without JavaScript errors; hosted copies fetched only two thumbnails after selection.


## Clean initial history — 4 November 2026

At the author's request, the current source, documentation and public demos (including activation-selected sidebar thumbnails) are retained as a fresh initial snapshot before public release. Old development branches, tags and local stash refs are archived in a private backup outside the versioned repository. Publication of the package repository remains a separate author action.
