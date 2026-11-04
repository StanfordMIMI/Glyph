# Reader API verification — 23 October 2026

The manuscript's pretrained interface is now implemented:

```python
ct = Glyph.from_pretrained('ct-merlin')
i87 = ct.measure('I87', metadata='ct_metadata.csv', sample_id='scan_001')
space = ct.measure(metadata='ct_metadata.csv')
ct.visualize()
```

The same interface applies to findings and laboratory bundles. A full table
contains 8,192 original code columns; single-code outputs are study-indexed
Series. Measurements require neither labels nor a train/test split. Repeated
requests reuse computed embeddings; new cohorts and explicit recomputation
create separate saved runs. Saved runs reload without encoder downloads.

## Actual published encoder and SAE weights

These tests ran from an installed wheel outside the source checkout. A test
assertion checks that `glyph.__file__` is inside that installation.

| Path | New inputs | Checks |
| --- | --- | --- |
| CT / Merlin | One official public MONAI CT volume, not a research-cohort study | I87, full 8,192-column table, original indices, HTML, exact reload |
| Labs | Four independently generated panels using published model units/statistics | L127, full table, cache reuse, direct-encoder agreement, HTML, exact reload |
| Findings / Qwen3-8B | Three independently generated short reports | R64, full table, direct-encoder agreement, HTML, exact reload |

All three installed-wheel tests passed. Inputs for labs and reports are
synthetic software checks, not independent clinical validation. The CT case is
`spleen_test.nii.gz` from `MONAI/testing_data`, dataset revision
`ebd9119da13163de97216aca1dd9b8d5ef0febd7`. No private research images, reports or
patient rows were used. Logs and per-case verification JSON are under
`test_outputs/reader-api/installed-real-pytest/`; the JUnit result is
`test_outputs/reader-api/installed-real-tests.xml`.

The current real Qwen encoder download is revision
`1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`. This records the tested current asset;
it does not resolve the historical encoder revision used during SAE training.
The published dictionary revisions and normalization remain pinned.

Very short reports produced an embedding-magnitude shift warning. The new API
records this diagnostic and continues by default; `strict=True` rejects it.
The test does not adjust normalization or rescale activations to manufacture
agreement. New-cohort semantic or clinical validity must be evaluated separately.

The CT installation check also caught and fixed an unavailable `merlin-vlm`
version constraint. The `ct` extra now requires `merlin-vlm>=0.0.7` and
PyTorch >=2.6 with compatible torchvision for the upstream loading path. The
real CT test used an isolated CPU Torch 2.6 installation; no checkpoint security
checks were disabled.

## Learning new concepts with DINOv3

The README configuration ran end to end with a tiny local DINOv3 architecture:
image preprocessing, CLS extraction, SAE fitting, association search and HTML.
The checkpoint and training budget were deliberately reduced for software tests.
Separate tests compare extracted features with direct model outputs and verify
custom output fields and pooling.

The official pretrained DINOv3 test now passed after the user received access.
The accessible checkpoint revision was
`114c1379950215c8b35dfcd4e90a5c251dde0d32`. The test ran from the installed wheel
on 24 public handwritten digit images, through real embedding extraction, SAE
fitting, association search and HTML generation. Only the SAE training budget
and dictionary size were reduced for a fast software check; no clinical
performance or reconstruction-quality claim is inferred from that budget.

The authenticated test uses `HF_HUB_CACHE` to isolate model files while preserving
the normal stored HF login. Setting a separate `HF_HOME` had hidden that login
in the earlier isolated access probe. No credentials were copied into the repo.
Evidence is in `test_outputs/reader-api/dinov3-real-tests.log` and its JUnit XML.
To repeat after obtaining the official model access:

```bash
GLYPH_DINOV3_REAL=1 pytest tests/test_hf_embeddings.py::test_real_dinov3_readme_workflow_on_public_digits
```

All four real-weight workflows (DINOv3 learning plus published CT/labs/findings
measurement) have now executed. Browser checks on the three published-space
HTML files also passed: paper-code names, no JavaScript errors, and no external
page requests. Their generated graphs contained 40 active imaging concepts,
56 active lab concepts and 92 active findings concepts; full tables retained
8,192 columns in each case. The final installed wheel additionally passed ten
reader-API checks, including exported embedding reuse and exact reload.

## Other regression checks

The reader/HF/visualization group passed 43 tests, and saved-run/adjustment/skill
regressions passed 53 tests. A further 11 targeted checks cover exported embedding
reuse, strict-mode diagnostics and preservation of fitting split validation.
These groups overlap; they are not a count of unique tests. Tests also check
sample-ID order, zero/inactive concepts, incorrect codes, no SAE training,
new-run preservation, and exact persistence.

No package, source, models, cohort data or Git-history changes were published.

## Follow-up: full-width demo dictionaries and initial encoder gain

Both reader demos now use 8,192-entry dictionaries. An initial medical run exposed
poor high-k optimization with the unscaled transposed-decoder initializer. At
k=512 with 8,192 entries, the initial normalized-training MSE was 24.309; a
positive training-only gain reduced it to 0.156. With the same 2,000-step budget,
validation R² improved from 0.128 to 0.876. No held-out inputs or outcome labels
were used for that calibration or comparison. These are reconstruction results,
not prognostic-performance claims.

New fits calibrate the starting encoder gain against the actual training
reconstruction objective, including configured Matryoshka prefixes. The scalar
minimizes squared reconstruction error before optimization; positive scaling
preserves the initial TopK support. The training objective and model architecture
are unchanged. `sae.calibrate_encoder_init=false` retains the previous initializer.
Selection histories record the applied scale. Calibration runs only when fitting
a new dictionary; pretrained inference and saved-run reconstruction use their
stored parameters.

Regression checks cover the reduction in initial loss, unchanged TopK support
and decoder, nested losses, zero-valued inputs, the legacy option, selection,
measurement and saved-run reload. Diagnostic runs using the earlier initializer
were retained locally and marked as superseded rather than used as final demos.


The final full-width runs passed exact activation reload. Both notebooks executed
six code cells and embed three PNG outputs. Browser checks confirmed 1,629
training-active digit nodes with ten outcome spokes and 8,192 medical nodes with
one outcome spoke, with no JavaScript errors or remote resource requests. Both
three-page PDFs were visually reviewed, including activating images with sample
IDs and activation values. The final focused regression group passed 34 tests;
the preceding broad group passed 161 tests with one skip.

The fresh wheel was built with the declared setuptools>=77 requirement, installed
into an isolated directory and imported outside the checkout. It exposed all
four bundled skills, retained the 8,192 default maximum width and doubled
sparsity allowance, reloaded the full medical run with exact activations, and
generated identical report counts and selected image records using its bundled
report harness. Verification is saved locally at
`test_outputs/final-reader/verification.json`.


## Six prediction variants and per-concept evidence

The literature harness now fits full and top-ten Literature and Glyph models, plus full and capped
combined models that retain the literature baseline. Feature rankings use training-only logistic score tests, with
ten total predictors in the capped combined model. Regression tests change
held-out labels and values without changing rankings, verify wide ridge fitting
against the Newton solver, and check all six paired-bootstrap arms. The final
affected group passed 38 tests. A freshly installed wheel outside the checkout
fitted all six variants and reproduced the real medical report's concept counts,
selected images and per-concept pages. Verification is stored locally at
`test_outputs/grouped-reader/verification.json`.

Both notebooks were executed again, with four embedded PNG outputs each. The
medical PDF has four pages and digits has three. Every selected concept now
shares a page with its zoom, statistics and activating training/held-out images.
PDFs are directly in the demo folders; support files live in `report_files/`.
The full-space medical combination does not improve on age/sex; the corrected capped combination (three literature predictors plus seven Glyph concepts) reaches AUC 0.931.
Its earlier three-concept fitted model is a different, archived local analysis.
The underlying SAE fit remains k=768 with a single 8,192-entry training level;
the package's default nested levels are 128, 512, 2,048 and 8,192.

The separate medical k=64 diagnostic allowed 4,000 steps and stopped early at
1,680, restoring its best validation checkpoint. Train R² was 0.958 and validation
R² 0.682, with 3,390 training-active concepts. Longer training did not materially
improve validation reconstruction versus the original shorter k=64 trial. No
held-out test data were evaluated and the selected demo fit remains unchanged.
The result is recorded in `demos/sbu_cxr/sparsity_diagnostic.json`; its weights
remain ignored under `test_outputs/`.


## Package-wide prevalence sizing and correlation links

PDF/report, static JPG and both HTML modes share absolute prevalence-based node
areas, with a tiny floor and a common density factor for large dictionaries. No
per-map maximum normalization or logarithmic count sizing remains. The report
uses available cohort sizes to derive missing prevalences from counts; supplied
maps without prevalence use a uniform small marker. Correlation links are shown
by default and capped at eight at both endpoints, including externally supplied
atlas/report graphs. Hovering highlights the selected concept's links.

The final combined numerical/browser regression group passed 54 tests. Real
Chromium checks found no errors or remote requests in either demo: digits has
1,629 nodes and 1,610 links; medical has 8,192 nodes and 8,652 links, with maximum
degree eight in each. A fourfold prevalence change gives a fourfold node area.
Both notebooks were executed again with four embedded figures. PDFs, full maps
and per-concept zooms were visually inspected.

A fresh wheel was installed outside the checkout. Its shared area calculation,
link cap and baseline-preserving comparison were verified; the bundled report
harness reproduced the real medical report's statistics, images and correlation
links. Verification is under `test_outputs/prevalence-reader/`. The capped model
now retains all three literature predictors and adds seven ranked Glyph concepts,
with test AUC 0.931. The full-space model remains a separate arm. k=768 and the
SAE activations, reconstruction results and layout coordinates were unchanged.


## Validation stopping and activation-scale reports

Comparison predictors now stop on validation patience and restore their best
validation-AUROC checkpoint; validation log loss breaks ties. Combined models
regularize literature and Glyph independently, with the literature model as the
initial fallback. Input ordering is canonicalized, and tests verify identical
fits under manifest permutations and unchanged fits after held-out label changes.
Traces and best/actual iteration counts are exported. This deliberately changes
the previous convergence-only prediction protocol; scores do not all increase.
For the medical demo, current test AUCs are 0.839 literature, 0.783 all Glyph,
0.767 Glyph top ten, 0.935 full combination and 0.932 capped combination. The full
combined row is omitted from the compact presentation, with all numerical results
retained in CSV exports. The SAE, k=768, activations and associations are unchanged.

Concept pages retain one page per concept, with full activation histograms, zero
plus positive-sample percentile examples (33/66/100), and 95–100% active-tail
detail. Four strong train/test images remain below it. Image selection never uses
outcomes. Optional HTML reports expose expandable tails. Plot brackets use the
existing paired-bootstrap p-values with conventional stars, labelled exploratory
and unadjusted. Correlation line width/opacity are scaled down with |correlation|.


## Fresh-cohort portability and complete digit colors

A freshly installed wheel was imported outside the checkout and run on 120 newly
generated striped RGB images, independent of either demo and the research data.
It extracted real official DINOv3 embeddings, fitted a new 8,192-entry fixed-k
SAE, ran associations, generated HTML and produced the full report format. Both
concept pages had context thumbnails, linked region zooms, whole-feature
distributions, four scale examples and four tail examples, plus expandable HTML
tails. This deliberately short software test found no corrected associations; its
report correctly showed zero and labelled the feature pages exploratory. The
record is `test_outputs/portable-report-reader/new-cohort/verification.json`.
No new scientific demo or clinical validation is claimed by this fixture.

All ten digit outcomes now have individual colors. The digits use fixed HSLuv
hues spanning 15–303 degrees at saturation 95 and lightness 60, keeping an ordered
0–9 hue progression without a faded endpoint. The PDF has a numeric color strip.
Defaults and explicit custom-color overrides remain supported in HTML.


The digits report illustrates two spatially separated digit-4 associations. Automatic selection finds a complementary pair from positive corrected training associations using training activation correlation and map separation. For this saved demo, training-image review then chose nearby, more prevalent C1978/C5058, with 85/700 and 108/700 active training images, within-digit-4 r=-0.512 and map distance 0.238. The saved-run-only profile records this curation; fresh fits select their own pair. Visual notes are unvalidated. Generic reports accept focus_concepts/focus_outcome/concept_notes through metadata, preserving original statistics. Tests check separation and saved-run curation.

Final affected-suite verification: 67 tests passed, including real browser tests. The additional saved-run curation regression test passed.


## Third application: raw smartphone skin-lesion photographs — 27 October 2026

PAD-UFES-20 v1 was downloaded from the original Mendeley release and all four publisher SHA-256 checksums matched. The outcome-blind first-image-per-patient selection gives 1,373 patients, split 961/206/206 with 417/89/89 cancers in the imaged lesion. Every selected cancer is biopsy-confirmed. This is contemporaneous diagnosis, not prognosis. No author research cohort or automated interpretation was used.

The real DINOv3 checkpoint and an independently installed local wheel ran the default scientific configuration: 8,192 entries, nested levels 128/512/2048/8192, candidate k=5/10/20/40/80, 1,500 steps, training-only normalization and default association/graph settings. Runtime settings were four CPU threads, batch size 16 and no loader workers. Fit took 20.18 minutes on this CPU. The selected k=80 reached validation R²=0.756981 and test R²=0.759241, below the 0.85 target; the warning was emitted and is visible in the report. All split activations matched exactly after saved-run reload. All 1,600 actually training-active concepts are mapped, including unsupported/untested nodes. There are 164 corrected discoveries, 17 same-direction corrected test reproductions among 138 evaluable discoveries, and 26 unavailable comparison estimates.

The clinical comparator reconstructs the eight inputs explicitly evaluated by Pacheco and Krohling (PMID 31760271, DOI 10.1016/j.compbiomed.2019.103545): age, location and six symptoms. One-hot location expands these to 22 columns, one of which is absent/constant. Missing symptoms remain missing before train-only imputation. Marker definitions were frozen before comparative fitting; coefficients are newly fitted rather than copied from the historical classifier. All six prediction arms share the patient cohort. Test AUCs are clinical 0.938874, clinical top ten 0.945981, Glyph 0.888217, Glyph top ten 0.821569, full combination 0.963123 and pooled top ten 0.921877. Full combination minus clinical is +0.024249 (paired 95% CI +0.006355 to +0.045041; exploratory raw p=0.017991, 2,000 patient resamples). The pooled cap selected four clinical measurements and six Glyph concepts. No selection was changed using test performance.

This run exposed a real constant-held-out-marker OR failure: statsmodels omitted the intercept and the coefficient indexing crashed. Constant split predictors now produce an explicit non-estimable fit status, and the direct regression test passes. The richer clinical pool also motivated an explicit pooled top-ten option and training-selected display limits for dense forest panels. The full matched estimates and reproduction counts are retained. The affected suite passed 38 tests; after the final display change, all 20 report tests passed, including the additional training-only forest-selection regression.

Chromium verified all 1,600 graph nodes, 6,166 edges, maximum degree eight, seven outcome spokes, prevalence-proportional area, tooltips, two expandable report panels and no JavaScript errors or external requests. The four-page PDF was visually inspected. The notebook executed all cells and embeds four PNGs in a 4.6 MB file. See [the demo](../demos/pad_ufes/README.md), its environment/settings records, source manifests and saved numerical tables. All work remains local; nothing was pushed, published or uploaded.
