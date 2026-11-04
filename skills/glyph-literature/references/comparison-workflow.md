# Comparison workflow

Start from the requested outcomes, index date, eligibility rules, cohorts and
available measurements. For a paper reproduction, use its fixed profile and
marker list. For a new literature study, search primary studies or authoritative
guidelines and record citations supporting each marker's definition. Distinguish
a direct measurement from an available-data proxy.

The marker manifest should record stable IDs, source citations, formulas, units,
cutoffs, measurement windows, required fields and missingness/proxy decisions.
Keep the manifest and any search decisions with the run artifacts.

Compare literature-only, Glyph-only and combined arms on the same eligible
observations. Include Phecode arms only when the study calls for them. Learn
preprocessing on train, tune on validation and freeze model/feature selections
before evaluating held-out sites. Use paired comparisons and the study's stated
observation unit; the Glyph manuscript treats CT studies as independent.

Preserve separate tasks: predictive model AUC comparisons, individual-marker
association/replication comparisons, and optional confounder adjustment. A good
predictive score does not make an individual association causal.

Export model settings, selections, predictions, sample/event counts, exclusions,
effect estimates, uncertainty and the exact multiple-testing families. Reusing
released computed outputs is replay; actually fitting the models is refit.
Label the two clearly. Changing held-out labels must not change fitting or selection.

When transferring a research harness into the distributable tooling, remove
private filesystem assumptions and use explicit input/configuration paths. Compare
its outputs against the reference on fixed inputs before treating it as equivalent.

## Runnable CSV contract

- Cohort: unique `sample_id`, `split` (`train`, `val`, then one or more held-out
  names), binary `y`; optional `patient_id` for cluster resampling. Patients must
  not cross fitting, tuning and held-out splits.
- Glyph vectors: unique `sample_id`, numeric concept columns. IDs must exactly
  cover the common cohort.
- Literature vectors: unique `sample_id`, exactly the numeric marker columns in
  the frozen manifest. Supply deterministic reconstructed values, not formulas
  that require an agent at fit time.
- Manifest: `markers` list with `column`, `definition`, `source_url` and preferably
  DOI, units, formulas and proxy status; `glyph_features` frozen column names;
  `glyph_selection_rule`; study limitations and search provenance.

Prediction candidates are fitted on training rows and monitored on the explicit
validation cohort. Each candidate restores its best validation-AUROC checkpoint;
validation log loss breaks AUROC ties. Twenty checks without improvement stop
optimization by default. The frozen manifest can specify
`prediction_early_stopping: {patience: 20, min_delta: 0.000001}`. Preprocessing,
mean imputation and missingness indicators use training rows only.

Standalone arms tune a ridge grid. Combined arms retain the literature arm's
validation-selected penalty and independently tune Glyph's penalty over the
same grid plus 1,000 / 10,000 / 100,000. Their iteration-zero checkpoint is the
fitted literature baseline, so validation can retain that fallback. Test outcomes
do not select penalties, iterations or features. Candidate traces, best iteration,
actual iterations, stopping status and both penalties are exported in `tuning.csv`
and `metrics.csv`. Coefficients use a canonical predictor order so equivalent
sets produce identical fits. Early-stopped checkpoint gradients need not be zero.

Metrics include AUC, Brier score and log loss. Paired subject/patient-cluster
resampling uses fixed predictions, rejects single-class draws and does not include
model-training uncertainty. A perfect ranking can give a zero-width interval.

Individual associations use the training-ranked min(number of literature markers, number
of frozen Glyph concepts) markers per arm. Their Bonferroni family includes both
arms, within each split. A saved larger discovery-family correction is not
silently reused as the matched-comparison correction. The runner saves exact
model coefficients and preprocessing, tuning scores, predictions, all metrics,
association estimates and correction families, bootstrap diagnostics, frozen
manifest and input hashes. Confounder adjustment is a separate optional analysis
and is not claimed by this runner.


## Six prediction variants

The runner fits Literature, Literature top 10, Glyph, Glyph top 10,
Literature + Glyph, and Literature + Glyph to 10. The full arms use every
marker or concept declared in the frozen manifest; they do not silently include
undeclared columns. For a full-space comparison, declare all training-active
concepts as `glyph_features`. `comparison_top_n` changes the default cap of ten.
The combined capped arm retains the training-ranked literature baseline (up to ten predictors), then fills remaining places with the highest-ranked Glyph concepts. With three literature predictors, it uses those three plus seven Glyph concepts.
If fewer than ten predictors exist, the capped arm uses all available predictors;
show the actual counts even when two arms coincide. Missingness indicators are
additional preprocessing columns, not separately ranked candidate markers.

Ranking uses two-sided univariate logistic score tests on training rows only,
with deterministic tie-breaking. Validation selects ridge strength, never the
feature ranking; held-out samples are used only for evaluation. Exports include
`training_rankings.csv` and `feature_selections.json`. All six arms share patient
resamples. The all-literature versus all-combined comparison remains primary;
other pairwise p-values are exploratory and unadjusted. Wide models solve the
same ridge objective using L-BFGS rather than forming a feature-by-feature Hessian.
Individual-marker comparisons remain a separate matched-count analysis.

The compact report omits the full all-variable combined row; its numerical results
remain in the comparison exports. The main displayed paired comparisons have
brackets, exact p-values and conventional stars, marked exploratory/unadjusted.

For larger clinical baselines, set `combined_top_n_strategy: "pooled"` in the frozen manifest to rank literature and Glyph columns together on training data and take ten total. The compatibility default, `"retain_literature"`, keeps up to ten literature columns first and fills remaining places with Glyph. Thus a clinical pool of ten or more columns can leave no Glyph columns under that default. Pooled selection may select either or both sources; inspect the saved feature provenance rather than assuming a mixture. Combined models retain the exact full clinical baseline checkpoint only when all its columns are present.
