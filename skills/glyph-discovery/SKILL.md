---
name: glyph-discovery
description: Run Glyph on cohort data to learn or apply sparse concepts, test associations, and optionally adjust for confounders.
---

# Glyph discovery

Use the installed package API for cohort analyses. The other bundled skills cover
literature studies (`glyph-literature`), viewer presentation (`glyph-html`) and
findings briefs (`glyph-report`).

The user supplies a metadata CSV containing a unique study ID, input columns,
outcome columns and optionally explicit train/val/test splits. Input columns are
configured in `data.input_cols`, and `data.embedding_map` selects their encoders.
Precomputed embeddings are supported. Read the installed signatures/configuration
before using an unfamiliar version; do not infer methods from the manuscript.

```python
from glyph import Glyph

glyph = Glyph('config', config_overrides=[
    'data.metadata=/path/to/metadata.csv',
    'data.input_cols=[image_path]',
    'data.outcome_cols=[outcome]',
    'data.embedding_map={image_path:resnet18}',
    'data.output_dir=/path/to/new_outputs',
])
glyph.fit()
glyph.hypothesis_search()
glyph.visualize(outcome='outcome', split='test')
```

`fit()` greedily selects sparsity using validation reconstruction R². Default
candidates are `[5,10,20,40,80]`. The first model reaching 0.85 defines `k0`;
0.90 allows `2*k0`, and 0.95 allows `3*k0`. The search is capped by the largest
configured candidate. A fixed list such as `[40]` fixes k. Inspect
`sae_selection.json`; a warning below 0.85 means the best tested reconstruction
was retained despite missing the quality target. Reconstruction quality alone
does not establish that concepts are clinically meaningful.

Association search supports binary 0/1 outcomes, with -1/NaN marking unavailable
labels. Regression probes do not imply that continuous-outcome PheWAS is supported.
Read results from `glyph.phewas_results` or the saved `hypotheses.pkl`;
`hypothesis_search()` returns the Glyph object for chaining.

Confounder adjustment is opt-in and needs the `confounders` installation extra:

```python
glyph.hypothesis_search(
    adjust_confounders=True,
    confounder_cols=['age', 'sex', 'bmi'],
    adjustment_device='cuda',  # omit for CPU
)
```

This is FMR orthogonal confounder adjustment, default tree depth 2. Report adjusted
and matched-population unadjusted ORs, CIs, p-values, eligible/event counts and fit
statuses. Do not enable it unless requested. Each study is the observation unit.
The map and rankings currently use the unadjusted associations.

For arbitrary HF model IDs, configurable output fields and user-owned models,
read [the HF/custom encoder guide](references/hf_embeddings.md). Select pooling
from the actual model recipe; do not infer scientific embedding semantics from
a field name or tensor shape.

Read [pretrained measurement](references/pretrained_measurement.md) for input
formats, caching, full-space outputs and saved runs. For direct measurement
without training or labels, use
`Glyph.from_pretrained('ct-merlin')`, then
`measure('I87', metadata='ct_metadata.csv', sample_id='scan_001')` for one study
or `measure(metadata='ct_metadata.csv')` for the full dictionary. Whole-space
results are DataFrames with original I/R/L codes; select `concepts['I87']`.
`visualize()` explores active cohort concepts. Sample/split filters refer to the
user's data. Repeated measurements reuse embeddings. Inputs and their formats
are described in the extraction guide; do not invent concept identities for
newly trained dictionaries.

For published dictionaries, call `fit(pretrained={input_column: bundle_name})`.
Names are `ct-merlin`, `findings-qwen3`, and `labs-loinc`. Glyph chooses the
matching Merlin, Qwen3, or labs encoder, downloads the pinned SAE assets, and
uses published normalization and concept IDs without training a new dictionary.
Install the `zoo` extra and the relevant `ct`, `text`, or `labs` encoder extra.
The labs encoder downloads the public `RobbieHolland/Glyph-labs-encoder` bundle
at Glyph's pinned revision automatically. A local `bundle_dir` is optional.
`LabsEncoder().bundle_dir` gives the downloaded folder for raw-lab preparation.
See `docs/extraction.md` for the opt-in anonymous HF integration test.
Use `python -m glyph.extraction.labs --help` to prepare raw LOINC tables. Supply
training-compatible units, resolve ambiguous mappings explicitly, and preserve
the published value-string and quarterly preprocessing profile. The preparation
report records empty studies and legacy unparsed numeric strings. Do not silently
drop studies or change the frozen encoder's value parser. Never
substitute randomly initialized weights or refit normalization for a published SAE.

Without an embeddings path, `fit()` computes fresh embeddings from raw inputs.
Pass `embeddings_path=` (one file, an export directory, or input-column-to-file
mapping) to skip encoder loading and extraction. These must be raw FM embeddings
from the exact matching encoder and preprocessing, not already-normalized vectors.
`recompute_embeddings=False` requires an explicit path; no implicit cache is used.

Each `fit()` creates a unique directory under the configured output root. Report
`glyph.run_id` and `glyph.run_dir`. This saves the SAE weights, normalization,
activations, sample IDs, outcomes, and selection/provenance. Reload with
`Glyph.load_run(run_dir)`; use `encode_embeddings(raw_tensor, modality=...)`
to apply the saved dictionary. A subsequent `fit()` creates another run.
Only load trusted saved runs because legacy result files use pickle.

Preserve original concept/dictionary identities when assembling results. Report
the multiple-testing family and corrected p-values, not just ranked effect sizes.
Use a fresh output directory for exploratory reruns of established research results.

Outcome labels are recommended but optional for `fit()` and concept exploration.
With `data.outcome_cols=[]`, no supervised probes or PheWAS are run; `visualize()`
can show the entire alive space after fitting without association claims.

For optional interpretation, read [configuration and scoring](references/interpretation.md).
Use `glyph harness interpret` for a trusted saved run. Gemini/Vertex credentials,
text columns and finite call scope must be explicit. Malformed or incomplete
held-out responses are unscored; correlation annotations are not semantic validation.
Use `autointerp.enabled=false` when interpretation is deferred.
