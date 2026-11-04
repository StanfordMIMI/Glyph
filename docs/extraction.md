# Embeddings, pretrained concepts, and fitted runs

Raw inputs are the default. `fit()` computes fresh FM embeddings, then trains a
new SAE. It does not silently reuse an embedding cache. Install only the encoder
extras you need; model weights are downloaded on demand, never included in the
Glyph wheel.

```bash
pip install 'glyph-discovery[zoo,ct]'       # Merlin CT extraction
pip install 'glyph-discovery[zoo,text]'     # Qwen3 report extraction
pip install 'glyph-discovery[zoo,labs]'     # labs; encoder downloads from HF
```

## Measure published concepts without training

```python
from glyph import Glyph

ct = Glyph.from_pretrained('ct-merlin')
i87 = ct.measure('I87', metadata='ct_metadata.csv')
one = ct.measure('I87', metadata='ct_metadata.csv', sample_id='scan_001')
concepts = ct.measure(metadata='ct_metadata.csv')
ct.visualize()
i87 = concepts['I87']
```

Metadata contains unique `sample_id` values and CT paths under `image_path`
(`ct_path` is also accepted). No labels or train/test partition are required.
`from_pretrained` loads the dictionary, not your cohort's embeddings. `measure`
loads the matching FM when needed, computes missing study embeddings, and
applies the frozen SAE with original indices. Calls on the same cohort reuse
measured embeddings. Whole-space results include all 8,192 concept columns;
the graph displays those active in your cohort, with their available published
interpretations. One- and two-study views do not estimate correlation edges.

Use `findings-qwen3` with `report_path`/`findings_path`, or `labs-loinc` with
`labs_path`. Supply `input_col=` to `from_pretrained` for custom column names.
The findings bundle requires the original 8B encoder, independently of the small
0.6B preset used when learning new text concepts. Lab JSON inputs are prepared
from raw dated measurements using the lab preparation tools below.

```python
labs = Glyph.from_pretrained('labs-loinc')
concepts = labs.measure(metadata='labs_metadata.csv')
l127 = concepts['L127']
```

Pass `device`, `cache_dir`, `output_dir`, `encoder_kwargs`, or `batch_size` to
`from_pretrained` when needed. Supply compatible raw FM embeddings to skip the FM:

```python
concepts = ct.measure(metadata='metadata.csv', embeddings_path='merlin_embeddings.pt')
```

`sample_id` selects your metadata row. `split=` is only an optional filter on
your own metadata's split column; it never selects original paper studies.
Omitting a concept code measures every concept. Existing fitted runs also support
`measure()` to retrieve a full table and `measure('I87', split='test')` for
specific held-out activations. Calls without a split cover all supplied studies.

New cohorts create separate saved runs. `recompute=True` starts a new run for
the same metadata. `Glyph.load_run(ct.run_dir)` restores cached measurements
without loading an FM or downloading dictionary weights. New-cohort embedding
magnitude differences warn by default and are recorded in the run manifest;
`strict=True` rejects those differences. Normalization is never re-fitted to
force a new cohort to match the paper. Transfer must be evaluated separately.

## Published concepts with labelled association testing


```python
from glyph import Glyph

glyph = Glyph('config', config_overrides=[
    'data.metadata=/path/to/metadata.csv',
    'data.input_cols=[ct_path]',
    'data.outcome_cols=[outcome]',
    'data.embedding_map={ct_path:merlin}',
    'data.extraction_batch_size=1',
    'data.output_dir=/path/to/runs',
])
glyph.fit(pretrained={'ct_path': 'ct-merlin'})
glyph.hypothesis_search()
glyph.visualize()
print(glyph.run_id, glyph.run_dir)
```

The pretrained mapping covers every input column. Bundles are `ct-merlin`
(2048-dimensional Merlin embeddings), `findings-qwen3` (4096-dimensional Qwen3
embeddings), and `labs-loinc` (768-dimensional labs embeddings). Each bundle
selects the matching encoder and pinned SAE revision. Published normalization,
concept indices, and descriptions are retained. No SAE is trained or sparsity
selected when applying these dictionaries.

Merlin follows HypothesisExtractor: install `merlin-vlm`, call
`Merlin(ImageEmbedding=True)`, and let the package obtain the StanfordMIMI HF
checkpoint. CT inputs are NIfTI files. Preprocessing is RAS orientation, spacing
(1.5, 1.5, 3), HU clipping/scaling from [-1000, 1000] to [0, 1], padding and center
crop to (224, 224, 160). The adapter handles both Merlin singleton-axis formats.

Qwen3 uses `Qwen/Qwen3-Embedding-8B`, FINDINGS extraction from report files, an
appended EOS, left padding, truncation at 8192 tokens, last-token pooling, and L2
normalization. These details are part of the embedding definition. The upstream
encoder revision can be set in `data.embedding_model_kwargs`; current paper
assets do not document a pinned encoder revision, so exact historical encoder
provenance still needs completing.

## Look up concepts by code

```python
i87 = glyph.measure('I87', split='test')  # pandas Series indexed by sample_id
r64 = glyph.measure('R64', split='test')  # requires findings-qwen3
l92 = glyph.measure('L92', split='test')  # requires labs-loinc
```

Paper codes select the corresponding published dictionary and original feature
index, regardless of the input column name. They work after pretrained `fit()`
and after `Glyph.load_run()`. Zero activations are retained. With two inputs
using the same dictionary, specify `modality='ct_path'` to choose one.

For newly learned dictionaries, use an integer index:

```python
activation = glyph.measure(87, modality='image_path', split='test')
```

A newly learned feature 87 is not the paper's I87. Paper-code lookup therefore
requires the matching pretrained dictionary instead of guessing from modality
or encoder names. The returned Series is a copy, preserves study order, and contains raw sparse
activations before association-test standardization.

## Explicit precomputed embeddings

```python
glyph.fit(embeddings_path='/path/to/export_directory')
# Or use HypothesisExtractor's saved file format.
glyph.fit(pretrained={'ct_path': 'ct-merlin'},
          embeddings_path={'ct_path': '/path/to/ct_embeddings.pt'})
```

An explicit path skips all encoder loading and raw extraction, even though
`recompute_embeddings` defaults to true. Setting it to false without a path is an
error; a cached file is never guessed. Every `fit()` still fits a new SAE unless
`pretrained` is provided.

Files must contain raw, unnormalized FM embeddings, not sparse activations.
Accepted `.pt` schemas are `{sample_ids, embeddings}` and Glyph's exported
`{sample_id, <input_without_path>_embedding, split}`. A directory resolves the
latter filenames. A file can be used directly for one input, or map each input
column to its file. Import aligns by sample ID, rejects duplicate/missing IDs,
nonfinite tensors, and conflicting split labels. Extra rows in a supplied file
are ignored. The metadata still defines the cohort, outcomes, and splits; it need not contain
raw input columns when explicit embeddings are supplied.

## Separate extraction

The `glyph/extraction/` module can run independently of training or association
analysis. It reuses the same encoder adapters as the main API.

```python
from glyph.extraction import extract_embeddings
embeddings = extract_embeddings(metadata_dataframe, {'ct_path': 'merlin'},
                                device='auto', batch_size=1)
```

```bash
python -m glyph.extraction --metadata metadata.csv --encoder ct_path=merlin \
    --output-dir new_embeddings --batch-size 1
```

Without a split column, standalone extraction labels all rows train; for a
subsequent analysis, provide explicit train/val/test splits in metadata. The
extractor loads one encoder at a time. Large encoders such as Qwen3 need enough
memory, and CT batch size should be chosen for the available hardware.

## Saved runs

Each call to `fit()` creates `<output_root>/<UTC timestamp>-<random ID>/`.
`glyph.run_dir` and `glyph.run_id` identify it. Earlier runs are retained.
`run.json` records source and completion status; failed fits are marked failed.

Each completed run contains raw embeddings and sample IDs, sparse activations,
active-concept masks, SAE selection/provenance, fitted encoder weights,
normalization, aligned outcomes, and a configuration snapshot. Credentials are
redacted. Published SAEs have encoder-only checkpoints; no decoder or
reconstruction performance is invented. Hypothesis search and HTML write to the
current run directory.

```python
loaded = Glyph.load_run('/path/to/run', device='cpu')
concepts = loaded.encode_embeddings(raw_fm_embeddings, modality='ct_path')
loaded.hypothesis_search()
```

Loading does not initialize FM encoders or retrain the SAE. A later `loaded.fit()`
creates a new sibling run and requires access to the configured cohort metadata.
Result files retain the package's existing pickle format: load only trusted runs.
Covariate adjustment and new raw extraction require the original input metadata.

## Labs release bundle

The foundation encoder and reviewed mappings are publicly available at
[RobbieHolland/Glyph-labs-encoder](https://huggingface.co/RobbieHolland/Glyph-labs-encoder)
under MIT. Glyph pins revision `972acbaea6d7ea2249a84d562a5f6dbabb994827` and downloads
it automatically. A local `bundle_dir` remains an optional override.

```python
from glyph.extraction.labs import LabsEncoder, prepare_lab_inputs

encoder = LabsEncoder()  # downloads the pinned public bundle
prepared_csv = prepare_lab_inputs('metadata.csv', 'measurements.csv',
                                 encoder.bundle_dir, 'new_lab_inputs')
```

Use `LabsEncoder(token=False)` to explicitly disable authentication. Set
`cache_dir=` to choose where HF stores the downloaded bundle.

Prepare the bundle from trusted research files with:

```bash
python -m glyph.extraction.bundle --checkpoint /path/to/labs_encoder.ckpt \
    --metadata-pickle /path/to/lab_types_analysis.pkl \
    --loinc-map /path/to/lab_loinc_map.csv --output-dir new_labs_bundle
```

This validates every trained parameter and writes `labs_encoder.ckpt`, a
350-lab JSON metadata catalog with training units and statistics, the supported
LOINC map, a human-readable `lab_catalog.csv`, and a checksummed `bundle.json`.
No weights or training statistics are shipped in the pip wheel. The released
bundle also includes mapping provenance and the later source audit.

The September 2026 mapping has 340 LOINC assignments, nine explicitly local codes
and one unresolved model lab. Provisional statuses and later source-audit conflicts
are retained. See [labs_mapping.md](labs_mapping.md) for provenance. Both LOINC
and `STANFORD_LAB//...` local codes are accepted in the `code` input column.

The raw-lab preparation command is:

```bash
python -m glyph.extraction.labs --metadata metadata.csv --labs measurements.csv \
    --bundle-dir /path/to/bundle --output-dir new_lab_inputs
```

Metadata needs `sample_id`, `patient_id`, and study `date`. Measurements need
`patient_id`, `date`, `value`, and either `loinc_code` or an explicit model
`lab_id`. Aliases start/code and Taken Date/Value/Units are accepted. Supply
measurement `unit` and, where needed, `lab_name`. Some LOINC codes map to multiple
trained lab IDs; the package disambiguates using units/names and refuses to pick
one silently. Explicit lab IDs can override selected ambiguous rows. Units must
match the training catalog; the package does not guess conversions. Omit unit
validation only with `--assume-training-units` when values are already known to
use those units.

The command writes per-study JSON inputs, a Glyph-ready metadata CSV, and a
preparation report. Studies without eligible labs cause an error unless excluded
explicitly with `--drop-empty`. Their number is reported. Fit with
`pretrained={'labs_path': 'labs-loinc'}` to download the matching encoder and SAE.
Set `data.embedding_model_kwargs={labs_path:{bundle_dir:/path/to/bundle}}` only
when using a local encoder bundle. The same preparation is available through
`prepare_lab_inputs` and `prepare_labs` in `glyph.extraction.labs`.

The `published-v1` profile matches the research input cache. It aggregates the
last measurement per patient/lab/calendar quarter before applying the study
window, timestamps it at the quarter start, retains records from 0 to 0.5 years
before the study, orders most recent first, and uses the first 128 tokens. A
quarter's selected measurement can occur later than its quarter-start timestamp.
This reproduces published inputs; it is not a claim that every contributing
measurement preceded CT. The old calendar-six-months-before/60-days-after helper
is available explicitly as `profile='hypothesisextractor-v1'`.

Published inference preserves raw value strings. The historical tokenizer's
numeric-string parsing is part of that model version; signed and scientific
strings can remain unparsed. The CLI preserves string inputs and the report
counts affected values. A different value policy requires an explicitly versioned
encoder/preprocessing definition, rather than silently changing published concepts.

Padding is fixed at 128 tokens for this profile. The encoder uses an explicit
boolean padding mask to reproduce published fused-inference semantics consistently
when PyTorch's optimized path changes. This fixes the substantial behavior change
caused by treating a float 0/1 mask as an additive attention bias. Round-trip and
fastpath-on/off tests cover this behavior. CPU/GPU rounding can still change
near-tied concepts at the top-k boundary; see the verification record.

## Real CT parity test

`tests/test_ct_parity.py` is an opt-in integration test for raw CT -> Merlin ->
published SAE versus both a cached embedding and a released activation row.
Point `GLYPH_CT_PARITY_CASE` at a private local JSON file as described in its
header. It checks exact active-concept IDs, embedding cosine >= 0.99999, and relative L2
error <= 0.5% for embeddings and activations. Strict numerical parity is recorded
separately rather than assumed. It writes
only aggregate errors to its report. The normal test suite skips this test.
Private image paths and identifiers belong in ignored local files, never fixtures
committed to the repository.

## Public labs download integration test

From a Glyph checkout, run:

```bash
GLYPH_HF_INTEGRATION=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  python -m pytest tests/test_labs_hf.py -q
```

This downloads the pinned public bundle into a fresh test cache without a login,
validates checksums and mapping counts, prepares synthetic LOINC inputs, extracts
768-dimensional embeddings through the package API, then downloads and applies
the published 8192-concept labs SAE. Synthetic inputs check the software path,
not clinical validity. The test is opt-in because it downloads model weights.
