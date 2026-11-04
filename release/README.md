# Local release preparation

Nothing in this directory publishes data, pushes Git branches, uploads models,
or rewrites Git history. The author tools below run from a repository checkout;
reader-facing study/interpretation/report harnesses are also in the pip wheel.
Install this checkout into your environment first (`pip install -e .`), or use
the wheel built from it, so the author scripts import the current Glyph package.

## Stage activations and dictionaries

Prepare an explicit CSV with exactly `sample_id,public_sample_id`, containing a
one-to-one mapping from every fitted-run sample to its intended public accession
identifier. For the paper, this mapping must align with the released Merlin data.
The exporter does not invent a mapping or assume internal IDs are public.

```bash
python scripts/prepare_activation_release.py --run trusted_fitted_run \
  --public-id-map explicit_public_ids.csv --output new_local_release
```

Outputs are aligned `samples.csv` (public IDs and split only), one SciPy CSR
activation `.npz` per modality retaining original concept indices, SAE model
parameters as numeric `.npz`, normalization JSON, alive-feature indices and a
checksum manifest. Train-fitted Glyph dictionaries include their decoders;
encoder-only upstream dictionaries remain encoder-only. The exporter preserves
actual saved numbers rather than re-extracting or refitting.

Raw images, reports, clinical labels, covariates, internal ID mapping, interpretation
prompts and sample-level interpretation examples are omitted. Release scope,
public-ID linkage, source-model provenance and licensing still need author review.
Only the explicitly mapped saved run is exported, and only to the new local folder.
No clinical research cohort was exported in the implementation checks; a synthetic
fixture verified lossless activation export and public-ID replacement.

## Audit existing public assets and paper references

```bash
python scripts/audit_release.py --paper /path/to/version26.pdf --output local_audit.json
```

This uses anonymous, read-only requests to the pinned Hugging Face repositories
and checks both PDF text and embedded links. Cached/downloaded assets stay local.
It cannot infer historical model revisions or guarantee that an unknown endpoint
is absent rather than private. See [the current findings](../docs/release_status.md).

Before any eventual public release, the remaining author inputs are the historical
encoder identifiers/revisions, original SAE decoders where available, intended
SAE asset license and public sample-ID mapping/release scope. Source and data
publishing, versioning and Git-history replacement remain explicitly deferred.
