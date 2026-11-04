# Glyph labs encoder

Hugging Face repository: `RobbieHolland/Glyph-labs-encoder`.
Weights are released under the MIT license, matching Glyph. The package downloads
the model separately from its wheel and pins a tested HF commit.

## Model and assets

This is the 768-dimensional foundation encoder that supplies embeddings to the
separately released `labs-loinc` SAE. It has three transformer layers, four heads,
width 768, feedforward width 3072, and mean pooling. Its unused trained CLS parameter
is retained so that the full checkpoint loads strictly, without random replacements.
All 743 parameter tensors match the original research checkpoint.

The bundle contains `labs_encoder.ckpt` (tensor-only), `lab_metadata.json`,
`lab_loinc_map.csv`, `lab_catalog.csv`, `bundle.json`, and a model card. The metadata
contains 350 retained model labs, all numeric, with their training normalization
statistics and units. The mapping contains 193 LOINC codes; 89 map to multiple
model labs. Units, an explicit lab name, or a model lab ID must resolve ambiguity.
Nine labs have explicit local codes and one remains unresolved. The September
review status and later source-audit conflicts are preserved; see
[labs_mapping.md](labs_mapping.md). No automatic unit conversion is performed.
No patient records are included.

## Published preprocessing

The `published-v1` profile selects the last measurement per patient, model lab,
and calendar quarter across the supplied history. It uses quarter-start timestamps,
keeps tokens with years-prior between 0 and 0.5, orders newest first, and takes the
first 128. Values use training normalization and clipping to [-3, 3]. The original
parser is preserved, including its handling of signed/scientific numeric strings;
raw strings must not be silently converted to floats. Preparation reports count
these legacy unparsed strings. Empty studies are reported and require an explicit
exclusion option rather than silently changing the cohort.

Quarter aggregation precedes the lookback filter. Consequently, a measurement may
occur after its quarter-start timestamp, potentially even after the scan. This
profile reproduces the published inputs; it does not ensure strictly prescan
measurements. Analyses requiring that restriction need a separately versioned
preprocessing profile and revalidation.

The padding mask is explicitly boolean, matching the published fused-inference
semantics while avoiding PyTorch's optimization-dependent float-mask behavior.
Padding is fixed to 128 tokens. The older HypothesisExtractor preprocessing remains
available only through the explicit `hypothesisextractor-v1` profile.

## Verification

On 20 studies from the research cache, prepared tokens match exactly. Embedding
relative L2 error is 0.0568%, with maximum absolute difference 0.00371. Eighteen
studies have identical active concept sets; two have 39/40 overlap, with the changed
concepts at ranks 40 and 41. Sparse activation relative L2 error is 1.49%, reflecting
those threshold swaps. This is close reproduction, not bitwise equality. Tests run
on CPU; historical cached embeddings came from a different execution environment.

The original research encoder and this implementation produce identical outputs
on the checked synthetic input. Boolean-mask tests cover both PyTorch fast paths.
The verification script saves aggregate errors and no patient-level lab rows.

## Provenance and license

Checkpoint provenance is the research run `woven-bee-1671`, epoch 25, validation
loss 0.3214. Source hashes and release-asset hashes are recorded in `bundle.json`.
The checkpoint matches the laboratory encoder in HypothesisExtractor. The exact training-cohort dataset version is not recorded in the available
checkpoint metadata. Model weights are released under the included MIT license. This encoder is intended for research feature
extraction and has not been validated for clinical decisions.

See [extraction.md](extraction.md) for preparation and loading commands.
