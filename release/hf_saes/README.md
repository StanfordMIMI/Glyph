# SAE model card draft — local preparation only

Intended target is the already-used `RobbieHolland/HypothesisExtractor` model
repository, or a separately approved model repository with an updated registry.
No destination or remote content is changed by this draft.

The current pinned bundle contains `sae_image.ckpt`, `sae_findings.ckpt`,
`sae_labs.ckpt`, `normalization.yaml` and `interpretations.csv`. The encoders map
2048-dimensional Merlin CT embeddings, 4096-dimensional Qwen3 findings embeddings,
and 768-dimensional lab embeddings to 8192-concept top-k-40 dictionaries.

The card should state:

- Exact SAE and upstream encoder revisions and preprocessing profiles. Current
  historical Merlin/Qwen commit identifiers are unresolved; current HEAD must
  not be substituted as historical provenance.
- Normalization, concept index identity and interpretation coverage. A concept
  label is an automated suggestion, and original-release interpretation accuracy
  is not new-cohort semantic validation.
- Decoder availability. Current published bundles are encoder-only, so their
  new-cohort reconstruction R² cannot be computed from these assets alone.
- Evaluation scope, cohort provenance and out-of-distribution limitations.
- Activation/data release location and explicit linkage to public study IDs,
  once that release exists. Do not claim a staged local export is public.
- The authors' intended license for the SAE weights and interpretations. The
  pinned HF revision currently has no license metadata or model card; openness
  alone does not supply the missing license declaration. The separate labs
  foundation encoder already has an MIT release.

This draft deliberately does not fabricate missing training provenance or assign
an unconfirmed asset license.
