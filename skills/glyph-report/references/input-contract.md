# Input contract

Saved-run mode uses `Glyph.load_run`, saved association rows, sparse features,
alive masks and cached interpretations. Only load a trusted local run because
its existing artifact format includes pickle. Layouts use train features; the
analysis itself is not rerun. A complete saved train HTML layout can be reused;
a partial layout is not a full-space map and must be recomputed.

Table mode accepts CSV or Parquet:

- Concepts: unique `concept_id`, finite `x`, `y`, and `modality`. Optional `name`,
  `interpretation`, `interpretation_accuracy` (fraction 0–1),
  `interpretation_accuracy_context` (e.g. Original release), and `layout_status`.
  Supply the whole space, not just concepts having significant associations.
- Associations: `concept_id`, `outcome`, `split`, `odds_ratio`, and
  `odds_ratio_p_value_bonferroni`. Optional raw `odds_ratio_p_value`,
  `odds_ratio_ci_lower`, `odds_ratio_ci_upper`, `c_f`, and `outcome_display`.
  Each concept/outcome/split key must be unique. Corrected p-values must already
  reflect the original study family, including tests omitted from the display.
- Optional metadata JSON: `scope`, `source_name`, `notes` and `testing_family_size`.
  These describe supplied data; they do not alter tests or estimate effects.

`--split train --comparison-split test` selects the two slices. Significance uses
corrected p < alpha, in both directions. Comparison reproduction requires a finite
estimate, corrected p < alpha and the same side of OR=1. The full map colors each
significant node by its strongest significant outcome. Five dominant colors are
shown, plus Other significant outcomes; gray means no significant estimate in the
selected split. Color does not assert an exclusive biological category.

Zooms select up to two distinct significant concepts by corrected p-value, using
cached interpretations when available and preferring nonoverlapping map regions.
The PDF/Markdown state this selection rather than implying representative sampling.
Geometry and effect estimates are separate: a zoom is a spatial neighborhood,
not an independently tested biological cluster.

The script writes only concept-level aggregates and rendered figures. It does not
export study rows, identifiers or clinical text. Never embed raw report examples
or patient identifiers in a findings brief unless explicitly requested.

Published interpretation accuracy describes the original release evaluation, not
performance on a new cohort. The report labels that distinction explicitly.
