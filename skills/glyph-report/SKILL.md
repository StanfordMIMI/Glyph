---
name: glyph-report
description: Produce a brief visual findings report from saved Glyph results, with the entire concept space, significant-node colors, zoomed evidence and cached interpretation suggestions.
---

# Glyph findings brief

Produce a punchy report from saved results, without adding a `.report()` method to
Glyph. This is a separate agent harness. HTML and viewer hosting remain optional.
Read [the input contract](references/input-contract.md) when adapting saved tables.

1. Establish the exact cohort, modality/dictionary, discovery split, outcome scope
   and testing family. Reuse saved effects, confidence intervals, p-values and
   cached interpretations. Do not retrain, run new interpretation calls or search
   literature unless separately requested. Label illustrative data as synthetic.
2. Include the complete supplied concept space, with nonsignificant/untested nodes
   in gray and significant nodes colored by their strongest significant outcome.
   Do not restrict the map to interpreted, significant or hand-picked concepts.
   Reuse a complete saved layout when available. Otherwise compute a deterministic
   layout from Discovery features and save it. Keep constant features visible in a
   labelled strip rather than assigning them spurious coactivation geometry.
3. Run `scripts/build_report.py` from this skill folder with either `--run RUN_DIR`
   or `--concepts TABLE --associations TABLE`, plus `--output NEW_DIR`. Choose the
   appropriate split and comparison split explicitly. Never infer Bonferroni
   correction from the displayed nodes: require the stored corrected p-values.
4. For a study-quality brief, include saved cohort/event counts, dictionary size
   and k, validation and held-out reconstruction R², and input-vector versus
   full-dictionary predictive probes when available. Pass `--literature STUDY_DIR`
   for the comparison runner's outputs. Report AUC with uncertainty and paired
   differences, calibration, and equal-count association estimates separately.
   Do not imply that no improvement, missing data or very wide CIs are successes.
   Optional `--annotations TABLE` adds training-selected measurement correlations
   with held-out checks; these are not semantic interpretations or accuracy scores.
5. Inspect `facts.json`, `report.md`, the full map and the PDF. Keep the text brief
   (normally about 200 words), factual and readable. Include two zoomed examples,
   exact effects/CIs/corrected p-values, comparison-split evidence if available, and
   cached interpretation suggestions with their evaluation accuracy when supplied.
   Interpretation labels are suggestions, not verified biology or proof of novelty.
6. Check every displayed statistic against the saved input row, confirm every
   supplied concept appears on the map, and inspect the rendered pages for clipping
   or illegible captions. State missing evidence and scope without inventing
   statistics. Do not call an unadjusted association causal. Call reproduction only
   when the same-direction, corrected comparison estimate passes the supplied gate.

Outputs: short Markdown, a one- or two-page PDF plus an optional activating-image page,
full-space and compact zoom PNG/SVG figures, optional literature panels, and
machine-readable facts and provenance. Make nodes readable at normal PDF size;
scale area by log(1 + activation count) when available. Zooms should contain a few
local neighbors and occupy small panels, rather than reproducing broad map regions.
Use `--preview-html` only when a browser preview is useful. It is a static report,
not a replacement for the interactive explorer. After installing this skill, agents
can run the script directly; `glyph skills list --json` locates the bundled folder.

Example:

```bash
python <skill-folder>/scripts/build_report.py --run /path/to/saved/run \
  --split train --comparison-split test --output /path/to/new/brief
```

For a reader integration test, use public or synthetic new inputs, optionally with
released pretrained models and dictionaries. Do not use the authors' research
cohorts. Identify the dataset and full testing family. Never present a synthetic
smoke test as scientific findings. See `facts.json` for the deterministic zoom-selection rule.

Autointerpretation may be explicitly deferred. State that once, omit empty
interpretation boxes, and do not substitute correlation labels for semantic
validation. Use source citations in the report itself, not just an input manifest.
The paper's outcome-specific thresholds and cohorts are reference standards,
not constants to copy into a new dataset.

Use `glyph harness report` after pip installation; scripts and references are bundled.
`--interpretations trusted_results.pkl` includes a separately generated held-out
interpretation stage. A literature-study output also carries the reviewed search
funnel and source citations. Optional adjusted estimates stay in a separate table;
labels on the ordinary association panel must continue to say unadjusted.

For saved runs with local 2D image paths, add `--image-examples` to include an
additional PDF page and PNG of actual activating examples. Concept selection
uses training associations only, and image selection uses activation ranks
without outcome labels. Training and held-out images are marked separately;
exact sample IDs, activation values and source hashes are recorded in facts.
These panels are image evidence, not semantic validation or an LLM interpretation.
