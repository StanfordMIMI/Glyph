# Measure published concepts on a reader cohort

```python
from glyph import Glyph
ct = Glyph.from_pretrained('ct-merlin')
i87 = ct.measure('I87', metadata='ct_metadata.csv')
one = ct.measure('I87', metadata='ct_metadata.csv', sample_id='scan_001')
space = ct.measure(metadata='ct_metadata.csv')
ct.visualize()
i87 = space['I87']
```

`from_pretrained` loads the SAE. The matching encoder is lazy and runs only when
raw inputs are measured. Cohort metadata needs unique string sample IDs and input
paths. There is no SAE training or train/test split requirement. Repeated calls
reuse already measured studies; omitted studies are encoded when requested.

| Bundle | Input column | Required FM | Concept codes |
| --- | --- | --- | --- |
| `ct-merlin` | `image_path` or `ct_path` | Merlin | `I...` |
| `findings-qwen3` | `report_path` or `findings_path` | Qwen3-Embedding-8B | `R...` |
| `labs-loinc` | `labs_path` | Released labs encoder | `L...` |

Use `input_col=` for another column name. All columns point to local files: CT
NIfTI, report text, or per-study prepared lab JSON. Raw dated LOINC values first
need the lab preparation helper with the published mappings and units. Do not
invent lab tokens or infer units from a code alone.

`measure('I87')` returns a pandas Series indexed by sample ID. Omit the code for
an entire DataFrame with 8,192 original concept columns. Raw zeros are retained.
The graph displays active cohort concepts, original codes and available published
interpretations; it does not establish association significance without labels.
With fewer than three measured studies, correlation edges are not estimated.

Filters refer to the user's own metadata. `sample_id=` selects a single study;
`split=` requires a matching metadata column. Without filters all supplied
studies are measured. Cohort association analyses can use the configured
`Glyph(...).fit(pretrained={column: bundle}).hypothesis_search()` route.

Supply `embeddings_path=` to `measure` to use aligned raw FM embeddings instead of
loading the encoder. `recompute=True` creates a fresh measurement run. New input
cohorts also get separate runs. Load a trusted saved run with
`Glyph.load_run(run_dir)`, then call `measure()` without downloading an FM.

The dictionary, normalization and feature identity remain fixed. New-cohort
embedding-magnitude diagnostics are recorded and warn by default; request
`strict=True` in `from_pretrained` for rejection. Do not normalize a new cohort
until it artificially passes the check. Interpretation or clinical transfer is
not established by successful software execution or embedding magnitude.

Keep the original matching FM: DINOv3 and the small Qwen preset are choices for
learning new dictionaries, not replacements for the paper's Merlin/8B models.
