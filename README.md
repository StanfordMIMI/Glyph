<p align="center">
  <img src="Glyph.svg" alt="Glyph" width="100%">
</p>

# Glyph

Glyph automates discovery in foundation-model embeddings. It learns sparse
concepts, searches for associations with outcomes, and summarizes its findings
in interactive visualizations and concise reports. Use it on your own data,
or apply the pretrained concept dictionaries released with the paper.

## Demos

See Glyph run on public images, with executed notebooks, interactive concept maps
and illustrated reports.

| Demo | Notebook | PDF report | Interactive HTML |
| --- | --- | --- | --- |
| [Skin-lesion diagnosis](demos/pad_ufes/README.md) · 1,373 patients, clinical baseline | [Open notebook](demos/pad_ufes/demo.ipynb) | [View report](demos/pad_ufes/report.pdf) | [Get HTML](demos/pad_ufes/explorer.html) |
| [Chest X-ray mortality](demos/sbu_cxr/README.md) · 948 patients, clinical baseline | [Open notebook](demos/sbu_cxr/demo.ipynb) | [View report](demos/sbu_cxr/report.pdf) | [Get HTML](demos/sbu_cxr/explorer.html) |
| [Handwritten digits](demos/dinov3_digits/README.md) · 1,000 images, ten digit labels | [Open notebook](demos/dinov3_digits/demo.ipynb) | [View report](demos/dinov3_digits/report.pdf) | [Get HTML](demos/dinov3_digits/explorer.html) |

GitHub displays the notebooks and PDFs. Download an HTML file and open it in your
browser to use the interactive explorer.

## Install

From a checkout of this repository:

```bash
pip install -e ".[hf,zoo]"
```

For the paper's pretrained dictionaries, install the matching encoder extra:
`pip install -e ".[zoo,ct]"`, `".[zoo,text]"`, or `".[zoo,labs]"`.
Model weights download when needed. DINOv3 requires approved access to its
[HF checkpoint](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).

## Learn new concepts

```python
from glyph import Glyph

glyph = Glyph(config_path='config', config_overrides=[
    'data.metadata=/path/to/metadata.csv',
    'data.input_cols=[image_path]',
    'data.outcome_cols=[pneumonia]',
    'data.embedding_map={image_path: "hf:facebook/dinov3-vits16-pretrain-lvd1689m"}',
    'data.embedding_model_kwargs={image_path: {input_type: image, pooling: cls}}',
])

glyph.fit()
glyph.hypothesis_search()
glyph.visualize(outcome='pneumonia', split='test')
```

The metadata supplies study IDs, image paths, and outcome labels. `fit` favors
sparse SAE configurations using validation reconstruction R² and warns below
0.85. `hypothesis_search` tests concept–outcome associations; `visualize` shows
an interactive concept space with association statistics and significant
concepts highlighted. Node area reflects activation prevalence, and correlation
links are visible by default with at most eight per concept. Confounder adjustment and automated interpretation are
optional. Labels are not required for learning and exploring concepts.

For a small text encoder, use `qwen3-embedding-0.6b`. Other HF models and your
own models can specify their preprocessing and embedding output; see the
[encoder guide](docs/hf_embeddings.md).

## Extract the paper's concepts on your data

```python
from glyph import Glyph

# 1. EXTRACT A SINGLE PUBLISHED CONCEPT ON YOUR DATA
ct = Glyph.from_pretrained('ct-merlin')
i87 = ct.measure('I87', metadata='ct_metadata.csv')

# Optionally select one study by its ID in your metadata.
i87 = ct.measure('I87', metadata='ct_metadata.csv', sample_id='scan_001')

# 2. EXTRACT THE FULL PRETRAINED CONCEPT SPACE ON YOUR DATA
labs = Glyph.from_pretrained('labs-loinc')
concepts = labs.measure(metadata='labs_metadata.csv')
labs.visualize()

# Select an individual concept from the measured space.
l127 = concepts['L127']
```

`ct_metadata.csv` contains `sample_id` and `image_path` (CT NIfTI paths).
`labs_metadata.csv` contains `sample_id` and `labs_path` (prepared laboratory JSON
paths); see [lab preparation](docs/extraction.md#labs-release-bundle) for raw LOINC data.

`from_pretrained` loads the published SAE. `measure` computes the matching
foundation-model embeddings on your data, then applies that SAE. No labels,
SAE training, or train/test splits are needed. Repeated requests reuse measured
studies; missing studies are encoded when requested.

A single concept returns a Series indexed by study ID. Whole-space extraction
returns all 8,192 original concept columns in a DataFrame. The explorer shows
concepts active in your cohort and available published interpretations; labels
are needed for association testing and significance highlighting.

Bundles are `ct-merlin`, `findings-qwen3`, and `labs-loinc`. Each preserves the
paper's normalization and concept indices and uses the original matching encoder.
See [extraction and saved runs](docs/extraction.md) for supplied embeddings,
reloading, and input options.

## Run through a coding agent

The package includes skills for using Glyph, literature comparisons, interactive
HTML, and brief visual reports. Install them for your coding agent:

```bash
glyph skills install --agent claude   # or --agent codex
glyph skills list
```

Then ask, for example:

```text
Run Glyph on my cohort.
Metadata: /path/to/metadata.csv
Inputs: image_path
Outcome: pneumonia
Encoder: hf:facebook/dinov3-vits16-pretrain-lvd1689m, using CLS pooling.

Report reconstruction R², draw the concept map, and list associated concepts
with odds ratios and Bonferroni significance. Compare against source-backed
literature markers and produce a brief visual report.

If my inputs match a modality in the paper, apply its pretrained concepts with
the matching encoder instead of fitting a new dictionary, and assess the
associations in my cohort.
```

Interpretation through Gemini/Vertex and supplementary literature search through
Paperclip are optional and require configured access. Numerical analyses and
visualization do not require an LLM key.

[Documentation](docs/README.md) · [Agent harnesses](docs/literature.md) ·
[Remote HTML viewing](docs/html_explorer.md)
