# Glyph examples

Start with either executed notebook:

- [Medical prognosis from public chest X-rays](../demos/sbu_cxr/README.md), with an age/sex literature comparison. Helpers: [sbu_cxr.py](sbu_cxr.py).
- [Concept discovery in public digits](../demos/dinov3_digits/README.md). Helpers: [dinov3_digits.py](dinov3_digits.py).

Both use official DINOv3 weights and an 8,192-entry dictionary. They produce
associations, a self-contained HTML explorer and a PDF including activating images.

Install `pip install -e ".[hf]"` from this checkout. For the medical example also
install `pydicom>=3.0` and `idc-index-data==25.0.5` using Python 3.10 or later.
DINOv3 requires approved HF access and a configured HF login. Public raw inputs
and fitted models are cached locally under the ignored `demo_outputs/` directory.
No author research data or LLM API calls are used.

`concept_explorer.py` is a synthetic standalone visualization example.
`reader_workflows.py` retains older tabular, ResNet18 and synthetic-lab checks.
