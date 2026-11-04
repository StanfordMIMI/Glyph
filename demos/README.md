# Glyph demos

These examples start with public raw images, extract official DINOv3 embeddings,
learn an **8,192-entry dictionary**, test labelled outcomes and produce an
interactive explorer and a PDF with actual activating sample images.

| Demo | Notebook | PDF | Interactive HTML |
| --- | --- | --- | --- |
| [Smartphone skin-lesion diagnosis](pad_ufes/README.md), 1,373 patients | [Open](pad_ufes/demo.ipynb) | [Open](pad_ufes/report.pdf) | [Download](pad_ufes/explorer.html) |
| [Chest X-ray mortality](sbu_cxr/README.md), 948 patients | [Open](sbu_cxr/demo.ipynb) | [Open](sbu_cxr/report.pdf) | [Download](sbu_cxr/explorer.html) |
| [Handwritten digits](dinov3_digits/README.md), 1,000 images | [Open](dinov3_digits/demo.ipynb) | [Open](dinov3_digits/report.pdf) | [Download](dinov3_digits/explorer.html) |

GitHub renders notebook images and PDFs. Download the self-contained HTML and
open it locally for interaction. GitHub's file viewer does not execute JavaScript.

The skin-lesion demo uses default scientific settings and compares with eight
published clinical fields. Its endpoint is contemporaneous lesion malignancy.
The chest X-ray example predicts later in-hospital death using an admission-time
image and compares with an age/sex literature baseline. Both medical examples
have patient-held-out internal tests and no external clinical validation. See each demo for sources,
limitations, installation and rerun instructions. No private research cohort or
LLM key is used. DINOv3 requires approved Hugging Face access.
