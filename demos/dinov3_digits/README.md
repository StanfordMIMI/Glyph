# DINOv3 → Glyph: handwritten digits

1,000 public handwritten digit images with ten digit-identity outcomes. Raw images are encoded by the official pretrained DINOv3 model, then a new dictionary is fitted.

The demos use **8,192 dictionary entries**, matching Glyph’s default width. No author research cohort or LLM calls are used.

- [Executed notebook](demo.ipynb)
- [Interactive explorer](explorer.html) — download and open locally
- [PDF report with activating images](report.pdf)
- [HTML report with expandable activation tails](report_files/report.html)
- [Markdown report](report_files/report.md)
- [Additional concept image examples](concept_examples.png)
- [Saved associations](associations.csv) and [provenance](provenance.json)

| Check | Result |
| --- | --- |
| Dictionary / training-active concepts | 8,192 / 1,629 |
| Selected active concepts per image, k | 64 |
| Validation / test reconstruction R² | 0.868 / 0.864 |
| Corrected discovery associations | 680 |
| Same-direction corrected test reproductions | 24 / 545 evaluable hits |
| Exact saved activation reload | Passed with the recorded CPU settings |

The full-width dictionary retains every original feature index. The map shows all training-active concepts; inactive entries are not plotted. The PDF includes actual highest-activating training and held-out images, selected without outcome-based image selection. No semantic interpretation or interpretation accuracy is claimed.

## Sparsity selection

| k | Validation R² |
| --- | ---: |
| 8 | 0.7485 |
| 16 | 0.7970 |
| 32 | 0.8337 |
| 64 | 0.8677 |
| 128 | 0.8976 |
| 192 | 0.9152 |

The first tested 0.85 crossing is k=64. The 0.90 and 0.95 targets permit k up to 128 and 192, respectively. The rule selects k=64. This CPU example uses a coarse grid; these are the smallest qualifying **tested** values. The initial encoder gain is calibrated on training reconstruction only; validation selects the configuration.

## Data and interpretation

Splits are 700 training / 150 validation / 150 test images. Labels are digit identities; this is a software demonstration, not a clinical benchmark. Writers are not explicitly separated. Some individual held-out concept estimates are unstable or have wide intervals. The report’s single probe-AUC line is for digit 0 versus the rest; all ten outcome probes are retained.

Source: [scikit-learn optical digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html). Selection and splits use seed 42. Original 8×8 pixels are converted to RGB before DINOv3 preprocessing.

## Rerun

From a Glyph checkout, install `pip install -e ".[hf]"` and open `demos/dinov3_digits/demo.ipynb`. The recorded-run branch reloads a real local run when available; a fresh checkout computes embeddings and fits a new dictionary.

The official encoder is [`facebook/dinov3-vits16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m), pinned to `114c1379950215c8b35dfcd4e90a5c251dde0d32`. Approved HF access and a configured login are required. The CPU profile uses 2,000 training steps and eight Torch threads. No model weights, full activation matrices or raw input cache are committed here. The notebooks embed their displayed images for GitHub. Download HTML to interact with it.

The PDF is directly in this demo folder. Supporting figures, per-concept pages and report facts are in `report_files/`. Each selected concept is presented with its local map, exact statistics and activating images together.

Node area directly represents activation prevalence, with a tiny visibility floor and a common density adjustment for large maps. Correlation lines are shown by default with at most eight per concept; the HTML display options can hide them.

Each concept page shows its whole activation distribution, a zero example and the 33rd/66th/100th percentiles among activating training samples, followed by a high-activation tail and four strong train/test examples. Correlation widths and opacity scale down with absolute correlation.

Concept pages include a miniature whole-space map, a boxed selected region linked to its zoom, color-matched connections from activation histograms to example images, and a highlighted tail linked to its magnified detail.

Digits 0–9 use a fixed ordered hue gradient at equal HSLuv lightness/saturation, with a numeric color strip. All ten outcomes are shown separately; no outcomes are grouped as “Other.”

The two illustrated pages focus on separated digit-4-associated concepts, C1978 and C5058. Training-image review selected nearby, higher-prevalence replacements for the initial pair: 85/700 (12.1%) and 108/700 (15.4%) active training images, respectively. Within-digit-4 activation correlation is −0.512 and map distance is 0.238. The angular/upright versus broader/open-bowl descriptions are unvalidated visual observations. [Selection profile](report_profile.json). Fresh runs select a complementary pair from their own learned dictionary; the curated profile applies only to this saved run.
