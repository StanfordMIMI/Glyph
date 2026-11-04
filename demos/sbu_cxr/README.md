# Medical prognosis from public chest X-rays

948 distinct public TCIA patients, with 164 later in-hospital deaths. One admission-period frontal chest X-ray per patient is paired with a subsequent hospital outcome.

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
| Dictionary / training-active concepts | 8,192 / 8,192 |
| Selected active concepts per image, k | 768 |
| Validation / test reconstruction R² | 0.905 / 0.902 |
| Corrected discovery associations | 46 |
| Same-direction corrected test reproductions | 1 / 31 evaluable hits |
| Exact saved activation reload | Passed with the recorded CPU settings |

The full-width dictionary retains every original feature index. The map shows all training-active concepts; inactive entries are not plotted. The PDF includes actual highest-activating training and held-out images, selected without outcome-based image selection. No semantic interpretation or interpretation accuracy is claimed.

## Sparsity selection

| k | Validation R² |
| --- | ---: |
| 8 | 0.4674 |
| 16 | 0.5376 |
| 32 | 0.6048 |
| 64 | 0.6819 |
| 128 | 0.7587 |
| 256 | 0.8241 |
| 512 | 0.8759 |
| 640 | 0.8912 |
| 768 | 0.9045 |
| 896 | 0.9140 |
| 1024 | 0.9213 |
| 1152 | 0.9273 |
| 1280 | 0.9328 |
| 1408 | 0.9377 |
| 1536 | 0.9413 |

The first tested 0.85 crossing is k=512. The 0.90 and 0.95 targets permit k up to 1024 and 1536, respectively. The rule selects k=768. This CPU example uses a coarse grid; these are the smallest qualifying **tested** values. The initial encoder gain is calibrated on training reconstruction only; validation selects the configuration.

A separate k=64 diagnostic with a 4,000-step budget stopped early at step 1,680. Its best validation R² was **0.682**, essentially the original k=64 result. It retained 3,390 training-active concepts, so lower per-image k still leaves a larger map than the digits example. This check used no held-out test inputs or labels and did not replace the demo. [Diagnostic record](sparsity_diagnostic.json). The current demo trains only the 8,192-entry level; it does not train the package's default nested 128/512/2,048/8,192 levels.

## Held-out prognosis comparison

The test set has **143 patients and 25 deaths** (training 663/115; validation 142/24). Rankings and preprocessing use training data only. Every predictor candidate restores its best validation-AUROC checkpoint, with validation log loss breaking ties and patience of 20 checks. Combined models independently regularize the literature and Glyph blocks, beginning with the fitted literature baseline. Test outcomes do not choose features, checkpoints or penalties.

| Model | Predictors | Test AUC | 95% patient-bootstrap interval |
| --- | ---: | ---: | --- |
| Literature | 3 | 0.839 | 0.771–0.906 |
| Literature top 10 | 3 | 0.839 | 0.771–0.906 |
| Glyph | 8,192 | 0.783 | 0.661–0.893 |
| Glyph top 10 | 10 | 0.767 | 0.625–0.880 |
| Literature + Glyph to 10 | 10 | 0.932 | 0.877–0.975 |

The capped combination retains all three available literature predictors and adds seven training-ranked Glyph concepts. Its ΔAUC versus literature is +0.093 (95% CI +0.031 to +0.162; paired p=0.0070, exploratory and unadjusted). Only three literature predictors exist here, so its full and top-ten variants coincide. The compact report omits the all-variable combined row; complete numerical results remain in [metrics](literature/metrics.csv) and [paired comparisons](literature/auc_comparisons.csv).

[Training rankings](literature/training_rankings.csv), [feature selections](literature/feature_selections.json), and [validation traces](literature/tuning.csv) document selection. Individual-marker associations remain a separate matched three-versus-three comparison with six-test correction. This is a small, single-site internal test.

The source is [Williamson et al., Nature 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7611074/). Age-band proxies and sex form a simplified literature-informed baseline, **not the published fitted risk model**. The retrieval is a targeted primary-paper lookup, not a systematic review. Comorbidity flags were omitted because their diagnosis timestamps are unavailable.

## Cohort and timing

The source collection has 1,384 patients. This analysis requires a selected inpatient visit, a stay spanning at least two calendar days, and a frontal chest radiograph on the day before or day of admission. One image per patient is selected by earliest eligible date, then series UID. The landmark is the beginning of hospital day two; later in-hospital death is positive and discharge is negative. Dates are shifted, exact death timestamps are unavailable, and ICU/ventilation flags are not asserted to be future outcomes. All images from a patient remain in one split.

Source: Saltz et al. (2021), [TCIA COVID-19-NY-SBU](https://www.cancerimagingarchive.net/collection/covid-19-ny-sbu/), [DOI 10.7937/TCIA.BBAG-2923](https://doi.org/10.7937/TCIA.BBAG-2923), CC BY 4.0. DICOM pixels are normalized using non-padding 0.5/99.5 percentiles, converted to RGB and padded to a 512-square image while preserving aspect ratio. DINOv3 performs its standard 224-square preprocessing.

## Rerun

Use Python 3.10+ and a Glyph checkout:

```bash
pip install -e ".[hf]" "pydicom>=3.0" "idc-index-data==25.0.5"
```

Open `demos/sbu_cxr/demo.ipynb`. First preparation transfers about 14 GB of public DICOM data. Only the converted PNGs and provenance are retained locally; raw DICOM responses are discarded. A saved-run branch makes local review fast; a fresh checkout runs extraction and fitting.

The official encoder is [`facebook/dinov3-vits16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m), pinned to `114c1379950215c8b35dfcd4e90a5c251dde0d32`. Approved HF access and a configured login are required. The CPU profile uses 2,000 training steps and eight Torch threads. No model weights, full activation matrices or raw input cache are committed here. The notebooks embed their displayed images for GitHub. Download HTML to interact with it.

The PDF is directly in this demo folder. Supporting figures, per-concept pages and report facts are in `report_files/`. Each selected concept is presented with its local map, exact statistics and activating images together.

Node area directly represents activation prevalence, with a tiny visibility floor and a common density adjustment for large maps. Correlation lines are shown by default with at most eight per concept; the HTML display options can hide them.

Each concept page shows its whole activation distribution, a zero example and the 33rd/66th/100th percentiles among activating training samples, followed by a high-activation tail and four strong train/test examples. Correlation widths and opacity scale down with absolute correlation.

Concept pages include a miniature whole-space map, a boxed selected region linked to its zoom, color-matched connections from activation histograms to example images, and a highlighted tail linked to its magnified detail.
