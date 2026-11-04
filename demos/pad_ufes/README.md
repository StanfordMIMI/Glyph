# Smartphone skin-lesion diagnosis

A third demo, and a second medical-image application, using original **PAD-UFES-20** smartphone photographs rather than radiographs. One photograph per patient gives **1,373 independent patient observations** with contemporaneous cancer and lesion-diagnosis labels.

- [Executed notebook](demo.ipynb)
- [PDF report](report.pdf), including one page per selected concept and actual activating photographs
- [Interactive explorer](explorer.html) — download and open locally
- [Expandable HTML report](report_files/report.html)
- [Saved associations](associations.csv), [source provenance](provenance.json) and [frozen clinical baseline](literature/manifest.json)

## Out-of-the-box test

The run started from raw images using an independently installed Glyph wheel and the real official DINOv3 checkpoint. Scientific settings were left at their defaults: 8,192 dictionary entries, Matryoshka levels `[128,512,2048,8192]`, sparsity candidates `[5,10,20,40,80]`, 1,500 training steps per candidate and the default association/graph settings. Runtime conveniences were batch size 16, zero loader workers and four CPU threads; automated interpretation was explicitly disabled.

| Check | Result |
| --- | --- |
| Patient split | 961 train / 206 validation / 206 test |
| Cancer cases | 417 train / 89 validation / 89 test |
| Dictionary / training-alive concepts | 8,192 / 1,600 |
| Selected k | 80 |
| Training / validation / test reconstruction R² | 0.965 / 0.757 / 0.759 |
| Corrected discovery associations across seven outcomes | 164 |
| Same-direction corrected test reproductions | 17 / 138 evaluable discoveries |
| Saved-run activation reload | Exact for all three splits |

**The default search did not reach the 0.85 reconstruction target.** Glyph warned and retained its best tested configuration. This demo keeps that result; it does not widen the search after inspecting held-out performance.

| Tried k | Validation R² |
| --- | ---: |
| 5 | 0.4348 |
| 10 | 0.5310 |
| 20 | 0.6085 |
| 40 | 0.6879 |
| 80 | 0.7570 |

## A proper clinical comparator

[Pacheco and Krohling](https://arxiv.org/html/1909.12912), *Computers in Biology and Medicine* 2020, [doi:10.1016/j.compbiomed.2019.103545](https://doi.org/10.1016/j.compbiomed.2019.103545), specify eight clinical inputs: **age, body site, itching, bleeding, pain, growth, change and elevation**. Those exact fields are available. The 15 predefined site indicators give 22 columns in total, including one site absent from this release. Unknown symptoms stay missing; imputation, normalization, ranking and coefficient fitting use training data. Validation chooses regularization and early-stopping checkpoints. Diagnosis, biopsy status and patient IDs never enter the predictors.

This reconstructs the published clinical inputs and fits a new ridge model. It does not reproduce the historical neural classifier, a validated risk score or its published performance. The targeted primary-source lookup is recorded in the [search protocol](literature/protocol.json), [screening ledger](literature/screening.csv) and frozen manifest. It is a worked benchmark, not a systematic literature review.

The overview’s diagnostic probe uses a separate AdamW tuning procedure; the literature panel uses the common ridge comparison engine. Their AUCs are separate fits.

The primary comparison is **cancer in the imaged lesion (BCC/SCC/MEL) versus ACK/NEV/SEK**. All six numerical arms are retained below and in the CSV files; the compact report omits the full combined row, as in the other demos.

| Predictor set | Test AUROC (95% CI) |
| --- | ---: |
| Clinical literature | 0.939 (0.903–0.967) |
| Clinical top 10 | 0.946 (0.915–0.971) |
| Glyph 8,192 | 0.888 (0.842–0.928) |
| Glyph top 10 | 0.822 (0.761–0.873) |
| Clinical + Glyph (all) | 0.963 (0.936–0.984) |
| Clinical + Glyph pooled top 10 | 0.922 (0.882–0.955) |

The combined top-ten arm uses `combined_top_n_strategy="pooled"`: rank clinical and Glyph columns together on training only and select ten total. This is explicit because the compatibility default retains clinical columns first, which would leave no Glyph slots with a large clinical pool. A pooled selection may contain either or both sources; inspect [feature rankings](literature/training_rankings.csv). Missingness indicators accompany selected measurements and are not extra ranked candidates. [Paired comparisons](literature/auc_comparisons.csv) use 2,000 patient bootstrap replicates with fixed fitted predictions; p-values are exploratory and unadjusted.

## Rerun

Install from the repository with `pip install -e '.[hf]'` and obtain approved access to [DINOv3](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m). Then run the notebook, or:

```python
from pathlib import Path
from examples.pad_ufes import prepare, freeze_baseline, make_glyph, literature_study, report

work = Path("demo_outputs/my_pad_run").resolve()
metadata = prepare(work / "inputs")
frozen = freeze_baseline(work)  # before fitting or evaluating outcomes
model = make_glyph(metadata, work / "runs")
model.fit()
model.hypothesis_search()
model.visualize(split="train")
study = literature_study(model, work / "inputs", work / "clinical_comparison",
                         frozen, work / "literature_search")
report(model.run_dir, work / "report_files", study)
```

Original downloads total about **3.6 GB** and are verified against the publisher's SHA-256 checksums. Archives, raw images, weights and activations stay in ignored local output folders. The notebook includes embedded figures that GitHub can display. Nothing is uploaded automatically.

## Scope

These are **diagnostic labels, not future outcomes**. The source has 2,298 images, 1,641 lesions and 1,373 patients; selecting the lexicographically first image ID without using diagnosis prevents repeated-image inflation. Every cancer label is biopsy-confirmed, while some non-cancer labels use clinical consensus. Six secondary one-vs-rest diagnoses are tested alongside the composite cancer endpoint, with the complete saved multiple-testing family. Melanoma has only 22 training, four validation and five test cases, so its results are underpowered. There is no external validation or confounder adjustment. Image examples and associations are not validated semantic interpretations or proof of new biology.

Images and metadata: Pacheco et al., **PAD-UFES-20 v1**, [doi:10.17632/zr7vgbcyr2.1](https://data.mendeley.com/datasets/zr7vgbcyr2/1), [**CC BY 4.0**](https://creativecommons.org/licenses/by/4.0/). [Dataset paper](https://doi.org/10.1016/j.dib.2020.106221). Report images are resized for display; input photographs were passed directly to the official DINOv3 processor without task-specific segmentation or cropping.
