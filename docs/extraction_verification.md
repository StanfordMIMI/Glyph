# Extraction and saved-run verification — 22 October 2026

Implementation and test artifacts are in Glyph. The labs foundation encoder
and reviewed mappings have now been published to HF; see the release verification
section below. No pip package or Git repository changes were pushed.

## Implemented

- Fresh raw-input extraction by default; explicit embedding files skip encoder loading.
- Separate extraction API/CLI with Merlin, Qwen3, and labs adapters referenced to
  HypothesisExtractor and the original research implementation.
- Main-API published SAE application with matching encoders, pinned SAE revisions,
  published normalization, original IDs and descriptions; no SAE retraining.
- Unique fitted runs, complete trained SAE checkpoints and normalization, saved
  embeddings/activations/outcomes, failed-run status, and exact reload inference.
- Strict labs checkpoint bundle preparation, CPU loading, JSON training metadata,
  LOINC mapping, checksums and source hashes; weights excluded from the wheel.

## Tests

The earlier extraction source suite and independently installed wheel each passed
**119 checks**, with the opt-in raw CT integration test skipped in those runs. That
real CT test was run separately and passed its documented cross-runtime
acceptance. Tests include pretrained application through association search,
interactive HTML, and reload, as well as raw extraction, explicit embedding
alignment, repeated fits, checkpoint round trips and encoder-specific handling.

The inspected wheel contains the extraction modules, run persistence, and three
agent skills. It contains no `.pt`, `.pth`, `.ckpt` or `.safetensors` files.
Earlier extraction wheel SHA-256: `da2801b4facb5690f50bf2c4e9a68ebb9268c62bbb115f82493c8ca4c8eac89d`.

### One real CT, existing cache and staged released activations

- Active concept IDs matched both comparison sources exactly.
- Embedding cosine similarity: 0.999998093.
- Embedding relative L2 error: 0.2008%.
- Activation relative L2 error against cached-embedding inference: 0.4356%.
- Maximum activation absolute difference: 0.071819.
- Strict numerical parity: **not achieved**. The cause of the remaining drift
  has not been isolated; identical concept IDs do not establish identical values
  or full-cohort parity. Cross-runtime acceptance requires exact active IDs,
  cosine >= 0.99999, and relative L2 errors <= 0.5%.
- SAE revision: `23c5da2d6d415e88255912da398e83035d507069`.

No private study IDs, image paths, or patient rows are included here. The opt-in
case and detailed logs remain in ignored local test outputs.

### Real labs checkpoint and inputs — updated

All 743 trained tensors match the original research checkpoint. The current
release candidate contains only the 350 retained numeric model labs, their units,
normalization statistics and LOINC mappings, without patient rows.

On 20 studies, prepared tokens match the research cache exactly. Embedding relative
L2 error is 0.0568%; 18 studies have identical active concept sets and two share
39/40 concepts. The swapped concepts occupy ranks 40 and 41, within the observed
numerical score perturbation. Sparse activation relative L2 error is 1.49%.
The explicit boolean padding mask makes fastpath-on/off inference consistent.
See [labs_model_card.md](labs_model_card.md) for preprocessing details and the
quarter-start timing limitation. The aggregate report is
`test_outputs/labs_release_parity_report.json`.

## Remaining for this extraction release

1. Complete underlying training-cohort version provenance if available. Labs
   weights are now published under MIT and pinned in the package.
2. Run a real Qwen3 encoder inference/cache comparison. Its preprocessing and
   pooling have been tested with a controlled model, not the full 8B checkpoint.
3. Expand real CT checks beyond one study and investigate numerical drift if
   stricter value parity is needed.
4. Complete underlying encoder revision provenance. Public SAE assets are
   currently encoder-only; decoder weights are needed to evaluate published
   reconstruction R² on a new cohort.
5. Publish a new package release after the remaining integration/release checks.

See [extraction.md](extraction.md) for usage and [labs_model_card.md](labs_model_card.md)
for the labs model card.

## Follow-up investigation of CT embedding drift

The freshly preprocessed CT tensor is bit-for-bit identical to its historical
MONAI image cache: same float32 dtype, zero differing voxels. The research and
merlin-vlm checkpoint files have identical SHA-256 hashes.

Using that exact tensor and checkpoint, the original research image encoder and
the merlin-vlm image encoder produce bit-for-bit identical outputs on CPU
(maximum difference 0). Both differ from the historical cached embedding by the
same maximum absolute value, 0.0031975507736206055. The comparison used PyTorch
2.6.0+cu124 with no CUDA device available.

This rules out preprocessing and the package encoder port as the explanation in
this case under the tested environment. The leading explanation is numerical
differences in the historical GPU execution, potentially TF32 convolution,
batching, or software/kernel versions. The exact historical cause is not proven;
a GPU comparison with controlled precision settings is still needed.

## Bug-fix follow-up

See [bugs_and_labs_2026-10-22.md](bugs_and_labs_2026-10-22.md) for the subsequent
bug fixes and final installed-wheel verification. The earlier 119-check result
and wheel hash above describe the extraction-only checkpoint, not this later build.

## Public HF labs release

Published under MIT at [RobbieHolland/Glyph-labs-encoder](https://huggingface.co/RobbieHolland/Glyph-labs-encoder).
Pinned revision: `972acbaea6d7ea2249a84d562a5f6dbabb994827`. The repository includes
the full encoder, 350-lab training statistics and units, reviewed September mapping,
later source audit, provenance, documentation, checksums and license. No patient
records are included.

The opt-in `tests/test_labs_hf.py` test uses a fresh cache with authentication
disabled. It downloaded the public bundle, validated all manifest assets, prepared
synthetic raw LOINC measurements, and extracted embeddings through the package API.
Downloaded and local-release embeddings matched exactly. It then anonymously
downloaded the released labs SAE and produced an 8192-dimensional concept vector
with 40 active concepts. Synthetic inputs verify the software path rather than
clinical performance. The HF integration plus extraction/mapping source tests
passed 31 checks. This test is included in the repository and skips by default
unless `GLYPH_HF_INTEGRATION=1` is set.

The independently installed release wheel also passed all 38 selected tests,
including a second fresh anonymous HF download and the packaged skill checks.
Wheel SHA-256: `ccae35c7a0fa40cf96bfd00a542aeec6c63c1936a038820c25b250940809207d`.


## Reader API follow-up — 23 October

The new `from_pretrained`/`measure` workflow has now been exercised using actual
released encoders and dictionaries for an independent public CT and synthetic
new labs/reports, including the full 8B findings encoder. See
[reader verification](reader_verification.md) for current evidence, installation
fixes and the completed real-DINOv3 test.
