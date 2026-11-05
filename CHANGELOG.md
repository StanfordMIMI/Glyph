# Changelog

## Unreleased

- Add optional readable outcome display names to explorers and reports while
  preserving original filter/export identifiers. Refresh the skin-lesion demo
  with full diagnosis names in its explorer, PDF, notebook and label guide.

## 0.2.0

- Move the source repository to StanfordMIMI/Glyph; keep the `glyph-discovery`
  installation name and `glyph` Python API.
- Learn sparse concepts from image/text encoders, including DINOv3 and
  Qwen3-Embedding-0.6B, supported HF models and custom embedding adapters.
- Measure single published I/R/L concepts or an entire pretrained dictionary on
  new data, with embedding extraction, stable study IDs and exact saved-run reloads.
- Add opt-in FMR orthogonal confounder adjustment with depth-2 nuisance models
  and an explicit CUDA option.
- Include interactive concept maps with activation-selected image examples and
  bundled agent skills for discovery, literature comparisons, HTML and reports.
- Use validation reconstruction to select sparsity and warn below the configured
  quality target. Keep model weights outside the Python distribution.
- Refresh installation instructions and package metadata for the current API.

Validation includes 49 successful installed-wheel API/regression checks and four
real-browser HTML checks. Real model weights were used on public/synthetic
inputs; a short SAE smoke-test budget does not establish scientific performance.
