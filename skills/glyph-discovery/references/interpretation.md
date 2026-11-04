# Optional automated interpretation

Glyph uses the current `google-genai` client for Gemini or Vertex. Install
`glyph-discovery[autointerp]` only when needed. No live model calls are needed to
fit concepts, test associations, search literature, or render saved results.

For the existing API, configure a modality-to-text-column mapping and a provider.
The Gemini key can come from `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or the environment
variable named by `autointerp.llm.gemini_api_key_env`. Vertex uses the configured
project or `GOOGLE_CLOUD_PROJECT`, plus application-default credentials. Keys
resolved from the environment are not copied into saved configuration snapshots.
`autointerp.enabled=false` disables calls even when credentials exist.

To interpret a previously fitted trusted run, supply a JSON configuration:

```json
{
  "provider": "gemini_api",
  "model": "gemini-2.5-flash",
  "api_key_env": "GEMINI_API_KEY",
  "metadata": "/path/to/metadata_with_text.csv",
  "text_map": {"image_path": "report"},
  "max_features": 20,
  "top_n": 5,
  "bottom_n": 2,
  "n_fit": 20,
  "n_test": 20,
  "top_pct": 0.05
}
```

```bash
glyph harness interpret --run trusted_run --config interpretation.json --output interpretations
```

`max_features` bounds the number of distinct features selected by training-only
association evidence. The same feature is interpreted once per modality/dictionary,
then reused across outcomes. Each feature needs enough activating and nonactivating
examples; very small cohorts may not support the requested scoring sample size.
The model sees balanced training examples and a shuffled, balanced held-out set.
No evaluation ground-truth labels are included in its classification prompt.

Scoring requires exactly one valid 0/1 prediction per held-out example. Missing,
duplicated or malformed predictions are unscored, never filled with zeros. Outputs
include sample counts, accuracy, balanced accuracy, sensitivity, specificity,
model name and scoring status. Insufficient held-out data have no accuracy value.
Interpretation accuracy measures discrimination of concept activators from controls,
not disease-prediction accuracy or biological correctness.

Saved interpretation pickles include prompts, examples and identifiers for local
reproducibility. They are not the public activation-release format. Only load
trusted local runs/results. The release preparer omits these sample-level files.

To include separately generated interpretations in a report:

```bash
glyph harness report --run trusted_run --interpretations interpretations/autointerp_results.pkl \
  --output findings_brief
```

The literature-study harness accepts `--interpretation-config` for the same optional
stage. No credentialed live interpretation was run during the local release checks;
provider requests and parsing are covered with controlled synthetic responses.
