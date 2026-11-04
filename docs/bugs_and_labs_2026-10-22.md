# Bugs and labs verification, 22 October 2026

Changes are confined to Glyph. Research inputs were read without modification.

Fixed probe checkpoint snapshots, model selection for negative R², missing-data
handling and prediction alignment; preserved concept IDs when resuming from CSV;
separated modality/dictionary result joins; corrected visualization correction
families; strengthened explicit embedding cache invalidation and numeric registry
version ordering. Automated interpretation now installs and uses the Google Gen AI
SDK consistently for both API-key and Vertex configurations.

Labs now has versioned preparation, unit-aware ambiguous LOINC resolution, strict
checkpoint loading, stable boolean padding masks, and a patient-free release
bundle. See labs_model_card.md for the exact preprocessing and limitations.

Verification: 141 focused source tests passed, followed by 14 bug-regression
tests including both SDK provider configurations. The final independently installed
wheel passed all 143 focused tests. Its test report is
`test_outputs/bugs-labs-wheel.xml`. No live LLM API calls were made.
The wheel includes all three skills and excludes model checkpoint files.
Wheel SHA-256: `a471747d429506c1c8ef190e80375499b60e5a741825b76e2ee0d887f2e85fa2`.
The real 20-study labs comparison passed; results are in
`test_outputs/labs_release_parity_report.json`. It shows exact input tokens,
0.0568% embedding relative L2 error, and 798/800 active concept slots shared with
the historical cache. The two swaps occur at the sparsity cutoff.

The labs bundle is now public at `RobbieHolland/Glyph-labs-encoder`, under MIT,
with pinned revision `972acbaea6d7ea2249a84d562a5f6dbabb994827`. It includes the reviewed
September mappings and source audit. No weights are included in the pip wheel.
See extraction_verification.md for anonymous download integration verification.
Harnesses and interactive HTML review remain separate follow-up work.
