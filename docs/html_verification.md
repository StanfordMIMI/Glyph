# HTML verification — 22 October 2026

The viewer retains the existing CKM/CORAL-derived renderer and uses the same map,
controls and all-outcome wheels. Changes are confined to Glyph.

Wheels retain every supplied finite outcome estimate. Up to 20 outcomes have
spoke labels by default; larger wheels keep all spokes with hover and keyboard
focus details. This threshold is configurable. Tooltips include outcome, OR,
95% CI, raw/Bonferroni p-values and support when supplied. CSV export retains
numeric precision. Search, clear selection, mobile controls, lazy edge rendering
and cursor-anchored zoom are exercised in the browser tests.

Validation:

- 159 focused source tests passed; the opt-in HF network test was skipped in this
  run because this change did not modify the model download path.
- 37 tests passed against an independently installed wheel, including Chromium
  browser checks and packaged skills.
- Browser cases cover 1, 4, 20, 21 and 150 outcomes, every spoke's focus tooltip,
  real hover, CSV downloads, outcome search, filtering, empty states, selections,
  pan/zoom in a nonsquare map, mobile layout and optional full-view rankings.
- Standalone files made no external HTTP requests and emitted no JavaScript errors.

Source and wheel test reports, screenshots and the wheel checksum are recorded in
`test_outputs/html-finish/`. The reusable browser test is
`tests/test_html_browser.py`; commands are in html_explorer.md.

The small review example is `examples/concept_explorer.html`; a 150-outcome example
is in `test_outputs/html-finish/concept_explorer_150.html`. Both contain synthetic
presentation data. A real-data end-to-end package run and user visual review are
the next integration step; these synthetic checks do not establish clinical results.
