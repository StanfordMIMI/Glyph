# Interactive HTML explorer

HTML is optional. `glyph.hypothesis_search()` computes associations without
creating HTML by default. Call `glyph.visualize(outcome='...', split='test')`
explicitly, including after `Glyph.load_run(...)`, or opt into automatic generation
with `visualization.auto_generate=true`. Hosting is a separate opt-in action.
Files are returned as paths and saved in the run's output directory. Each
modality/dictionary has its own map. Independently fitted dictionaries with a shared
layout can use `build_interactive_concept_atlas`. A narrative report belongs to an
agent report harness, rather than the HTML renderer or a required core API step.

The default simple view keeps the map, outcome selector, searchable concepts,
Options and concept details. Options cover association direction, extreme effects,
modalities/families, node color, connections and an optional restriction to concepts
with selected associations. Display filters change what is shown; they do not
recalculate effect estimates or multiple-testing corrections. Connections are
rendered only when enabled in the simple view. Search/filter counts and an empty
state make it clear when no concepts match. Escape or Back to map clears selection.

## High-activation image examples

For local 2D image inputs, `glyph.visualize()` embeds up to two highest-activating
examples from the displayed split in each concept's sidebar. Captions show the
study ID, split and raw activation. Selection uses activations, not outcome labels.
Each study thumbnail is stored once and shared across concepts; originals are
unchanged. Missing images and non-image modalities do not prevent exploration.
A concept with only one readable activating study shows one example.

```python
glyph.visualize()                  # two image examples per concept when available
glyph.visualize(image_examples=0)  # omit cohort images from the exported HTML
```

Configure `visualization.image_examples` (default 2) and
`visualization.thumbnail_size` (default 256 pixels). These are embedded cohort
images: share the exported file accordingly. Existing raw-input, precomputed and
pretrained workflows retain their saved association statistics and dictionary IDs.
The atlas/network builders also accept an `image_examples` payload produced by
`glyph.visualization.image_examples.collect_image_examples` for adapters with
an existing layout.

The curated demo files stay self-contained for offline use. Their GitHub Pages
copies place shared thumbnails in separate image files and load only the two
selected examples when a node is clicked. `scripts/build_demo_site.py` builds
that site from an explicit public-demo allowlist without loading models or
copying raw inputs, source files or Git history:

```bash
python scripts/build_demo_site.py --output /tmp/glyph-demo-site
```

## Wheels and statistics

Every supplied outcome estimate is retained on a concept's wheel, including dense
150-outcome wheels. OR = 1 is fixed halfway between the inner and outer rings.
An OR of exactly 1 draws no filled spoke. Lower odds ratios extend inward from 1;
higher odds ratios extend outward. The spacing is symmetric in log OR, so 0.5 and
2 occupy equal distances on opposite sides of 1. Inner/outer ring labels show the
actual reciprocal bounds; they are not labelled 0 because log OR has no finite
zero-OR endpoint. Wheel spokes use the outcome color; map nodes retain direction
stripes. Faded spokes are not significant under
the supplied Bonferroni correction. Nonsignificant outcomes remain inspectable.

At most 20 outcomes are labelled directly on their spokes. Change
`visualization.wheel_label_limit` in Glyph configuration, or `wheel_label_limit=`
on either standalone builder. Above the threshold, labels are omitted; spokes are
not omitted. Hover or keyboard focus on a spoke shows its full name, odds ratio,
95% CI, raw p-value, Bonferroni p-value and support where available. Missing
statistics are identified rather than invented. Clicking a spoke or table outcome
highlights that outcome on the map. All outcomes remain on the wheel.

The concept table lists every supplied association by effect magnitude. Export CSV
saves full stored numeric estimates, p-values, intervals and support. Very small
p-values retain scientific notation; exporting does not round them to zero.
Current Glyph-generated maps use unadjusted associations, even if optional
confounder-adjusted results were computed. The map identifies the displayed split.

## Local review examples

From a source checkout:

```bash
python -m examples.concept_explorer examples/concept_explorer.html
python -m examples.concept_explorer concept_explorer_150.html --outcomes 150
```

Open the resulting files directly in a browser; no server, CDN, fonts or external
network requests are required. The data in these examples are synthetic. They
check presentation and interactions, not clinical validity.

## Browser verification

```bash
pip install -e '.[browser-tests]'
python -m playwright install chromium
GLYPH_BROWSER_TESTS=1 python -m pytest tests/test_html_browser.py -q
```

The opt-in suite exercises 1, 4, 20, 21 and 150 outcomes, every spoke's keyboard
statistics, real hover targets, scientific p-values, CSV download, dropdown search,
selection reset, filtering/empty states, lazy connections, full-view rankings,
cursor-anchored zoom on a nonsquare viewport, dragging and mobile controls.
It also checks for JavaScript errors and external network requests. Ordinary tests
skip the browser suite unless explicitly enabled. Final review on the user's data
belongs in the subsequent end-to-end package integration session.

## Optional hosting and browser opening

On the same computer as your browser:

```bash
glyph serve /path/to/concept_map.html
```

This binds to `127.0.0.1`, chooses a free port and opens a new browser tab directly
on the viewer. It serves only the specified HTML reports, not adjacent metadata
or model files. `--port 8766` chooses a fixed port; `--no-open` only prints the URL.
Ctrl+C stops the server. A directory or list of maps opens the first map directly;
other report URLs remain available without inserting a file picker.

For Python:

```python
glyph.visualize(serve=True)  # generation + hosting are both explicitly requested
# Or, after generating HTML:
viewer = glyph.serve()
print(viewer.url)
viewer.close()
```

For SSH, run this on your **laptop**, with Glyph installed at both ends:

```bash
glyph serve --ssh rholland@torino /remote/path/to/concept_map.html
```

It starts the remote viewer, forwards a port, waits for that viewer to respond and
opens a new tab in the laptop browser. Ctrl+C closes the connection and signals
the remote viewer to stop. If the remote environment needs a particular Python,
add `--remote-python /path/to/environment/bin/python`. The default invokes the
remote `glyph` command. This command uses the system SSH client and your normal
SSH authentication; Glyph stores no SSH credentials.

A command running only on the server cannot launch the laptop's browser. For an
already-running viewer on port 8766, this macOS laptop command forwards the port
and opens the browser:

```bash
ssh -fN -o ExitOnForwardFailure=yes -L 8766:127.0.0.1:8766 rholland@torino && open http://localhost:8766
```

Remote/headless `glyph serve` skips launching a browser on the server and prints
the viewer URL instead. Local browser opening and viewer hosting are independently
optional; analysis can be run without either.
