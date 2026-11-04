---
name: glyph-html
description: Generate or refine Glyph interactive HTML concept explorers with clickable outcome wheels and display controls from existing analysis results.
---

# Glyph interactive HTML

Use the existing renderer and saved analysis outputs. A presentation change should
not refit models, rerun association tests or silently change the concept layout.
The implementation is `glyph.visualization.interactive_concept_network`; its
shared atlas/wheel renderer comes from the CKM/CORAL viewers.

For a fitted Glyph object, call `glyph.visualize(outcome='...', split='test')`.
The default `visualization.view=simple` provides a clean map, concept detail,
outcome wheels, search and an Options menu. `visualization.view=full` retains
rankings and technical/report panels when their data are supplied.

For an already-computed layout, use:

```python
from glyph.visualization import build_interactive_concept_atlas

build_interactive_concept_atlas(
    concepts, associations, 'concept_explorer.html',
    edges=edges, title='Study concept explorer', view='simple',
)
```

Concepts contain unique `concept_id`, `x`, `y` and `modality` columns, with optional
name, interpretation, activation count/rate and embedding family. Associations
join by `concept_id` and supply outcome, OR and corrected significance information.
Use the installed function's signature for supported optional fields. Preserve
the study's stored testing family and effect estimates.

Use a neutral general style and the provided outcome labels/colors. Keep rich
reporting optional. Preserve the reference's node interactions, wheels and filters;
adapt its data interface rather than creating a separate viewer. A synthetic demo
is a presentation check, not clinical evidence. Browser-check opening the standalone
file, selecting a concept, changing outcomes/options and displaying a wheel; leave
the user's final visual/integration review for the agreed integration session.

Keep every outcome on the wheel. Label spokes directly at or below
`visualization.wheel_label_limit` (default 20); dense wheels rely on hover or
keyboard-focus tooltips showing outcome, OR, 95% CI, raw/Bonferroni p-values and
support where supplied. Do not truncate the wheel to the top outcomes.

The browser checks are in `tests/test_html_browser.py`; set
`GLYPH_BROWSER_TESTS=1` with Playwright Chromium installed to run them. They cover
1, 4, 20, 21 and 150 spokes, hover/focus statistics, exact CSV export, search,
filtering, zoom/pan, standalone operation and mobile layout. See
`docs/html_explorer.md` for the review examples and configuration.

HTML generation is opt-in: call `glyph.visualize()` or explicitly enable
`visualization.auto_generate`. Hosting is opt-in too. Use `glyph serve FILE` to
host an existing viewer and open a local browser tab, or run
`glyph serve --ssh HOST REMOTE_FILE` on the laptop to manage the remote viewer,
tunnel and local tab. Never launch a remote browser and imply it opens on the
user's laptop. The URL opens the viewer directly without a file picker. A written
report is a separate agent harness. See `docs/html_explorer.md`.

Keep OR = 1 at the fixed middle radius. Exactly 1 draws no effect; lower ORs
extend inward and higher ORs outward, using symmetric log spacing so reciprocal
effects have equal lengths. Label the actual inner/outer OR bounds.
