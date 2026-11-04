# Laboratory code mapping — 28 September 2026

This mapping uses the approved `lab_loinc_source_first_proposal.csv`, accompanied
by the later STARR source audit from the same day. Original source file hashes
are recorded in `lab_loinc_mapping_provenance.json`.

There are 350 model lab IDs: 340 LOINC assignments (193 distinct codes), nine
`STANFORD_LAB` local codes, and one unresolved mixed-context pH lab. Of the LOINC
assignments, 305 match Torino by normalized name and unit; 35 are provisional.
Eighty-two LOINC assignments differ from the legacy map.

`loinc_code` contains only LOINC identifiers. `selected_code` and
`selected_code_system` preserve local codes without presenting them as LOINC.
`status`, `mapping_basis`, and `legacy_*` retain the review provenance. The old
confidence labels remain in `legacy_confidence`; `confidence` is intentionally
empty because the new review did not assign replacement confidence levels.

The later audit identifies eight conflicts with the selected proposal and one
source code for the unresolved pH lab. They remain audit evidence rather than
silent overrides. `lab_loinc_source_audit.csv` preserves the full 45-row audit.

Some codes map to several model labs. Loaders require units, an exact lab name,
or an explicit model `lab_id` to resolve these; they no longer silently select
the last dictionary entry. Units are not converted automatically. The unresolved
pH lab can be supplied only through an explicit model ID, not a fabricated LOINC.
Model weights and model lab IDs are unchanged; cached internal-ID inputs retain
their meaning. Applying revised mappings to external LOINC inputs may change
which model lab is selected.

## Verification

Glyph's focused regression suite passed 148 tests; an independently installed
wheel passed 30 extraction/mapping tests. HypothesisExtractor passed 13 lab and
mapping tests. Both mapping CSVs are byte-identical, all 350 model IDs are retained,
and the release bundle checksums validate. Weights were not changed. These
checks cover code/name/unit disambiguation, local codes, unresolved entries,
application through both loaders, and audit preservation during bundle export.

Changes are local; no Git commits, pushes or HF uploads were made.
